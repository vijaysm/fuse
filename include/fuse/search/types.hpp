#pragma once

#include <cctype>
#include <cstdlib>
#include <string>

#include <fmt/format.h>

#include "fuse/base/result.hpp"

/// fuse::search -- a spatial-index facade over ArborX (its scalable BVH and
/// point-cloud interpolants; see cmake/FuseDependencies.cmake's
/// fuse_resolve_arborx()), the way fuse::linalg is a facade over its dense
/// solver: nothing outside src/search/*.cpp names an ArborX type.
///
/// LAYERING (docs/dev/layering.md, tools/check-layering.py). This layer sits
/// BELOW fuse::backend so that fuse::backend can depend on it, which is the
/// opposite direction from fuse::partition's mesh_adapter.hpp (ABOVE
/// backend). That placement is exactly why this header, and every other
/// public header in fuse::search, never includes `fuse/backend/...`: the
/// `fuse::MeshView` the adapter builds boxes from (mesh_adapter.hpp) is
/// consumed through a structural (duck-typed) template constraint instead of
/// a named dependency -- see that file's header comment for the full
/// reasoning.
namespace fuse::search {

/// A point in physical space. Independent of `fuse::Point`
/// (fuse/backend/mesh_view.hpp) on purpose -- naming that type here would be
/// the upward dependency this layer's placement forbids. Same convention:
/// always three components, a 1D or 2D dataset's unused coordinates are
/// exactly zero (geometry.hpp's `classify_dim` relies on that).
struct Point {
  double x = 0.0, y = 0.0, z = 0.0;
};

/// An axis-aligned bounding box, `lo` and `hi` diagonal corners. No
/// orientation, no cell topology -- exactly what a broad-phase spatial index
/// needs and nothing else (mesh_adapter.hpp builds these from a mesh
/// element's simplex decomposition).
struct Box {
  Point lo;
  Point hi;
};

/// The relative tolerance `geometry.hpp`'s `padded()` grows a box by, so a
/// point sitting exactly on a shared face between two adjacent elements is
/// found inside BOTH of their (padded) boxes rather than falling through a
/// floating-point gap. `1e-8` is a few ULPs looser than double round-off for
/// a mesh whose coordinates are O(1)-O(1e6); `absolute_floor` guards a
/// degenerate box (zero diagonal, e.g. a single point treated as a box)
/// where a relative term alone would pad by exactly zero.
struct Tolerance {
  double relative = 1e-8;
  double absolute_floor = 1e-12;
};

/// A communicator handle that names no MPI type, so this header (and every
/// other public header in this layer) stays free of `mpi.h` -- the same
/// boundary convention `fuse::partition::CommHandle` and
/// `fuse::transport::MpiTransport` use (`MPI_Comm_c2f`/`MPI_Comm_f2c`).
/// Defined independently here rather than reused from `fuse::partition`:
/// partition sits ABOVE both backend and search in the layer ordering, so
/// depending on it would be the same forbidden upward reach `mesh_adapter.hpp`
/// avoids for `fuse::backend`. `kNullComm` (-1) means "no communicator":
/// exactly one participant, no MPI symbol touched at all. Negative, not 0:
/// OpenMPI's `MPI_Comm_c2f` hands out small sequential NON-NEGATIVE Fortran
/// handles with no reserved always-invalid value (even `MPI_COMM_WORLD`
/// itself can legitimately come back as `0`), so `0` would silently collide
/// with a real communicator and get treated as "no MPI".
using CommHandle = int;
inline constexpr CommHandle kNullComm = -1;

/// Which Kokkos execution space `BoxIndex::build`/`query_points_distributed`
/// run on. Mirrors `fuse::RunnerConfig::RunnerExecutionSpace`
/// (include/fuse/runtime/runner.hpp) in both shape and the environment
/// variable it reads, without depending on `fuse::runtime`: this layer sits
/// below `backend`, which sits below `runtime` (docs/dev/layering.md), so an
/// `#include "fuse/runtime/runner.hpp"` here would be an upward dependency
/// this layer's placement forbids. `Host` is `Kokkos::DefaultHostExecutionSpace`
/// / `Kokkos::HostSpace`, byte-identical to this facade's behaviour before
/// this field existed; `Device` is `Kokkos::DefaultExecutionSpace` and its
/// `memory_space` (the same pair `fuse::native::MeshArrays`'s default
/// template argument already names), which on a build with no CUDA/HIP/SYCL
/// backend compiled in is `Kokkos::DefaultExecutionSpace == Kokkos::DefaultHostExecutionSpace`
/// -- i.e. `Device` degrades to exactly `Host`, never a refusal.
enum class ExecutionSpace { Default, Host, Device };

/// `Default` resolved from the `FUSE_EXECUTION_SPACE` process environment
/// variable, read and parsed identically to
/// `RunnerConfig::resolved_execution_space()`: `device` selects `Device`,
/// `host`, unset or empty selects `Host`, anything else is `E-SEARCH-SPACE`.
/// `Host` or `Device` passed in is returned unchanged -- an explicit
/// argument on the search API is never overridden by the environment,
/// exactly as the Runner's own convention promises.
[[nodiscard]] inline Result<ExecutionSpace> resolve_execution_space(
    ExecutionSpace requested = ExecutionSpace::Default) {
  if (requested != ExecutionSpace::Default) return requested;
  const char* env = std::getenv("FUSE_EXECUTION_SPACE");
  std::string v = env == nullptr ? std::string{} : std::string{env};
  for (char& ch : v) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  if (v.empty() || v == "host") return ExecutionSpace::Host;
  if (v == "device") return ExecutionSpace::Device;
  return Failure{"E-SEARCH-SPACE",
                fmt::format("FUSE_EXECUTION_SPACE='{}' is not a search execution space; "
                           "use 'host' or 'device' (or unset it), or pass an explicit "
                           "fuse::search::ExecutionSpace",
                           env)};
}

}  // namespace fuse::search
