// The only translation unit in fuse::native's halo machinery that names an
// MPI type (the box allgather); everything declared in native_halo.hpp
// without one is a pure function, exercised at tier 0 by an in-process
// emulation of several ranks (test/backend/native_halo_test.cpp). Same
// isolation convention native_migration.cpp's own header comment states.

#include "fuse/backend/native_halo.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

#include <fmt/format.h>

#include "fuse/backend/native/mesh_pack_device.hpp"
#include "fuse/backend/native/mesh_merge_device.hpp"
#include "fuse/base/config.hpp"
#include "fuse/base/error.hpp"

#if FUSE_HAVE_MPI
#include <mpi.h>
#endif

namespace fuse::native {

namespace {

/// D73/D5 ROUND 3 (owner request, 2026-10-18, host measurement): off by
/// default, in the style of this tree's other per-rank trace env vars
/// (FUSE_COLLECTIVE_TRACE, FUSE_NEWTON_DEBUG) -- cached once per process,
/// never re-read per call, so the getenv cost is paid at most once. Writes
/// to stderr, not stdout, matching the convention the Runner's own debug
/// traces already use (never interleaved with a test harness's own
/// stdout-captured assertions).
[[nodiscard]] bool remap_halo_trace_enabled() {
  static const bool enabled = std::getenv("FUSE_REMAP_HALO_TRACE") != nullptr;
  return enabled;
}

/// The axis-aligned union of `boxes`, or an explicitly empty box (all
/// bounds at +inf/-inf, printed as such rather than as a plausible-looking
/// zero) when `boxes` is empty -- an empty box is itself diagnostic
/// evidence (hypothesis (b)/(c) in the owner's own round-3 request: did
/// this rank have ANY owned/adopted cells to bound at all).
[[nodiscard]] std::string trace_box_union(const std::vector<Box3>& boxes) {
  if (boxes.empty()) return "EMPTY (no cells)";
  Box3 u = boxes.front();
  for (std::size_t i = 1; i < boxes.size(); ++i) {
    for (int k = 0; k < 3; ++k) {
      u.lo[k] = std::min(u.lo[k], boxes[i].lo[k]);
      u.hi[k] = std::max(u.hi[k], boxes[i].hi[k]);
    }
  }
  return fmt::format("[{:.6g},{:.6g}]x[{:.6g},{:.6g}]x[{:.6g},{:.6g}]", u.lo[0], u.hi[0], u.lo[1],
                      u.hi[1], u.lo[2], u.hi[2]);
}

#if FUSE_HAVE_MPI
MPI_Comm resolve(CommHandle comm) {
  FUSE_ASSERT(comm != kNullComm,
             "native halo collective called with kNullComm; build_native_remap_halo must "
             "short-circuit locally before reaching here");
  return MPI_Comm_f2c(comm);
}
#endif

// ---- byte-buffer read helper (unpack only; pack now goes through
// `fuse::native::device::pack_fanout`, mesh_pack_device.hpp, matching
// native_migration.cpp's own post-wiring shape) ----------------------------

template <class T>
T read_pod(const std::vector<std::byte>& buf, std::size_t& pos) {
  static_assert(std::is_trivially_copyable_v<T>);
  T v{};
  std::memcpy(&v, buf.data() + pos, sizeof(T));
  pos += sizeof(T);
  return v;
}

/// See native_migration.cpp's identical helper's comment: the one host
/// round trip a halo pack still pays, for the wire bytes themselves, not
/// the mesh's own geometry.
std::vector<std::byte> to_host_bytes(const Kokkos::View<std::byte*, NativeMeshView::DeviceMemSpace>& v) {
  auto mirror = Kokkos::create_mirror_view(v);
  Kokkos::deep_copy(mirror, v);
  std::vector<std::byte> out(mirror.extent(0));
  if (!out.empty()) std::memcpy(out.data(), mirror.data(), mirror.extent(0));
  return out;
}

/// See native_migration.cpp's identical helper's comment: wraps every
/// host byte buffer as a `Kokkos::View<std::byte*, Kokkos::HostSpace>` so
/// `merge_halo_cells` below can call straight into
/// `fuse::native::device::merge_halo_cells_device<Kokkos::HostSpace>`
/// instead of maintaining its own by-hand sequential parser for the wire-
/// format-v2 header `device::pack_fanout` now always writes.
std::vector<Kokkos::View<std::byte*, Kokkos::HostSpace>> to_host_views(
    const std::vector<std::vector<std::byte>>& received) {
  std::vector<Kokkos::View<std::byte*, Kokkos::HostSpace>> views(received.size());
  for (std::size_t i = 0; i < received.size(); ++i) {
    views[i] = Kokkos::View<std::byte*, Kokkos::HostSpace>(
        Kokkos::view_alloc("merge_halo_cells::host_buf", Kokkos::WithoutInitializing),
        received[i].size());
    if (!received[i].empty()) {
      std::memcpy(views[i].data(), received[i].data(), received[i].size());
    }
  }
  return views;
}

}  // namespace

Box3 cell_box(const NativeMeshView& mesh, std::size_t c) {
  const auto& verts = mesh.cell_vertices(c);
  FUSE_ASSERT(!verts.empty(), "cell_box: cell {} has no corners", c);
  Box3 b;
  const int v0 = verts[0];
  for (int k = 0; k < 3; ++k) {
    b.lo[static_cast<std::size_t>(k)] = mesh.vertex_x(static_cast<std::size_t>(v0), k);
    b.hi[static_cast<std::size_t>(k)] = b.lo[static_cast<std::size_t>(k)];
  }
  for (std::size_t i = 1; i < verts.size(); ++i) {
    for (int k = 0; k < 3; ++k) {
      const double x = mesh.vertex_x(static_cast<std::size_t>(verts[i]), k);
      b.lo[static_cast<std::size_t>(k)] = std::min(b.lo[static_cast<std::size_t>(k)], x);
      b.hi[static_cast<std::size_t>(k)] = std::max(b.hi[static_cast<std::size_t>(k)], x);
    }
  }
  return b;
}

namespace {
// A margin, not a bare `<`: two boxes that only TOUCH along a face (one's
// `hi` exactly equal to the other's `lo` on some axis) have an actual
// intersection of measure zero -- real geometric cells are CONTAINED in
// their bounding boxes, so a zero-measure box intersection proves the real
// cells cannot overlap with positive area either, regardless of their
// shape. Counting a touch as "overlap" (a bare `<` exclusion test) costs
// nothing for correctness -- an extra adopted cell with zero true overlap
// just sits unused in the supermesh -- but it is NOT what "needs a copy"
// means, and it is measurably wrong at an EXACTLY aligned partition
// boundary (measured: a 16x16/24x24 quad-grid pair at 2 ranks, whose row
// split lands on y = 0.5 for both meshes bit-identically since 8/16 and
// 12/24 both reduce to the exactly-representable binary fraction 0.5,
// adopted one full boundary row -- 16 cells -- on EACH rank before this
// margin was added, zero after). The margin is an absolute tolerance in
// the mesh's own coordinate units, far below any cell size this track
// targets (quad cells of width >= 1/24) and far above double-precision
// rounding in a coordinate computed as `index / count` for the counts
// these generators take.
constexpr double kTouchTol = 1e-9;
// Below this width, an axis is "flat" for a box -- not a partition boundary
// artifact, but the mesh's own embedding: EVERY cell of a planar (dimension
// 2) mesh has z `lo == hi == 0` exactly, by construction (native_generators
// .cpp sets every vertex's z to 0.0 for a 2D mesh), so a bare `kTouchTol`
// margin applied uninconditionally would exclude EVERY pair on that axis
// alone -- measured: it did, and dropped every single cross-rank overlap at
// 3 ranks to zero. The margin only makes sense, and is only applied, on an
// axis where at least one of the two boxes has real extent.
constexpr double kDegenerateWidth = 1e-12;

[[nodiscard]] bool boxes_overlap(const Box3& a, const Box3& b) {
  for (int k = 0; k < 3; ++k) {
    const std::size_t kk = static_cast<std::size_t>(k);
    const bool degenerate =
        (a.hi[kk] - a.lo[kk] < kDegenerateWidth) && (b.hi[kk] - b.lo[kk] < kDegenerateWidth);
    const double tol = degenerate ? 0.0 : kTouchTol;
    if (a.hi[kk] < b.lo[kk] + tol || b.hi[kk] < a.lo[kk] + tol) {
      return false;
    }
  }
  return true;
}
}  // namespace

std::vector<std::vector<int>> covering_set_destinations(
    const std::vector<Box3>& my_owned_src_boxes, int my_rank,
    const std::vector<Box3>& all_target_boxes, const std::vector<int>& owner_of_target_box) {
  FUSE_ASSERT(all_target_boxes.size() == owner_of_target_box.size(),
             "covering_set_destinations: {} target boxes but {} owners", all_target_boxes.size(),
             owner_of_target_box.size());
  std::vector<std::vector<int>> result(my_owned_src_boxes.size());
  for (std::size_t i = 0; i < my_owned_src_boxes.size(); ++i) {
    std::vector<int> dests;
    for (std::size_t t = 0; t < all_target_boxes.size(); ++t) {
      const int owner = owner_of_target_box[t];
      if (owner == my_rank) continue;  // already local to its own target
      if (!boxes_overlap(my_owned_src_boxes[i], all_target_boxes[t])) continue;
      if (std::find(dests.begin(), dests.end(), owner) == dests.end()) dests.push_back(owner);
    }
    std::sort(dests.begin(), dests.end());
    result[i] = std::move(dests);
  }
  return result;
}

std::vector<std::size_t> owned_block_index(const NativeMeshView& mesh) {
  const int dim = mesh.dimension();
  const std::size_t n = mesh.entity_count(dim);
  std::vector<std::size_t> idx(n, static_cast<std::size_t>(-1));
  std::size_t running = 0;
  for (std::size_t c = 0; c < n; ++c) {
    if (mesh.is_owned(dim, c)) idx[c] = running++;
  }
  return idx;
}

/// The device-resident pack both `pack_halo_cells` (below, host-byte
/// return for the in-process emulation technique native_halo_test.cpp
/// uses) and `build_native_remap_halo` (device buffers straight into
/// `exchange_migration_buffers<Space>`) share -- see
/// `pack_for_migration_device`'s identical split in native_migration.cpp.
std::vector<Kokkos::View<std::byte*, NativeMeshView::DeviceMemSpace>> pack_halo_cells_device(
    const NativeMeshView& mesh, const std::vector<std::vector<std::size_t>>& send_by_local_index,
    const std::vector<std::size_t>& owned_block_index, int my_rank, int num_ranks) {
  const int dim = mesh.dimension();
  const std::size_t n = mesh.cell_count();

  // Flatten the per-destination send lists into one (local cell index,
  // destination rank) pair per send -- the domain `device::pack_fanout`
  // iterates. A halo copy, unlike a migrated cell, may name the SAME
  // owned cell for SEVERAL destinations (a source cell overlapping more
  // than one target rank's covering set), which is exactly why the
  // single-destination `device::pack_to` (wired into `pack_for_migration`)
  // cannot serve this caller.
  std::vector<int> cell_of_send_host;
  std::vector<int> dest_of_send_host;
  for (int r = 0; r < num_ranks; ++r) {
    for (const std::size_t c : send_by_local_index[static_cast<std::size_t>(r)]) {
      FUSE_ASSERT(mesh.is_owned(dim, c),
                 "pack_halo_cells: cell {} is being sent as a halo copy but this rank does not "
                 "own it",
                 c);
      cell_of_send_host.push_back(static_cast<int>(c));
      dest_of_send_host.push_back(r);
    }
  }
  const std::size_t n_sends = cell_of_send_host.size();

  using Space = NativeMeshView::DeviceMemSpace;
  Kokkos::View<int*, Kokkos::HostSpace> cell_h(
      Kokkos::view_alloc("pack_halo_cells::cell_h", Kokkos::WithoutInitializing), n_sends);
  Kokkos::View<int*, Kokkos::HostSpace> dest_h(
      Kokkos::view_alloc("pack_halo_cells::dest_h", Kokkos::WithoutInitializing), n_sends);
  for (std::size_t i = 0; i < n_sends; ++i) {
    cell_h(i) = cell_of_send_host[i];
    dest_h(i) = dest_of_send_host[i];
  }
  Kokkos::View<int*, Space> cell_d(
      Kokkos::view_alloc("pack_halo_cells::cell_d", Kokkos::WithoutInitializing), n_sends);
  Kokkos::View<int*, Space> dest_d(
      Kokkos::view_alloc("pack_halo_cells::dest_d", Kokkos::WithoutInitializing), n_sends);
  Kokkos::deep_copy(cell_d, cell_h);
  Kokkos::deep_copy(dest_d, dest_h);

  // `owner_rank` is one constant (`my_rank`: a halo cell is COPIED, never
  // migrated) and `owned_block_idx_of_cell` is `owned_block_index` itself
  // (already indexed by local cell), both per LOCAL CELL (size `n`), not
  // per send -- `device::pack_fanout`'s own convention (its header
  // comment), chosen so neither array needs a second indirection through
  // the send list.
  Kokkos::View<std::int32_t*, Kokkos::HostSpace> owner_h("pack_halo_cells::owner_h", n);
  Kokkos::View<std::int64_t*, Kokkos::HostSpace> owned_idx_h("pack_halo_cells::owned_idx_h", n);
  for (std::size_t c = 0; c < n; ++c) {
    owner_h(c) = my_rank;
    owned_idx_h(c) = static_cast<std::int64_t>(owned_block_index[c]);
  }
  Kokkos::View<std::int32_t*, Space> owner_d(
      Kokkos::view_alloc("pack_halo_cells::owner_d", Kokkos::WithoutInitializing), n);
  Kokkos::View<std::int64_t*, Space> owned_idx_d(
      Kokkos::view_alloc("pack_halo_cells::owned_idx_d", Kokkos::WithoutInitializing), n);
  Kokkos::deep_copy(owner_d, owner_h);
  Kokkos::deep_copy(owned_idx_d, owned_idx_h);

  return device::pack_fanout<true, Space>(mesh.device_arrays(), cell_d, dest_d, num_ranks,
                                          owner_d, owned_idx_d);
}

std::vector<std::vector<std::byte>> pack_halo_cells(
    const NativeMeshView& mesh, const std::vector<std::vector<std::size_t>>& send_by_local_index,
    const std::vector<std::size_t>& owned_block_index, int my_rank, int num_ranks) {
  auto packed =
      pack_halo_cells_device(mesh, send_by_local_index, owned_block_index, my_rank, num_ranks);
  std::vector<std::vector<std::byte>> result(packed.size());
  for (std::size_t r = 0; r < packed.size(); ++r) result[r] = to_host_bytes(packed[r]);
  return result;
}

/// The device-resident merge both `merge_halo_cells` (below, the public
/// host `MeshArrays<HostSpace>` + `std::vector<int>`/`std::vector
/// <std::int64_t>` signature native_halo_test.cpp already calls) and
/// `build_native_remap_halo` (device buffers end to end) share --
/// `device::merge_halo_cells_device<Space>` itself has no refusal path
/// (no existing test's wire data is malformed; see this track's report),
/// so this wrapper's only job is converting `Kokkos::View<std::int32_t*,
/// Space>`/`Kokkos::View<std::int64_t*, Space>` GhostPattern outputs to
/// `std::vector<int>`/`std::vector<std::int64_t>` for callers that have
/// not themselves moved to the device-resident GhostPattern views.
template <class Space>
MeshArrays<Space> merge_halo_cells_to_vectors(
    const MeshArrays<Space>& source, const std::vector<Kokkos::View<std::byte*, Space>>& received,
    int my_rank, std::vector<int>* owner_rank, std::vector<std::int64_t>* owner_owned_index) {
  (void)my_rank;
  Kokkos::View<std::int32_t*, Space> owner_rank_d;
  Kokkos::View<std::int64_t*, Space> owned_idx_d;
  MeshArrays<Space> out =
      device::merge_halo_cells_device<Space>(source, received, my_rank, &owner_rank_d, &owned_idx_d);

  auto owner_h = Kokkos::create_mirror_view(owner_rank_d);
  Kokkos::deep_copy(owner_h, owner_rank_d);
  auto idx_h = Kokkos::create_mirror_view(owned_idx_d);
  Kokkos::deep_copy(idx_h, owned_idx_d);

  const std::size_t n_new = owner_h.extent(0);
  owner_rank->resize(n_new);
  owner_owned_index->resize(n_new);
  for (std::size_t i = 0; i < n_new; ++i) {
    (*owner_rank)[i] = static_cast<int>(owner_h(i));
    (*owner_owned_index)[i] = idx_h(i);
  }
  return out;
}

Result<MeshArrays<Kokkos::HostSpace>> merge_halo_cells(
    const NativeMeshView& source, const std::vector<std::vector<std::byte>>& received,
    int my_rank, std::vector<int>* owner_rank, std::vector<std::int64_t>* owner_owned_index) {
  FUSE_ASSERT(owner_rank != nullptr && owner_owned_index != nullptr,
             "merge_halo_cells: both output vectors are required");
  return merge_halo_cells_to_vectors<Kokkos::HostSpace>(
      to_memory_space<Kokkos::HostSpace>(source.device_arrays()), to_host_views(received),
      my_rank, owner_rank, owner_owned_index);
}

namespace {

#if FUSE_HAVE_MPI
/// `MPI_Allgather` of counts, then `MPI_Allgatherv` of the flattened
/// `(lo[3], hi[3])` doubles -- the same two-phase shape
/// `resolve_vertex_ownership` (native_migration.cpp) already uses for a
/// variable-length allgather. Returns every rank's target boxes
/// concatenated, plus which rank each one came from.
void allgather_boxes(const std::vector<Box3>& my_boxes, CommHandle comm, int num_ranks,
                    std::vector<Box3>* all_boxes, std::vector<int>* owner_of) {
  MPI_Comm c = resolve(comm);
  const int local_count = static_cast<int>(my_boxes.size());
  std::vector<int> counts(static_cast<std::size_t>(num_ranks));
  MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, c);

  std::vector<int> displs(static_cast<std::size_t>(num_ranks));
  int total = 0;
  for (int r = 0; r < num_ranks; ++r) {
    displs[static_cast<std::size_t>(r)] = total * 6;
    total += counts[static_cast<std::size_t>(r)];
  }
  std::vector<double> local_flat(static_cast<std::size_t>(local_count) * 6);
  for (int i = 0; i < local_count; ++i) {
    const Box3& b = my_boxes[static_cast<std::size_t>(i)];
    for (int k = 0; k < 3; ++k) {
      local_flat[static_cast<std::size_t>(i) * 6 + static_cast<std::size_t>(k)] =
          b.lo[static_cast<std::size_t>(k)];
      local_flat[static_cast<std::size_t>(i) * 6 + 3 + static_cast<std::size_t>(k)] =
          b.hi[static_cast<std::size_t>(k)];
    }
  }
  std::vector<int> recv_counts(static_cast<std::size_t>(num_ranks));
  for (int r = 0; r < num_ranks; ++r) recv_counts[static_cast<std::size_t>(r)] =
                                          counts[static_cast<std::size_t>(r)] * 6;
  std::vector<double> flat(static_cast<std::size_t>(total) * 6);
  MPI_Allgatherv(local_flat.data(), local_count * 6, MPI_DOUBLE, flat.data(),
                recv_counts.data(), displs.data(), MPI_DOUBLE, c);

  all_boxes->clear();
  owner_of->clear();
  all_boxes->reserve(static_cast<std::size_t>(total));
  owner_of->reserve(static_cast<std::size_t>(total));
  for (int r = 0; r < num_ranks; ++r) {
    for (int i = 0; i < counts[static_cast<std::size_t>(r)]; ++i) {
      const std::size_t base = static_cast<std::size_t>(displs[static_cast<std::size_t>(r)]) +
                              static_cast<std::size_t>(i) * 6;
      Box3 b;
      for (int k = 0; k < 3; ++k) {
        b.lo[static_cast<std::size_t>(k)] = flat[base + static_cast<std::size_t>(k)];
        b.hi[static_cast<std::size_t>(k)] = flat[base + 3 + static_cast<std::size_t>(k)];
      }
      all_boxes->push_back(b);
      owner_of->push_back(r);
    }
  }
}
#endif  // FUSE_HAVE_MPI

}  // namespace

Result<std::size_t> build_native_remap_halo(NativeMeshBackend& backend, MeshId source,
                                            MeshId target, int dim, ComponentId placement) {
  // The identity case this two-placement form's own logic already
  // reduces to correctly when the two placements coincide -- see this
  // function's OWN two-placement overload's header comment
  // (native_halo.hpp) for why `source_placement` had to become a
  // separate parameter rather than reusing `placement` for both sides.
  return build_native_remap_halo(backend, source, target, dim, placement, placement);
}

Result<std::size_t> build_native_remap_halo(NativeMeshBackend& backend, MeshId source,
                                            MeshId target, int dim, ComponentId source_placement,
                                            ComponentId placement) {
  const int ranks = backend.communicator_size();
  if (ranks <= 1 || backend.communicator() == kNullComm || source.value() == target.value()) {
    const Result<void> rec = backend.commit_remap_halo(source, placement, nullptr, target, dim,
                                                       fuse::linalg::GhostPattern{});
    if (!rec.ok()) return rec.error();
    return std::size_t{0};
  }

#if !FUSE_HAVE_MPI
  return Failure{"E-MESH-NO-MPI",
               fmt::format("the native backend reports a communicator of {} ranks, but this "
                          "build has no MPI",
                          ranks)};
#else
  // The same guard MoabMeshBackend::set_communicator() makes: a build WITH
  // MPI whose process never called MPI_Init (every tier-0 test binary in a
  // tier-1 build tree) must refuse here, not reach the MPI_Allgather below
  // with a handle MPI_Comm_f2c cannot interpret. Measured 2026-10-11: the
  // tier-0 row verification.brusselator_species_core_decomposition aborted
  // inside internal_Allgather ("Invalid communicator") in the user's MPI
  // build, where the same case refuses E-MESH-NO-MPI in a build without MPI.
  {
    int initialized = 0;
    if (MPI_Initialized(&initialized) != MPI_SUCCESS || initialized == 0) {
      return Failure{"E-MESH-MPIINIT",
                   fmt::format("the native backend reports a communicator of {} ranks, but "
                              "MPI is not initialised in this process; the covering-set "
                              "halo is a collective and cannot be built",
                              ranks)};
    }
  }
  // With MPI initialised (checked above), `ranks` is `backend.communicator_size()`, which is pure local
  // bookkeeping (`communicator_ == kNullComm ? 1 : num_ranks_`) -- it says
  // nothing about what `backend.communicator()` actually resolves to.
  // `fuse_test_main.cpp` calls `MPI_Init` unconditionally in every build
  // with FUSE_HAVE_MPI, even run as a single bare process with no
  // `mpiexec`, so `MPI_Initialized` is true and `MPI_Comm_f2c` on a
  // placeholder handle (every tier-0 test that sets one without a real
  // multi-rank launch, documented at each call site as "never
  // dereferenced") does not crash -- it silently resolves to whatever that
  // handle happens to mean on a singleton MPI (commonly a 1-rank
  // communicator). Proceeding into the real exchange below with a `ranks`
  // the resolved communicator cannot back then failed deep inside
  // `exchange_migration_buffers`'s own invariant check instead of here,
  // cleanly. Comparing the two directly is the actual precondition for
  // everything below.
  const MPI_Comm resolved = resolve(backend.communicator());
  int real_size = 1;
  MPI_Comm_size(resolved, &real_size);
  // NM-5: the sibling check just above this one validates communicator
  // SIZE but not RANK -- `backend.rank()` is what every self-exclusion
  // and wire-owner decision below uses, while box owners and the
  // Allgather/Alltoall below address participants by the COMMUNICATOR's
  // own rank numbering. The two coincide for a world-sized backend
  // (every tier-0/tier-1 test here), but a backend adopting a
  // SUB-communicator (a future caller, not in this tree today) could
  // disagree, silently misrouting every exchange below to the wrong
  // peer instead of refusing.
  int real_rank = 0;
  MPI_Comm_rank(resolved, &real_rank);
  const int my_rank = backend.rank();
  if (real_size != ranks || real_rank != my_rank) {
    return Failure{"E-MESH-NO-MPI",
                 fmt::format("the native backend reports rank {} of {}, but its actual MPI "
                            "communicator reports rank {} of {}; this backend was not given "
                            "the real multi-rank communicator it is supposed to decompose "
                            "over",
                            my_rank, ranks, real_rank, real_size)};
  }

  // LOCAL validity, computed before any collective: resolving both views
  // (native_view can refuse per placement with E-MESH-ID/E-MESH-PLACEMENT)
  // and comparing their dimension (E-REMAP-DIM, data-dependent) are both
  // rank-local -- reduced here with every other rank, mirroring migrate()'s
  // own pattern (native_migration.cpp), BEFORE `allgather_boxes` below, so
  // no rank can return its own refusal while a peer proceeds into that
  // collective.
  // KNOWN_ISSUES D5 HALO BRIDGING (round 3, 2026-10-18): `source` resolves
  // through `source_placement` -- NOT `placement` -- because `source`'s
  // own owned cells live under its OWN component's decomposition, which
  // is generally a DIFFERENT processor set than the target's. Resolving
  // it through `placement` (the target's own) used to see the EMPTY view
  // `register_default_placements()` hands every component for a mesh it
  // does not itself decompose -- this function's own header comment has
  // the full measured account.
  auto src_r = backend.native_view(source, source_placement);
  auto tgt_r = backend.native_view(target, placement);
  Result<void> local_check = Result<void>::success();
  if (!src_r.ok()) {
    local_check = src_r.error();
  } else if (!tgt_r.ok()) {
    local_check = tgt_r.error();
  } else if (src_r.value()->dimension() != dim || tgt_r.value()->dimension() != dim) {
    local_check = Failure{"E-REMAP-DIM",
                 fmt::format("build_native_remap_halo: dimension {} was asked for, but the "
                            "source view reports {} and the target view {}",
                            dim, src_r.value()->dimension(), tgt_r.value()->dimension())};
  }
  {
    const int locally_ok = local_check.ok() ? 1 : 0;
    int globally_ok = locally_ok;
    MPI_Allreduce(&locally_ok, &globally_ok, 1, MPI_INT, MPI_MIN, resolved);
    if (globally_ok == 0) {
      if (!local_check.ok()) return local_check.error();
      return Failure{"E-MESH-HALO-PEER-REFUSED",
                   fmt::format("rank {} is proceeding, but at least one other rank in this "
                              "communicator refused its own source/target resolution or "
                              "dimension check; every rank refuses together so none hangs "
                              "waiting on the covering-set exchange below", my_rank)};
    }
  }
  const NativeMeshView& src = *src_r.value();
  const NativeMeshView& tgt = *tgt_r.value();

  const CommHandle comm = backend.communicator();

  // This rank's own target boxes (one per OWNED target cell -- selective,
  // per-cell, not one union box for the whole slab).
  std::vector<Box3> my_target_boxes;
  for (std::size_t c = 0; c < tgt.entity_count(dim); ++c) {
    if (tgt.is_owned(dim, c)) my_target_boxes.push_back(cell_box(tgt, c));
  }

  std::vector<Box3> all_target_boxes;
  std::vector<int> owner_of_target_box;
  allgather_boxes(my_target_boxes, comm, ranks, &all_target_boxes, &owner_of_target_box);

  // This rank's own owned source cells, in ascending local-index (= owned
  // block) order, with the LOCAL index each one came from so the pack
  // step below can read its data back.
  const std::vector<std::size_t> owned_idx = owned_block_index(src);
  std::vector<Box3> my_owned_src_boxes;
  std::vector<std::size_t> owned_local_index;
  for (std::size_t c = 0; c < src.entity_count(dim); ++c) {
    if (src.is_owned(dim, c)) {
      my_owned_src_boxes.push_back(cell_box(src, c));
      owned_local_index.push_back(c);
    }
  }

  if (remap_halo_trace_enabled()) {
    fmt::print(stderr,
               "FUSE_REMAP_HALO_TRACE rank={} source_mesh={} target_mesh={} dim={} "
               "owned_source_cells={} source_box={} target_cells={} target_box={}\n",
               my_rank, source.value(), target.value(), dim, my_owned_src_boxes.size(),
               trace_box_union(my_owned_src_boxes), my_target_boxes.size(),
               trace_box_union(my_target_boxes));
  }

  const std::vector<std::vector<int>> dest_for_owned =
      covering_set_destinations(my_owned_src_boxes, my_rank, all_target_boxes,
                               owner_of_target_box);

  std::vector<std::vector<std::size_t>> send_by_local_index(static_cast<std::size_t>(ranks));
  for (std::size_t i = 0; i < dest_for_owned.size(); ++i) {
    for (int r : dest_for_owned[i]) {
      send_by_local_index[static_cast<std::size_t>(r)].push_back(owned_local_index[i]);
    }
  }

  // DEVICE-RESIDENT END TO END (Migration track, 2026-10-15): `send` is
  // `Space` buffers straight from `device::pack_fanout` (the still-public
  // `pack_halo_cells` stages to host bytes only for the in-process
  // emulation test technique, same split `migrate`'s own comment names
  // for `pack_for_migration`), `exchange_migration_buffers<Space>` moves
  // them without this function touching a host mirror, and
  // `device::merge_halo_cells_device<Space>` merges `src.device_arrays()`
  // against them directly -- no `NativeMeshView` host-serving accessor
  // (`cell_vertices`, `vertex_x`, `cell_faces`) is called anywhere in this
  // path, so no host mirror is materialized by a halo round either.
  using Space = NativeMeshView::DeviceMemSpace;
  const std::vector<Kokkos::View<std::byte*, Space>> send =
      pack_halo_cells_device(src, send_by_local_index, owned_idx, my_rank, ranks);
  const std::vector<Kokkos::View<std::byte*, Space>> received =
      exchange_migration_buffers<Space>(send, comm);

  // KNOWN_ISSUES.md D73 (RemapHalo track round 5, host measurement
  // 2026-10-19): everything from here through the post-exchange verdict
  // Allreduce below is LOCAL, not collective (`exchange_migration_buffers`
  // just above is the last real MPI call before it) -- but it was not
  // EXCEPTION-safe across that gap: a C++ exception thrown by the merge
  // or the rebuilt view's own construction unwinds straight out of this
  // function, skipping the post-exchange Allreduce entirely. Measured on
  // the owner's host: rank 1 (zero owned source cells, receiving 4 from
  // rank 0 -- the "rank hosts only the target" case every disjoint-
  // placement run has) threw inside `device::merge_halo_cells_device`
  // (the `cell_vertex_offsets` empty-mesh defect fixed in mesh_merge_
  // device.hpp this same round) and never reached its own Allreduce call,
  // while rank 0 (which merged cleanly) reached ITS OWN call and blocked
  // forever waiting for a peer that had already exited the function --
  // exactly the standing rule this violates ("a collective path never
  // returns on a locally computed condition"), here by an uncaught
  // exception rather than an early return. Caught here, between the two
  // Allreduces, and folded into `post_check` so the EXISTING post-exchange
  // Allreduce below still runs on every rank regardless of which (if any)
  // threw locally.
  std::unique_ptr<NativeMeshView> new_source_view;
  Result<void> post_check = Result<void>::success();
  std::vector<int> owner_rank;
  std::vector<std::int64_t> owner_owned_index;
  try {
    Kokkos::View<std::int32_t*, Space> owner_rank_d;
    Kokkos::View<std::int64_t*, Space> owned_idx_d;
    MeshArrays<Space> merged_arrays = device::merge_halo_cells_device<Space>(
        src.device_arrays(), received, my_rank, &owner_rank_d, &owned_idx_d);
    auto owner_rank_h = Kokkos::create_mirror_view(owner_rank_d);
    Kokkos::deep_copy(owner_rank_h, owner_rank_d);
    auto owned_idx_h = Kokkos::create_mirror_view(owned_idx_d);
    Kokkos::deep_copy(owned_idx_h, owned_idx_d);
    const std::size_t n_new = owner_rank_h.extent(0);
    owner_rank.resize(n_new);
    owner_owned_index.resize(n_new);
    for (std::size_t i = 0; i < n_new; ++i) {
      owner_rank[i] = static_cast<int>(owner_rank_h(i));
      owner_owned_index[i] = owned_idx_h(i);
    }

    // POST-EXCHANGE verdict: `create<Space>` can still fail on data this
    // rank alone received (a connectivity refusal from the rebuilt view) --
    // reconciled here, BEFORE returning, the same way the pre-exchange
    // verdict above was: the caller of `build_native_remap_halo`
    // (`build_weights`, ultimately) performs its own collectives against
    // the triple this call just committed, and a rank that returned success
    // here while a peer returned failure would desync that caller's next
    // collective. `device::merge_halo_cells_device` itself has no refusal
    // path of its own (this track's report names the one case it does not
    // validate -- a Polyhedron's cross-cell face sharing, not attempted on
    // this path either way), but it also does not reproduce
    // `NativeMeshBuilder::build()`'s own `E-MESH-DIM` check (a received
    // cell's topological dimension against the mesh's own) the way
    // `merge_halo_cells` used to get for free by routing through the
    // builder -- `all_cells_match_dimension` (mesh_merge_device.hpp) is
    // that one check, folded into this same post-exchange verdict alongside
    // `create<Space>`'s own.
    if (!device::all_cells_match_dimension<Space>(merged_arrays)) {
      post_check = Failure{"E-MESH-DIM",
                          fmt::format("rank {}: a halo cell's topological dimension disagrees "
                                     "with this rank's own mesh dimension {}",
                                     my_rank, dim)};
    } else {
      auto new_view = NativeMeshView::create<Space>(merged_arrays, my_rank);
      if (!new_view.ok()) {
        post_check = new_view.error();
      } else {
        new_source_view = std::move(new_view.value());
      }
    }
  } catch (const std::exception& e) {
    post_check = Failure{"E-MESH-HALO-MERGE",
                        fmt::format("rank {}: merging the received halo cells into this "
                                   "rank's own mesh threw ({}); refusing rather than "
                                   "propagating the exception past the post-exchange "
                                   "agreement every peer waits in",
                                   my_rank, e.what())};
  } catch (...) {
    post_check = Failure{"E-MESH-HALO-MERGE",
                        fmt::format("rank {}: merging the received halo cells into this "
                                   "rank's own mesh threw a non-std::exception value; "
                                   "refusing rather than propagating it past the "
                                   "post-exchange agreement every peer waits in",
                                   my_rank)};
  }
  {
    const int locally_ok = post_check.ok() ? 1 : 0;
    int globally_ok = locally_ok;
    MPI_Allreduce(&locally_ok, &globally_ok, 1, MPI_INT, MPI_MIN, resolved);
    if (globally_ok == 0) {
      if (!post_check.ok()) return post_check.error();
      return Failure{"E-MESH-HALO-PEER-REFUSED",
                   fmt::format("rank {} merged its received halo cells successfully, but at "
                              "least one peer failed merging or rebuilding its own mesh "
                              "view; every rank refuses together so none hangs on this "
                              "triple's next collective", my_rank)};
    }
  }

  const std::size_t adopted = owner_rank.size();

  if (remap_halo_trace_enabled()) {
    fmt::print(stderr,
               "FUSE_REMAP_HALO_TRACE rank={} source_mesh={} target_mesh={} dim={} "
               "adopted_source_cells={}\n",
               my_rank, source.value(), target.value(), dim, adopted);
  }

  fuse::linalg::GhostPattern pattern;
  pattern.owner_rank = std::move(owner_rank);
  pattern.remote_local_index = std::move(owner_owned_index);

  auto committed = backend.commit_remap_halo(source, placement, std::move(new_source_view),
                                            target, dim, std::move(pattern));
  if (!committed.ok()) return committed.error();
  return adopted;
#endif  // FUSE_HAVE_MPI
}

// ---------------------------------------------------------------------------
// KNOWN_ISSUES D27: the reverse exchange (native_halo.hpp's own header
// comment carries the full contract). Built from
// `fuse::linalg::PackedExchanger::exchange_packed_adjoint`, the same
// adjoint Alltoallv `src/backend/nodal_remap_distributed.cpp`'s own H^T
// scatter-add already uses -- `unpack_index()`/`pack_index()` are the two
// index maps that primitive exposes (comm.hpp's own header comment: "this
// participant's REQUEST side"/"ANSWER side"), and neither depends on any
// detail of the concrete MPI exchanger beyond the `PackedExchanger`
// interface itself.
// ---------------------------------------------------------------------------
Result<void> accumulate_remap_halo(const fuse::linalg::GhostExchanger& exchanger, double* owned,
                                   std::size_t owned_n, double* ghosts, std::size_t ghost_n) {
  if (ghost_n != exchanger.ghost_size()) {
    return Failure{"E-MESH-ARG",
                 fmt::format("accumulate_remap_halo: ghost_n={} does not match the exchanger's "
                            "own pattern ({} ghosts)",
                            ghost_n, exchanger.ghost_size())};
  }
  (void)owned_n;  // not indexed directly: every write goes through pack_index()'s own mapping.
  // Unconditional on FUSE_HAVE_MPI and on ghost_n == 0's own fix below
  // (next comment): a caller handing a nonzero ghost count a null buffer
  // is a caller bug this refuses regardless of what kind of exchanger
  // backs it -- unchanged from before this round.
  if (ghost_n > 0 && (owned == nullptr || ghosts == nullptr)) {
    return Failure{"E-MESH-ARG",
                 fmt::format("accumulate_remap_halo: null buffer with {} ghost(s) outstanding",
                            ghost_n)};
  }
#if FUSE_HAVE_MPI
  const auto* packed = dynamic_cast<const fuse::linalg::PackedExchanger*>(&exchanger);
  if (packed == nullptr) {
    if (ghost_n == 0) {
      // A true no-op: no outstanding ghosts AND no PackedExchanger facet
      // to exchange them through -- `make_serial_exchanger()`'s own
      // result (ghost_size() == 0, not a PackedExchanger at all).
      return Result<void>::success();
    }
    return Failure{"E-MESH-HALO",
                 fmt::format("accumulate_remap_halo: this exchanger carries {} ghost(s) but "
                            "exposes no PackedExchanger facet -- unreachable for a real "
                            "make_mpi_exchanger() result, which always implements both",
                            ghost_n)};
  }
  // `packed` is a REAL make_mpi_exchanger() result: `exchange_packed_
  // adjoint()` below is one `MPI_Alltoallv` over the exchanger's own
  // communicator, a COLLECTIVE every rank in that communicator must call
  // together. `ghost_n == 0` says only that THIS rank has no ghost
  // copies of its own to hand back -- it says nothing about whether a
  // PEER'S ghost copy of one of THIS rank's own owned entries is
  // waiting to arrive (`pack_index()`'s size, the answer-side count,
  // depends on what OTHER ranks requested from this rank, not on this
  // rank's own ghost_n). Short-circuiting here on ghost_n == 0 alone (as
  // this function did before) skips the call on exactly that rank,
  // losing every contribution it was due: measured 2026-10-19 by an
  // in-process reproduction of the 16x16/24x24 fixture at 3 ranks (round
  // 3) -- the rank with ghost_n == 0 there still owns 16 of a peer's
  // ghost cells, and omitting its own call to the collective lost
  // exactly those 16 (host-measured parallel.native_halo_mpi.3,
  // NativeHaloMpi.AccumulateRemapHaloRoundTripsExactly:
  // global_owned_sum=272 vs the expected 288). A real `MpiExchanger`
  // must therefore always proceed to the exchange below, exactly as
  // `MpiExchanger::update()` itself does -- that function has no such
  // short-circuit either. The ghost_n > 0 null check above already
  // covers `ghosts`; only `owned` needs a fresh check here, for the
  // ghost_n == 0 case the check above does not reach (`owned` is
  // dereferenced below through `pack_index()`, whose size depends on
  // peers' requests, not on ghost_n).
  if (owned == nullptr) {
    return Failure{"E-MESH-ARG", "accumulate_remap_halo: null owned buffer"};
  }
  // Pack this rank's own ghost values in REQUEST order (`unpack_index()`'s
  // own order: entry k is which GHOST SLOT this request-ordered position
  // answers -- the same map `update()` itself reads FROM when filling
  // ghosts; here the roles are reversed, so it is read, not written).
  // Zero-length when ghost_n == 0, which is a valid (possibly locally
  // empty) send side of the same collective Alltoallv every peer calls.
  const auto& unpack_idx = packed->unpack_index();
  std::vector<double> send_buf(ghost_n, 0.0);
  for (std::size_t k = 0; k < ghost_n; ++k) {
    send_buf[k] = ghosts[static_cast<std::size_t>(unpack_idx[k])];
  }
  // The adjoint Alltoallv: this rank's REQUEST-ordered ghost values travel
  // to their owners, landing in ANSWER order -- `pack_index()[k]` is which
  // of THIS rank's own owned local indices answer-position k's value
  // belongs to (the same map `update()` reads FROM to pack an owned value
  // for a REQUESTER; here it names the DESTINATION of an incoming one).
  const auto& pack_idx = packed->pack_index();
  std::vector<double> recv_buf(pack_idx.size(), 0.0);
  const auto ex = packed->exchange_packed_adjoint(send_buf.data(), recv_buf.data());
  if (!ex.ok()) return ex.error();
  // Accumulate: a caller receiving into the same destination index more
  // than once (several requesters holding a halo copy of the same owned
  // entity) must add, exactly as `comm.hpp`'s own `exchange_packed_adjoint`
  // header comment states -- the ordinary scatter-add a plain `+=` loop
  // already performs, with no further grouping needed.
  for (std::size_t k = 0; k < pack_idx.size(); ++k) {
    owned[static_cast<std::size_t>(pack_idx[k])] += recv_buf[k];
  }
  // The halo copies have been handed to their owners; leaving them in
  // place would let a second call add them again -- the identical
  // bookkeeping `MoabMeshBackend::Impl::run_halo_exchange`'s own reverse
  // branch performs (moab_backend.cpp).
  for (std::size_t g = 0; g < ghost_n; ++g) ghosts[g] = 0.0;
  return Result<void>::success();
#else
  if (ghost_n == 0) return Result<void>::success();
  return Failure{"E-MESH-HALO",
               fmt::format("accumulate_remap_halo: {} ghost(s) outstanding but this build has "
                          "no MPI, so no PackedExchanger facet can exist",
                          ghost_n)};
#endif  // FUSE_HAVE_MPI
}

}  // namespace fuse::native
