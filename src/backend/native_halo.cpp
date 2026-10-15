// The only translation unit in fuse::native's halo machinery that names an
// MPI type (the box allgather); everything declared in native_halo.hpp
// without one is a pure function, exercised at tier 0 by an in-process
// emulation of several ranks (test/backend/native_halo_test.cpp). Same
// isolation convention native_migration.cpp's own header comment states.

#include "fuse/backend/native_halo.hpp"

#include <algorithm>
#include <cstring>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

#include "fuse/backend/native/mesh_pack_device.hpp"
#include "fuse/base/config.hpp"
#include "fuse/base/error.hpp"

#if FUSE_HAVE_MPI
#include <mpi.h>
#endif

namespace fuse::native {

namespace {

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

std::vector<std::vector<std::byte>> pack_halo_cells(
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

  auto packed = device::pack_fanout<true, Space>(mesh.device_arrays(), cell_d, dest_d, num_ranks,
                                                 owner_d, owned_idx_d);
  std::vector<std::vector<std::byte>> result(packed.size());
  for (std::size_t r = 0; r < packed.size(); ++r) result[r] = to_host_bytes(packed[r]);
  return result;
}

Result<MeshArrays<Kokkos::HostSpace>> merge_halo_cells(
    const NativeMeshView& source, const std::vector<std::vector<std::byte>>& received,
    int my_rank, std::vector<int>* owner_rank, std::vector<std::int64_t>* owner_owned_index) {
  (void)my_rank;
  FUSE_ASSERT(owner_rank != nullptr && owner_owned_index != nullptr,
             "merge_halo_cells: both output vectors are required");
  owner_rank->clear();
  owner_owned_index->clear();

  NativeMeshBuilder builder(source.dimension());
  std::unordered_map<std::int64_t, int> vertex_by_gid;
  std::unordered_set<std::int64_t> held_gids;

  // 1. Every existing vertex, verbatim, in `source`'s own local-index order
  //    -- so `source`'s local vertex index v is ALSO builder's local index
  //    v for the first `vertex_count()` vertices added, and every existing
  //    cell's own `cell_vertices(c)` can be handed to the builder with no
  //    translation at all.
  for (std::size_t v = 0; v < source.vertex_count(); ++v) {
    const int lv = builder.add_vertex(source.vertex_gid_of(v), source.vertex_x(v, 0),
                                      source.vertex_x(v, 1), source.vertex_x(v, 2),
                                      source.vertex_owner_of(v), source.vertex_owned_of(v));
    FUSE_ASSERT(static_cast<std::size_t>(lv) == v,
               "merge_halo_cells: existing vertex {} landed at builder index {}, breaking the "
               "identity mapping every existing cell's connectivity relies on",
               v, lv);
    vertex_by_gid[source.vertex_gid_of(v)] = lv;
  }

  // 2. Every existing cell, verbatim -- ownership and owner rank preserved,
  //    including a cell adopted by an earlier call for a different target.
  for (std::size_t c = 0; c < source.cell_count(); ++c) {
    const auto type = source.cell_type_of(c);
    const auto gid = static_cast<std::int64_t>(source.global_id(source.dimension(), c));
    held_gids.insert(gid);
    if (type == CellType::Polyhedron) {
      builder.add_polyhedron_cell(source.cell_faces(c), gid, source.cell_owner_of(c),
                                  source.cell_owned_of(c));
    } else {
      builder.add_cell(type, source.cell_vertices(c), gid, source.cell_owner_of(c),
                       source.cell_owned_of(c));
    }
  }

  // 3. Every NEW cell from `received`, skipping any whose global id this
  //    rank already holds (idempotency, and the "copy, don't duplicate"
  //    rule `MoabMeshBackend::adopt_remap_halo` documents) -- appended
  //    strictly after every existing cell, so `*owner_rank`/
  //    `*owner_owned_index` are exactly the new slots' GhostPattern, in
  //    append order.
  for (const auto& buf : received) {
    std::size_t pos = 0;
    while (pos < buf.size()) {
      const auto sender_owner_rank = read_pod<std::int32_t>(buf, pos);
      const auto sender_owned_index = read_pod<std::int64_t>(buf, pos);
      const auto cell_gid = read_pod<std::int64_t>(buf, pos);
      const auto type = static_cast<CellType>(read_pod<std::uint8_t>(buf, pos));
      const auto n = static_cast<std::size_t>(read_pod<std::int32_t>(buf, pos));

      std::vector<int> closure(n);
      for (std::size_t k = 0; k < n; ++k) {
        const auto vgid = read_pod<std::int64_t>(buf, pos);
        const double x = read_pod<double>(buf, pos);
        const double y = read_pod<double>(buf, pos);
        const double z = read_pod<double>(buf, pos);
        if (held_gids.count(cell_gid) == 0) {
          // Only materialize vertices for a cell we are actually about to
          // keep -- skipped below, but the wire format is fixed-length per
          // record regardless, so the read above always has to happen to
          // stay positioned for the next record.
          auto it = vertex_by_gid.find(vgid);
          if (it == vertex_by_gid.end()) {
            const int lv = builder.add_vertex(vgid, x, y, z, sender_owner_rank, false);
            it = vertex_by_gid.emplace(vgid, lv).first;
          }
          closure[k] = it->second;
        }
      }

      std::vector<std::vector<int>> faces;
      std::size_t nfaces = 0;
      if (type == CellType::Polyhedron) {
        nfaces = static_cast<std::size_t>(read_pod<std::int32_t>(buf, pos));
        faces.resize(nfaces);
        for (std::size_t f = 0; f < nfaces; ++f) {
          const auto nfv = static_cast<std::size_t>(read_pod<std::int32_t>(buf, pos));
          faces[f].resize(nfv);
          for (std::size_t k = 0; k < nfv; ++k) {
            const auto ci = static_cast<std::size_t>(read_pod<std::int32_t>(buf, pos));
            faces[f][k] = closure[ci];
          }
        }
      }

      if (held_gids.count(cell_gid) != 0) continue;  // already local: skip, do not duplicate
      held_gids.insert(cell_gid);

      int new_local_cell = 0;
      if (type == CellType::Polyhedron) {
        new_local_cell = builder.add_polyhedron_cell(faces, cell_gid, sender_owner_rank, false);
      } else {
        new_local_cell = builder.add_cell(type, closure, cell_gid, sender_owner_rank, false);
      }
      (void)new_local_cell;
      owner_rank->push_back(sender_owner_rank);
      owner_owned_index->push_back(sender_owned_index);
    }
  }

  return builder.build();
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
  auto src_r = backend.native_view(source, placement);
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

  const std::vector<std::vector<int>> dest_for_owned =
      covering_set_destinations(my_owned_src_boxes, my_rank, all_target_boxes,
                               owner_of_target_box);

  std::vector<std::vector<std::size_t>> send_by_local_index(static_cast<std::size_t>(ranks));
  for (std::size_t i = 0; i < dest_for_owned.size(); ++i) {
    for (int r : dest_for_owned[i]) {
      send_by_local_index[static_cast<std::size_t>(r)].push_back(owned_local_index[i]);
    }
  }

  const std::vector<std::vector<std::byte>> send =
      pack_halo_cells(src, send_by_local_index, owned_idx, my_rank, ranks);
  const std::vector<std::vector<std::byte>> received = exchange_migration_buffers(send, comm);

  std::vector<int> owner_rank;
  std::vector<std::int64_t> owner_owned_index;
  auto merged = merge_halo_cells(src, received, my_rank, &owner_rank, &owner_owned_index);

  // POST-EXCHANGE verdict: `merge_halo_cells` and `create_from_host` can
  // each fail on data this rank alone received (a malformed incoming
  // record, a connectivity refusal from the rebuilt view) -- reconciled
  // here, BEFORE returning, the same way the pre-exchange verdict above
  // was: the caller of `build_native_remap_halo` (`build_weights`,
  // ultimately) performs its own collectives against the triple this
  // call just committed, and a rank that returned success here while a
  // peer returned failure would desync that caller's next collective.
  std::unique_ptr<NativeMeshView> new_source_view;
  Result<void> post_check = Result<void>::success();
  if (!merged.ok()) {
    post_check = merged.error();
  } else {
    auto new_view = NativeMeshView::create_from_host(merged.value(), my_rank);
    if (!new_view.ok()) {
      post_check = new_view.error();
    } else {
      new_source_view = std::move(new_view.value());
    }
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
  fuse::linalg::GhostPattern pattern;
  pattern.owner_rank = std::move(owner_rank);
  pattern.remote_local_index = std::move(owner_owned_index);

  auto committed = backend.commit_remap_halo(source, placement, std::move(new_source_view),
                                            target, dim, std::move(pattern));
  if (!committed.ok()) return committed.error();
  return adopted;
#endif  // FUSE_HAVE_MPI
}

}  // namespace fuse::native
