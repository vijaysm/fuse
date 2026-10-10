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

// ---- byte-buffer append/read helpers, matching native_migration.cpp's own
// convention exactly (duplicated rather than shared: the halo wire format
// carries two extra leading fields native_migration.cpp's own per-cell
// packer does not, and a "copy, never move" cell is a different enough
// operation from a migrated one that sharing a helper would mean threading
// an unused owner/index pair through every migration call site instead). --

template <class T>
void append_pod(std::vector<std::byte>& buf, const T& v) {
  static_assert(std::is_trivially_copyable_v<T>);
  const auto* p = reinterpret_cast<const std::byte*>(&v);
  buf.insert(buf.end(), p, p + sizeof(T));
}

template <class T>
T read_pod(const std::vector<std::byte>& buf, std::size_t& pos) {
  static_assert(std::is_trivially_copyable_v<T>);
  T v{};
  std::memcpy(&v, buf.data() + pos, sizeof(T));
  pos += sizeof(T);
  return v;
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

namespace {

void pack_one_halo_cell(const NativeMeshView& mesh, std::size_t c, int owner_rank,
                        std::int64_t owned_block_idx, std::vector<std::byte>& out) {
  const auto type = mesh.cell_type_of(c);
  const auto& verts = mesh.cell_vertices(c);
  append_pod<std::int32_t>(out, owner_rank);
  append_pod<std::int64_t>(out, owned_block_idx);
  append_pod<std::int64_t>(out, static_cast<std::int64_t>(mesh.global_id(mesh.dimension(), c)));
  append_pod<std::uint8_t>(out, static_cast<std::uint8_t>(type));
  append_pod<std::int32_t>(out, static_cast<std::int32_t>(verts.size()));
  std::unordered_map<int, int> local_to_closure;
  local_to_closure.reserve(verts.size());
  for (std::size_t k = 0; k < verts.size(); ++k) {
    const int lv = verts[k];
    local_to_closure[lv] = static_cast<int>(k);
    append_pod<std::int64_t>(out, static_cast<std::int64_t>(mesh.vertex_gid_of(
                                      static_cast<std::size_t>(lv))));
    append_pod<double>(out, mesh.vertex_x(static_cast<std::size_t>(lv), 0));
    append_pod<double>(out, mesh.vertex_x(static_cast<std::size_t>(lv), 1));
    append_pod<double>(out, mesh.vertex_x(static_cast<std::size_t>(lv), 2));
  }
  if (type == CellType::Polyhedron) {
    const auto& faces = mesh.cell_faces(c);
    append_pod<std::int32_t>(out, static_cast<std::int32_t>(faces.size()));
    for (const auto& face : faces) {
      append_pod<std::int32_t>(out, static_cast<std::int32_t>(face.size()));
      for (int lv : face) {
        append_pod<std::int32_t>(out, static_cast<std::int32_t>(local_to_closure.at(lv)));
      }
    }
  }
}

}  // namespace

std::vector<std::vector<std::byte>> pack_halo_cells(
    const NativeMeshView& mesh, const std::vector<std::vector<std::size_t>>& send_by_local_index,
    const std::vector<std::size_t>& owned_block_index, int my_rank, int num_ranks) {
  std::vector<std::vector<std::byte>> result(static_cast<std::size_t>(num_ranks));
  for (int r = 0; r < num_ranks; ++r) {
    for (const std::size_t c : send_by_local_index[static_cast<std::size_t>(r)]) {
      FUSE_ASSERT(mesh.is_owned(mesh.dimension(), c),
                 "pack_halo_cells: cell {} is being sent as a halo copy but this rank does not "
                 "own it",
                 c);
      pack_one_halo_cell(mesh, c, my_rank,
                        static_cast<std::int64_t>(owned_block_index[c]),
                        result[static_cast<std::size_t>(r)]);
    }
  }
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
               "the native backend reports a communicator of " + std::to_string(ranks) +
                   " ranks, but this build has no MPI"};
#else
  auto src_r = backend.native_view(source, placement);
  if (!src_r.ok()) return src_r.error();
  auto tgt_r = backend.native_view(target, placement);
  if (!tgt_r.ok()) return tgt_r.error();
  const NativeMeshView& src = *src_r.value();
  const NativeMeshView& tgt = *tgt_r.value();
  if (src.dimension() != dim || tgt.dimension() != dim) {
    return Failure{"E-REMAP-DIM",
                 "build_native_remap_halo: dimension " + std::to_string(dim) +
                     " was asked for, but the source view reports " +
                     std::to_string(src.dimension()) + " and the target view " +
                     std::to_string(tgt.dimension())};
  }

  const int my_rank = backend.rank();
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
  if (!merged.ok()) return merged.error();

  auto new_view = NativeMeshView::create_from_host(merged.value(), my_rank);
  if (!new_view.ok()) return new_view.error();

  const std::size_t adopted = owner_rank.size();
  fuse::linalg::GhostPattern pattern;
  pattern.owner_rank = std::move(owner_rank);
  pattern.remote_local_index = std::move(owner_owned_index);

  auto committed = backend.commit_remap_halo(source, placement, std::move(new_view.value()),
                                            target, dim, std::move(pattern));
  if (!committed.ok()) return committed.error();
  return adopted;
#endif  // FUSE_HAVE_MPI
}

}  // namespace fuse::native
