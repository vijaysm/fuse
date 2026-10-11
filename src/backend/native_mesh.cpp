#include "fuse/backend/native_mesh.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "fuse/base/error.hpp"

namespace fuse::native {

// ------------------------------------------------------------- the builder

int NativeMeshBuilder::add_vertex(std::int64_t gid, double x, double y, double z, int owner,
                                  bool is_owned) {
  vx_.push_back(x);
  vy_.push_back(y);
  vz_.push_back(z);
  vgid_.push_back(gid);
  vowner_.push_back(owner);
  vowned_.push_back(is_owned);
  return static_cast<int>(vgid_.size()) - 1;
}

int NativeMeshBuilder::add_cell(CellType type, const std::vector<int>& local_vertices,
                                std::int64_t gid, int owner, bool is_owned) {
  ctype_.push_back(type);
  cverts_.push_back(local_vertices);
  cgid_.push_back(gid);
  cowner_.push_back(owner);
  cowned_.push_back(is_owned);
  cfaces_.emplace_back();
  cface_flip_.emplace_back();
  return static_cast<int>(ctype_.size()) - 1;
}

int NativeMeshBuilder::find_or_add_face(const std::vector<int>& local_vertices_outward,
                                        bool* flip) {
  std::vector<std::int64_t> key;
  key.reserve(local_vertices_outward.size());
  for (int lv : local_vertices_outward) key.push_back(vgid_[static_cast<std::size_t>(lv)]);
  std::vector<std::int64_t> sorted_key = key;
  std::sort(sorted_key.begin(), sorted_key.end());

  for (auto& [existing_key, idx] : face_index_) {
    if (existing_key == sorted_key) {
      // Found the same vertex SET. Determine whether this cell's cyclic
      // order matches the stored one or is its reverse: the two records
      // may start at a different corner, so compare rotation-invariantly
      // by global id (matching or reversed cyclic sequences of the same
      // multiset are the only two ways two faces of a manifold mesh can
      // agree on their corners).
      const auto& stored = faces_[static_cast<std::size_t>(idx)];
      std::vector<std::int64_t> stored_gids;
      stored_gids.reserve(stored.size());
      for (int lv : stored) stored_gids.push_back(vgid_[static_cast<std::size_t>(lv)]);

      const std::size_t n = key.size();
      bool forward_match = false, reverse_match = false;
      if (stored_gids.size() == n) {
        for (std::size_t shift = 0; shift < n && !forward_match; ++shift) {
          bool ok = true;
          for (std::size_t k = 0; k < n; ++k) {
            if (key[k] != stored_gids[(shift + k) % n]) { ok = false; break; }
          }
          if (ok) forward_match = true;
        }
        for (std::size_t shift = 0; shift < n && !reverse_match; ++shift) {
          bool ok = true;
          for (std::size_t k = 0; k < n; ++k) {
            if (key[k] != stored_gids[(shift + n - k) % n]) { ok = false; break; }
          }
          if (ok) reverse_match = true;
        }
      }
      *flip = !forward_match && reverse_match;
      return idx;
    }
  }
  faces_.push_back(local_vertices_outward);
  const int idx = static_cast<int>(faces_.size()) - 1;
  face_index_.emplace_back(sorted_key, idx);
  *flip = false;
  return idx;
}

int NativeMeshBuilder::add_polyhedron_cell(const std::vector<std::vector<int>>& faces,
                                           std::int64_t gid, int owner, bool is_owned) {
  std::vector<int> all_verts;
  for (const auto& f : faces) {
    for (int v : f) {
      if (std::find(all_verts.begin(), all_verts.end(), v) == all_verts.end()) {
        all_verts.push_back(v);
      }
    }
  }
  ctype_.push_back(CellType::Polyhedron);
  cverts_.push_back(all_verts);
  cgid_.push_back(gid);
  cowner_.push_back(owner);
  cowned_.push_back(is_owned);

  std::vector<int> face_idxs;
  std::vector<bool> flips;
  face_idxs.reserve(faces.size());
  flips.reserve(faces.size());
  for (const auto& f : faces) {
    bool flip = false;
    const int idx = find_or_add_face(f, &flip);
    face_idxs.push_back(idx);
    flips.push_back(flip);
  }
  cfaces_.push_back(face_idxs);
  cface_flip_.push_back(flips);
  return static_cast<int>(ctype_.size()) - 1;
}

Result<MeshArrays<Kokkos::HostSpace>> NativeMeshBuilder::build() const {
  MeshArrays<Kokkos::HostSpace> out;
  out.dim = dim_;

  const std::size_t nv = vgid_.size();
  out.vertex_coords = decltype(out.vertex_coords)("native_mesh.vertex_coords", nv);
  out.vertex_gid = decltype(out.vertex_gid)("native_mesh.vertex_gid", nv);
  out.vertex_owner = decltype(out.vertex_owner)("native_mesh.vertex_owner", nv);
  out.vertex_is_owned = decltype(out.vertex_is_owned)("native_mesh.vertex_is_owned", nv);
  for (std::size_t i = 0; i < nv; ++i) {
    out.vertex_coords(i, 0) = vx_[i];
    out.vertex_coords(i, 1) = vy_[i];
    out.vertex_coords(i, 2) = vz_[i];
    out.vertex_gid(i) = vgid_[i];
    out.vertex_owner(i) = vowner_[i];
    out.vertex_is_owned(i) = vowned_[i];
  }

  const std::size_t nc = ctype_.size();
  std::vector<int> cv_offsets(nc + 1, 0);
  for (std::size_t c = 0; c < nc; ++c) {
    const int expect = fixed_corner_count(ctype_[c]);
    if (expect != 0 && static_cast<int>(cverts_[c].size()) != expect) {
      return Failure{"E-MESH-CONNECTIVITY",
                     fmt::format("cell {} is a shape needing {} corners but was given {}",
                                 c, expect, cverts_[c].size())};
    }
    for (int lv : cverts_[c]) {
      if (lv < 0 || static_cast<std::size_t>(lv) >= nv) {
        return Failure{"E-MESH-CONNECTIVITY",
                       fmt::format("cell {} names vertex {}, but the mesh has {} vertices",
                                   c, lv, nv)};
      }
    }
    if (cell_dimension(ctype_[c]) != dim_) {
      return Failure{"E-MESH-DIM",
                     fmt::format("cell {} has topological dimension {} but the mesh's is {}",
                                 c, cell_dimension(ctype_[c]), dim_)};
    }
    cv_offsets[c + 1] = cv_offsets[c] + static_cast<int>(cverts_[c].size());
  }
  out.cell_vertex_offsets = decltype(out.cell_vertex_offsets)("native_mesh.cell_vertex_offsets",
                                                               nc + 1);
  out.cell_vertex_indices = decltype(out.cell_vertex_indices)("native_mesh.cell_vertex_indices",
                                                               static_cast<std::size_t>(
                                                                   cv_offsets[nc]));
  out.cell_type = decltype(out.cell_type)("native_mesh.cell_type", nc);
  out.cell_gid = decltype(out.cell_gid)("native_mesh.cell_gid", nc);
  out.cell_owner = decltype(out.cell_owner)("native_mesh.cell_owner", nc);
  out.cell_is_owned = decltype(out.cell_is_owned)("native_mesh.cell_is_owned", nc);
  for (std::size_t c = 0; c < nc; ++c) {
    out.cell_vertex_offsets(c) = cv_offsets[c];
    for (std::size_t k = 0; k < cverts_[c].size(); ++k) {
      out.cell_vertex_indices(static_cast<std::size_t>(cv_offsets[c]) + k) = cverts_[c][k];
    }
    out.cell_type(c) = static_cast<std::uint8_t>(ctype_[c]);
    out.cell_gid(c) = cgid_[c];
    out.cell_owner(c) = cowner_[c];
    out.cell_is_owned(c) = cowned_[c];
  }
  out.cell_vertex_offsets(nc) = cv_offsets[nc];

  std::vector<int> cf_offsets(nc + 1, 0);
  for (std::size_t c = 0; c < nc; ++c) {
    cf_offsets[c + 1] = cf_offsets[c] + static_cast<int>(cfaces_[c].size());
  }
  out.cell_face_offsets = decltype(out.cell_face_offsets)("native_mesh.cell_face_offsets",
                                                           nc + 1);
  out.cell_face_indices = decltype(out.cell_face_indices)("native_mesh.cell_face_indices",
                                                           static_cast<std::size_t>(
                                                               cf_offsets[nc]));
  out.cell_face_flip = decltype(out.cell_face_flip)("native_mesh.cell_face_flip",
                                                     static_cast<std::size_t>(cf_offsets[nc]));
  for (std::size_t c = 0; c < nc; ++c) {
    out.cell_face_offsets(c) = cf_offsets[c];
    for (std::size_t k = 0; k < cfaces_[c].size(); ++k) {
      out.cell_face_indices(static_cast<std::size_t>(cf_offsets[c]) + k) = cfaces_[c][k];
      out.cell_face_flip(static_cast<std::size_t>(cf_offsets[c]) + k) = cface_flip_[c][k];
    }
  }
  out.cell_face_offsets(nc) = cf_offsets[nc];

  const std::size_t nf = faces_.size();
  std::vector<int> fv_offsets(nf + 1, 0);
  for (std::size_t f = 0; f < nf; ++f) fv_offsets[f + 1] = fv_offsets[f] + static_cast<int>(
                                                                faces_[f].size());
  out.face_vertex_offsets = decltype(out.face_vertex_offsets)("native_mesh.face_vertex_offsets",
                                                               nf + 1);
  out.face_vertex_indices = decltype(out.face_vertex_indices)("native_mesh.face_vertex_indices",
                                                               static_cast<std::size_t>(
                                                                   fv_offsets[nf]));
  for (std::size_t f = 0; f < nf; ++f) {
    out.face_vertex_offsets(f) = fv_offsets[f];
    for (std::size_t k = 0; k < faces_[f].size(); ++k) {
      out.face_vertex_indices(static_cast<std::size_t>(fv_offsets[f]) + k) = faces_[f][k];
    }
  }
  out.face_vertex_offsets(nf) = fv_offsets[nf];

  return out;
}

// -------------------------------------------------------- oriented faces

std::vector<OrientedFace> oriented_faces_of_cell(const MeshArrays<Kokkos::HostSpace>& arrays,
                                                 std::size_t cell) {
  const auto type = static_cast<CellType>(arrays.cell_type(cell));
  const int voff = arrays.cell_vertex_offsets(cell);
  const int vcount = arrays.cell_vertex_offsets(cell + 1) - voff;

  auto point_of = [&](int local_vertex) -> Vec3 {
    return {arrays.vertex_coords(local_vertex, 0), arrays.vertex_coords(local_vertex, 1),
            arrays.vertex_coords(local_vertex, 2)};
  };
  auto gid_of = [&](int local_vertex) -> std::int64_t {
    return arrays.vertex_gid(local_vertex);
  };

  std::vector<OrientedFace> out;
  if (type == CellType::Polyhedron) {
    const int foff = arrays.cell_face_offsets(cell);
    const int fcount = arrays.cell_face_offsets(cell + 1) - foff;
    out.reserve(static_cast<std::size_t>(fcount));
    for (int f = 0; f < fcount; ++f) {
      const int face_idx = arrays.cell_face_indices(foff + f);
      const bool flip = arrays.cell_face_flip(foff + f);
      const int fvoff = arrays.face_vertex_offsets(face_idx);
      const int fvcount = arrays.face_vertex_offsets(face_idx + 1) - fvoff;
      OrientedFace of;
      of.points.resize(static_cast<std::size_t>(fvcount));
      of.global_ids.resize(static_cast<std::size_t>(fvcount));
      for (int k = 0; k < fvcount; ++k) {
        const int kk = flip ? (fvcount - 1 - k) : k;
        const int lv = arrays.face_vertex_indices(fvoff + kk);
        of.points[static_cast<std::size_t>(k)] = point_of(lv);
        of.global_ids[static_cast<std::size_t>(k)] = gid_of(lv);
      }
      out.push_back(std::move(of));
    }
    return out;
  }

  const int nfaces = fixed_face_count(type);
  out.reserve(static_cast<std::size_t>(nfaces));
  for (int f = 0; f < nfaces; ++f) {
    const int fc = fixed_face_corner_count(type, f);
    int local[4] = {0, 0, 0, 0};
    (void)fixed_face_corners(type, f, local);
    OrientedFace of;
    of.points.resize(static_cast<std::size_t>(fc));
    of.global_ids.resize(static_cast<std::size_t>(fc));
    for (int k = 0; k < fc; ++k) {
      FUSE_ASSERT(local[k] < vcount, "face corner {} out of range for cell {} ({} corners)",
                 local[k], cell, vcount);
      const int lv = arrays.cell_vertex_indices(voff + local[k]);
      of.points[static_cast<std::size_t>(k)] = point_of(lv);
      of.global_ids[static_cast<std::size_t>(k)] = gid_of(lv);
    }
    out.push_back(std::move(of));
  }
  return out;
}

// ------------------------------------------------------------ NativeMeshView

namespace {
Point to_point(const Vec3& v) { return Point{v.x, v.y, v.z}; }
}  // namespace

Result<std::unique_ptr<NativeMeshView>> NativeMeshView::create_from_host(
    const MeshArrays<Kokkos::HostSpace>& arrays, int my_rank) {
  auto view = std::make_unique<NativeMeshView>();
  view->dim_ = arrays.dim;
  view->my_rank_ = my_rank;

  const std::size_t nv = arrays.vertex_count();
  view->vertex_xyz_.resize(nv);
  view->vertex_gid_.resize(nv);
  view->vertex_owner_.resize(nv);
  view->vertex_owned_.resize(nv);
  for (std::size_t v = 0; v < nv; ++v) {
    view->vertex_xyz_[v] = {arrays.vertex_coords(v, 0), arrays.vertex_coords(v, 1),
                            arrays.vertex_coords(v, 2)};
    view->vertex_gid_[v] = arrays.vertex_gid(v);
    view->vertex_owner_[v] = arrays.vertex_owner(v);
    view->vertex_owned_[v] = arrays.vertex_is_owned(v);
  }

  const std::size_t nc = arrays.cell_count();
  view->cell_type_.resize(nc);
  view->cell_vertices_.resize(nc);
  view->cell_faces_.resize(nc);
  view->cell_gid_.resize(nc);
  view->cell_owner_.resize(nc);
  view->cell_owned_.resize(nc);
  view->cell_simplices_.resize(nc);
  view->centroid_.resize(nc);
  view->measure_.resize(nc);

  for (std::size_t c = 0; c < nc; ++c) {
    const auto type = static_cast<CellType>(arrays.cell_type(c));
    view->cell_type_[c] = type;
    view->cell_gid_[c] = arrays.cell_gid(c);
    view->cell_owner_[c] = arrays.cell_owner(c);
    view->cell_owned_[c] = arrays.cell_is_owned(c);

    const int voff = arrays.cell_vertex_offsets(c);
    const int vcount = arrays.cell_vertex_offsets(c + 1) - voff;
    for (int k = 0; k < vcount; ++k) {
      const int lv = arrays.cell_vertex_indices(voff + k);
      if (lv < 0 || static_cast<std::size_t>(lv) >= nv) {
        return Failure{"E-MESH-CONNECTIVITY",
                       fmt::format("cell {} corner {} refers to vertex {}, but the mesh has "
                                  "{} vertices", c, k, lv, nv)};
      }
      view->cell_vertices_[c].push_back(lv);
    }

    // Centroid: the mean of the cell's own corner vertices (not of its
    // simplex decomposition -- for a non-convex cell those can differ, and
    // "centroid" here means the vertex average, matching the rest of the
    // codebase's convention, e.g. native/mesh.hpp).
    double cx = 0.0, cy = 0.0, cz = 0.0;
    for (int lv : view->cell_vertices_[c]) {
      cx += view->vertex_xyz_[static_cast<std::size_t>(lv)][0];
      cy += view->vertex_xyz_[static_cast<std::size_t>(lv)][1];
      cz += view->vertex_xyz_[static_cast<std::size_t>(lv)][2];
    }
    const double inv_n = 1.0 / static_cast<double>(vcount);
    view->centroid_[c] = Point{cx * inv_n, cy * inv_n, cz * inv_n};

    if (view->dim_ == 0) {
      view->measure_[c] = 0.0;  // a point has no extent (not a degeneracy)
      continue;
    }
    if (view->dim_ == 1) {
      // A segment IS its own simplex.
      const Vec3 a = {view->vertex_xyz_[static_cast<std::size_t>(view->cell_vertices_[c][0])]
                          [0],
                      view->vertex_xyz_[static_cast<std::size_t>(view->cell_vertices_[c][0])]
                          [1],
                      view->vertex_xyz_[static_cast<std::size_t>(view->cell_vertices_[c][0])]
                          [2]};
      const Vec3 b = {view->vertex_xyz_[static_cast<std::size_t>(view->cell_vertices_[c][1])]
                          [0],
                      view->vertex_xyz_[static_cast<std::size_t>(view->cell_vertices_[c][1])]
                          [1],
                      view->vertex_xyz_[static_cast<std::size_t>(view->cell_vertices_[c][1])]
                          [2]};
      view->measure_[c] = norm(b - a);
      view->cell_simplices_[c].push_back({to_point(a), to_point(b), Point{}, Point{}});
      continue;
    }
    if (view->dim_ == 2 && type == CellType::Triangle) {
      const auto& lv = view->cell_vertices_[c];
      const Vec3 a{view->vertex_xyz_[static_cast<std::size_t>(lv[0])][0],
                  view->vertex_xyz_[static_cast<std::size_t>(lv[0])][1],
                  view->vertex_xyz_[static_cast<std::size_t>(lv[0])][2]};
      const Vec3 b{view->vertex_xyz_[static_cast<std::size_t>(lv[1])][0],
                  view->vertex_xyz_[static_cast<std::size_t>(lv[1])][1],
                  view->vertex_xyz_[static_cast<std::size_t>(lv[1])][2]};
      const Vec3 cc{view->vertex_xyz_[static_cast<std::size_t>(lv[2])][0],
                   view->vertex_xyz_[static_cast<std::size_t>(lv[2])][1],
                   view->vertex_xyz_[static_cast<std::size_t>(lv[2])][2]};
      view->measure_[c] = triangle_area(a, b, cc);
      view->cell_simplices_[c].push_back({to_point(a), to_point(b), to_point(cc), Point{}});
      continue;
    }
    if (view->dim_ == 2) {
      // Quad or Polygon: the canonical global-id fan (geometry.hpp).
      std::vector<std::int64_t> gids;
      std::vector<Vec3> pts;
      for (int lv : view->cell_vertices_[c]) {
        gids.push_back(view->vertex_gid_[static_cast<std::size_t>(lv)]);
        pts.push_back({view->vertex_xyz_[static_cast<std::size_t>(lv)][0],
                      view->vertex_xyz_[static_cast<std::size_t>(lv)][1],
                      view->vertex_xyz_[static_cast<std::size_t>(lv)][2]});
      }
      auto tri_result = canonical_fan_triangulation(gids, pts);
      if (!tri_result.ok()) {
        return Failure{tri_result.error().code,
                       fmt::format("cell {}: {}", c, tri_result.error().message)};
      }
      double sum = 0.0;
      for (const auto& tri : tri_result.value()) {
        const Vec3& a = pts[static_cast<std::size_t>(tri[0])];
        const Vec3& b = pts[static_cast<std::size_t>(tri[1])];
        const Vec3& cc = pts[static_cast<std::size_t>(tri[2])];
        sum += triangle_area(a, b, cc);
        view->cell_simplices_[c].push_back({to_point(a), to_point(b), to_point(cc), Point{}});
      }
      view->measure_[c] = sum;
      continue;
    }

    if (view->dim_ == 3 && type == CellType::Tet) {
      const auto& lv = view->cell_vertices_[c];
      auto p = [&](int i) {
        return Vec3{view->vertex_xyz_[static_cast<std::size_t>(lv[i])][0],
                   view->vertex_xyz_[static_cast<std::size_t>(lv[i])][1],
                   view->vertex_xyz_[static_cast<std::size_t>(lv[i])][2]};
      };
      Tetrahedron t{p(0), p(1), p(2), p(3)};
      view->measure_[c] = tetrahedron_volume(t);
      view->cell_simplices_[c].push_back(
          {to_point(t.a), to_point(t.b), to_point(t.c), to_point(t.d)});
      continue;
    }

    // 3D, non-tet: Pyramid, Prism, Hex or Polyhedron. Fan every
    // outward-oriented face (canonically, by global id) to the cell
    // centroid, validated against the divergence-theorem volume.
    view->cell_faces_[c] = [&]() {
      std::vector<std::vector<int>> out;
      if (type == CellType::Polyhedron) {
        const int foff = arrays.cell_face_offsets(c);
        const int fcount = arrays.cell_face_offsets(c + 1) - foff;
        for (int f = 0; f < fcount; ++f) {
          const int face_idx = arrays.cell_face_indices(foff + f);
          const bool flip = arrays.cell_face_flip(foff + f);
          const int fvoff = arrays.face_vertex_offsets(face_idx);
          const int fvcount = arrays.face_vertex_offsets(face_idx + 1) - fvoff;
          std::vector<int> face(static_cast<std::size_t>(fvcount));
          for (int k = 0; k < fvcount; ++k) {
            const int kk = flip ? (fvcount - 1 - k) : k;
            face[static_cast<std::size_t>(k)] = arrays.face_vertex_indices(fvoff + kk);
          }
          out.push_back(std::move(face));
        }
      } else {
        const int nfaces = fixed_face_count(type);
        for (int f = 0; f < nfaces; ++f) {
          const int fc = fixed_face_corner_count(type, f);
          int local[4] = {0, 0, 0, 0};
          (void)fixed_face_corners(type, f, local);
          std::vector<int> face(static_cast<std::size_t>(fc));
          for (int k = 0; k < fc; ++k) {
            face[static_cast<std::size_t>(k)] = view->cell_vertices_[c][static_cast<std::size_t>(
                local[k])];
          }
          out.push_back(std::move(face));
        }
      }
      return out;
    }();

    std::vector<OrientedFace> oriented;
    oriented.reserve(view->cell_faces_[c].size());
    for (const auto& face : view->cell_faces_[c]) {
      OrientedFace of;
      of.points.reserve(face.size());
      of.global_ids.reserve(face.size());
      for (int lv : face) {
        of.points.push_back({view->vertex_xyz_[static_cast<std::size_t>(lv)][0],
                            view->vertex_xyz_[static_cast<std::size_t>(lv)][1],
                            view->vertex_xyz_[static_cast<std::size_t>(lv)][2]});
        of.global_ids.push_back(view->vertex_gid_[static_cast<std::size_t>(lv)]);
      }
      oriented.push_back(std::move(of));
    }

    const Vec3 centroid_v{view->centroid_[c].x, view->centroid_[c].y, view->centroid_[c].z};
    auto tet_result = canonical_cell_tetrahedra(oriented, centroid_v);
    if (!tet_result.ok()) {
      return Failure{tet_result.error().code,
                     fmt::format("cell {}: {}", c, tet_result.error().message)};
    }
    double sum = 0.0;
    for (const auto& t : tet_result.value()) {
      sum += tetrahedron_volume(t);
      view->cell_simplices_[c].push_back(
          {to_point(t.a), to_point(t.b), to_point(t.c), to_point(t.d)});
    }
    view->measure_[c] = sum;
  }

  return std::move(view);
}

void NativeMeshView::check(int dim, std::size_t i, const char* what) const {
  if (dim != 0 && dim != dim_) {
    FUSE_THROW(InvalidArgument, "NativeMeshView::{}: dimension {} is not represented "
              "(mesh dimension is {})", what, dim, dim_);
  }
  const std::size_t n = (dim == 0) ? vertex_count() : cell_count();
  if (i >= n) {
    FUSE_THROW(InvalidArgument,
              "NativeMeshView::{}: index {} out of range (dim {} has {} entities)", what, i,
              dim, n);
  }
}

std::size_t NativeMeshView::entity_count(int dim) const {
  if (dim == 0) return vertex_count();
  if (dim == dim_) return cell_count();
  return 0;
}

std::size_t NativeMeshView::owned_count(int dim) const {
  std::size_t n = 0;
  if (dim == 0) {
    for (bool o : vertex_owned_) if (o) ++n;
  } else if (dim == dim_) {
    for (bool o : cell_owned_) if (o) ++n;
  }
  return n;
}

bool NativeMeshView::is_owned(int dim, std::size_t i) const {
  check(dim, i, "is_owned");
  return (dim == 0) ? vertex_owned_[i] : cell_owned_[i];
}

Point NativeMeshView::centroid(int dim, std::size_t i) const {
  check(dim, i, "centroid");
  if (dim == 0) {
    return Point{vertex_xyz_[i][0], vertex_xyz_[i][1], vertex_xyz_[i][2]};
  }
  return centroid_[i];
}

double NativeMeshView::measure(int dim, std::size_t i) const {
  check(dim, i, "measure");
  if (dim == 0) {
    FUSE_THROW(InvalidArgument, "{}",
              "NativeMeshView::measure: this backend defines no measure for vertices "
              "(dim 0); a zero weight would silently drop them from a conservation sum");
  }
  return measure_[i];
}

std::size_t NativeMeshView::global_id(int dim, std::size_t i) const {
  check(dim, i, "global_id");
  return static_cast<std::size_t>((dim == 0) ? vertex_gid_[i] : cell_gid_[i]);
}

std::size_t NativeMeshView::simplex_count(int dim, std::size_t i) const {
  check(dim, i, "simplex_count");
  if (dim == 0) return 0;
  return cell_simplices_[i].size();
}

void NativeMeshView::simplex_points(int dim, std::size_t i, std::size_t s, Point* out) const {
  check(dim, i, "simplex_points");
  FUSE_ASSERT(out != nullptr, "simplex_points given a null output");
  FUSE_ASSERT(s < cell_simplices_[i].size(), "simplex {} of cell {} does not exist (it has {})",
             s, i, cell_simplices_[i].size());
  const int corners = dim_ + 1;
  for (int k = 0; k < corners; ++k) out[k] = cell_simplices_[i][s][static_cast<std::size_t>(k)];
}

// ---------------------------------------------------------------------------
// mesh_view.hpp's cell-to-vertex connectivity query (nodal-remap track). The
// native mesh already stores exactly this -- `cell_vertices_[c]`, corner
// indices in `native/cell_topology.hpp`'s convention, which is worded
// identically to `cell_simplices.hpp`'s `CellShape` tables ("quad base 0-3,
// apex 4"; "triangle 0-2, opposite triangle 3-5"; "face 0-3, opposite face
// 4-7") -- so no reordering is needed, only a type translation from this
// mesh's own `CellType` (which additionally has `Point` and `Polyhedron`,
// neither representable as a `CellShape`) to `CellShape`.
// ---------------------------------------------------------------------------
Result<MeshView::CellVertices> NativeMeshView::cell_vertices(int dim, std::size_t i) const {
  check(dim, i, "cell_vertices");
  if (dim != dim_) {
    return Failure{"E-MESH-NO-CELL-VERTICES",
                   "NativeMeshView::cell_vertices: connectivity is stored only for "
                   "this mesh's top-dimension cells (dimension " +
                       std::to_string(dim_) + "), not dimension " + std::to_string(dim)};
  }
  CellShape shape{};
  switch (cell_type_[i]) {
    case CellType::Segment: shape = CellShape::Edge; break;
    case CellType::Triangle: shape = CellShape::Triangle; break;
    case CellType::Quad: shape = CellShape::Quad; break;
    case CellType::Polygon: shape = CellShape::Polygon; break;
    case CellType::Tet: shape = CellShape::Tet; break;
    case CellType::Pyramid: shape = CellShape::Pyramid; break;
    case CellType::Prism: shape = CellShape::Prism; break;
    case CellType::Hex: shape = CellShape::Hex; break;
    case CellType::Point:
    case CellType::Polyhedron:
    default:
      return Failure{"E-MESH-NO-CELL-VERTICES",
                     "NativeMeshView::cell_vertices: cell " + std::to_string(i) +
                         " is a Point or a general Polyhedron, neither of which has "
                         "a fuse::CellShape equivalent (a Polyhedron's faces are an "
                         "explicit CSR, not a fixed corner list)"};
  }
  MeshView::CellVertices out;
  out.shape = shape;
  out.vertices.assign(cell_vertices_[i].begin(), cell_vertices_[i].end());
  return out;
}

}  // namespace fuse::native
