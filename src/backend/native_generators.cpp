#include "fuse/backend/native/cell_topology.hpp"
#include "fuse/backend/native_backend.hpp"

#include <array>
#include <charconv>
#include <cmath>
#include <limits>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>

#include "fuse/backend/cell_simplices.hpp"
#include "fuse/base/error.hpp"

namespace fuse::native {

std::pair<std::size_t, std::size_t> rank_slice(std::size_t n, int rank, int num_ranks) {
  if (num_ranks <= 1) return {0, n};
  const std::size_t base = n / static_cast<std::size_t>(num_ranks);
  const std::size_t extra = n % static_cast<std::size_t>(num_ranks);
  const std::size_t r = static_cast<std::size_t>(rank);
  const std::size_t lo = (r < extra) ? r * (base + 1) : extra * (base + 1) + (r - extra) * base;
  const std::size_t size = base + ((r < extra) ? 1 : 0);
  return {lo, lo + size};
}

namespace {

/// Which rank's slice (per `rank_slice`) contains global index `idx` of
/// `[0, n)`. Used only by generators, to decide vertex ownership at a
/// partition boundary: never called on more than a few thousand ranks, so
/// the linear scan costs nothing next to building the mesh itself.
int owner_of_index(std::size_t idx, std::size_t n, int num_ranks) {
  for (int r = 0; r < num_ranks; ++r) {
    const auto [lo, hi] = rank_slice(n, r, num_ranks);
    if (idx >= lo && idx < hi) return r;
  }
  return num_ranks > 0 ? num_ranks - 1 : 0;
}

[[nodiscard]] bool parse_size(std::string_view s, std::size_t& out) {
  if (s.empty()) return false;
  for (const char c : s) if (c < '0' || c > '9') return false;
  std::size_t v = 0;
  const auto res = std::from_chars(s.data(), s.data() + s.size(), v);
  if (res.ec != std::errc{} || res.ptr != s.data() + s.size()) return false;
  if (v == 0) return false;
  out = v;
  return true;
}

/// Splits "NxMxP" (or "NxM", or "N") into up to 3 positive decimal counts.
[[nodiscard]] bool parse_dims(std::string_view s, int want, std::array<std::size_t, 3>& n) {
  n = {1, 1, 1};
  int k = 0;
  std::size_t pos = 0;
  while (true) {
    const auto sep = s.find('x', pos);
    const std::string_view tok =
        s.substr(pos, sep == std::string_view::npos ? std::string_view::npos : sep - pos);
    if (k >= want || !parse_size(tok, n[static_cast<std::size_t>(k)])) return false;
    ++k;
    if (sep == std::string_view::npos) break;
    pos = sep + 1;
  }
  return k == want;
}

/// A structured vertex grid over `[lo_k, hi_k]` (inclusive) in the split
/// axis (the last one, index `dim - 1`) and `[0, n[k]]` in every other
/// axis, indexed x-fastest. Shared by every box-shaped generator so their
/// vertex numbering and ownership rule agree with each other -- which
/// matters when a test migrates one generator's mesh and expects it to
/// still tile the domain.
struct Grid {
  int dim;
  std::array<std::size_t, 3> n;  ///< cells per axis; n[2] == 1 in 2D
  std::size_t split_lo, split_hi;  ///< this rank's CELL range on the split axis
  int rank, num_ranks;

  [[nodiscard]] std::size_t split_axis() const { return dim == 3 ? 2 : 1; }
  [[nodiscard]] std::size_t split_extent() const { return n[split_axis()]; }

  [[nodiscard]] std::int64_t vertex_gid(std::size_t ix, std::size_t iy,
                                        std::size_t iz) const {
    const std::size_t sx = n[0] + 1, sy = n[1] + 1;
    return static_cast<std::int64_t>(ix + iy * sx + (dim == 3 ? iz * sx * sy : 0));
  }
  [[nodiscard]] std::int64_t cell_gid(std::size_t ix, std::size_t iy, std::size_t iz) const {
    return static_cast<std::int64_t>(ix + iy * n[0] + (dim == 3 ? iz * n[0] * n[1] : 0));
  }
  /// Owner of the vertex at split-axis grid index `s` (global, 0..extent):
  /// the cell immediately "below" it on the split axis, or cell 0 if `s`
  /// is the domain's own lower boundary -- deterministic and independent
  /// of the rank count, since `owner_of_index` derives it from `s` and
  /// `num_ranks` alone.
  [[nodiscard]] int vertex_owner_on_split_axis(std::size_t s) const {
    const std::size_t cell = (s == 0) ? 0 : s - 1;
    return owner_of_index(cell, split_extent(), num_ranks);
  }
};

/// Builds this rank's vertices for `g` and returns a lookup from grid index
/// to local vertex index, plus `g`'s own lo/hi so a caller can iterate
/// cells directly.
///
/// Indexing: `at(ix, iy, iz)`'s THIRD argument is always this rank's LOCAL
/// index along the split axis (`0` when the split axis is not `z`); its
/// SECOND argument is the local split-axis index too when `dim == 2` (the
/// split axis there IS `y`), or the FULL `y` range when `dim == 3` (there
/// the split axis is `z`, so `y` is never sliced). `sy`/`sz` below are
/// sized accordingly, which is what makes both callers' index arithmetic
/// address a correctly-sized `local` array.
struct VertexLookup {
  std::vector<int> local;
  std::size_t sx, sy, sz;

  [[nodiscard]] int at(std::size_t ix, std::size_t iy, std::size_t iz_local) const {
    return local[ix + iy * sx + iz_local * sx * sy];
  }
};

/// `native_backend.hpp` documents `E-MESH-SIZE` for a spec whose rank-local
/// vertex count would not fit this generator's own int-indexed builder
/// (the narrowing is recorded separately); without this check, nothing
/// would produce it -- a huge spec's `sx*sy*sz` product either overflowed
/// silently or stayed representable but vastly exceeded available memory,
/// so `VertexLookup::local.assign` failed as an uncaught
/// `std::bad_alloc`/`std::length_error` instead of this library's own
/// `Result` contract. `kMaxGeneratedEntitiesPerRank` is `INT32_MAX`: local
/// vertex and cell indices are `int` throughout the builder, so a rank's
/// own slice is refused once it would not fit one, well before the
/// allocation below is attempted.
constexpr std::size_t kMaxGeneratedEntitiesPerRank =
    static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max());

[[nodiscard]] Result<std::size_t> checked_entity_count(std::size_t a, std::size_t b,
                                                        std::size_t c, std::string_view what) {
  std::size_t product = a;
  for (std::size_t factor : {b, c}) {
    if (factor != 0 && product > kMaxGeneratedEntitiesPerRank / factor) {
      return Failure{"E-MESH-SIZE",
                     fmt::format("{} would need more than {} entities on one rank, which this "
                                "generator's int-indexed builder cannot hold",
                                what, kMaxGeneratedEntitiesPerRank)};
    }
    product *= factor;
  }
  if (product > kMaxGeneratedEntitiesPerRank) {
    return Failure{"E-MESH-SIZE",
                   fmt::format("{} requests {} entities on one rank, more than this "
                              "generator's {} limit",
                              what, product, kMaxGeneratedEntitiesPerRank)};
  }
  return product;
}

Result<VertexLookup> build_grid_vertices(NativeMeshBuilder& b, const Grid& g) {
  VertexLookup lut;
  const std::size_t split_count = g.split_hi - g.split_lo + 1;  // local vertex range, split axis
  lut.sx = g.n[0] + 1;

  auto place = [&](std::size_t ix, std::size_t iy, std::size_t iz, std::size_t lut_ix,
                   std::size_t lut_iy, std::size_t lut_iz, std::size_t s_global) {
    const std::int64_t gid = g.vertex_gid(ix, iy, iz);
    const int owner = g.vertex_owner_on_split_axis(s_global);
    const bool owned = (owner == g.rank);
    const double x = static_cast<double>(ix) / static_cast<double>(g.n[0]);
    const double y = static_cast<double>(iy) / static_cast<double>(g.n[1]);
    const double z = (g.dim == 3) ? static_cast<double>(iz) / static_cast<double>(g.n[2]) : 0.0;
    const int local_idx = b.add_vertex(gid, x, y, z, owner, owned);
    lut.local[lut_ix + lut_iy * lut.sx + lut_iz * lut.sx * lut.sy] = local_idx;
  };

  if (g.dim == 3) {
    lut.sy = g.n[1] + 1;  // full range: z is the split axis, not y
    lut.sz = split_count;
    const auto n = checked_entity_count(lut.sx, lut.sy, lut.sz, "this rank's vertex grid");
    if (!n.ok()) return n.error();
    lut.local.assign(n.value(), -1);
    for (std::size_t izl = 0; izl < split_count; ++izl) {
      const std::size_t iz = g.split_lo + izl;
      for (std::size_t iy = 0; iy < lut.sy; ++iy) {
        for (std::size_t ix = 0; ix < lut.sx; ++ix) place(ix, iy, iz, ix, iy, izl, iz);
      }
    }
  } else {
    lut.sy = split_count;  // local range: y IS the split axis in 2D
    lut.sz = 1;
    const auto n = checked_entity_count(lut.sx, lut.sy, lut.sz, "this rank's vertex grid");
    if (!n.ok()) return n.error();
    lut.local.assign(n.value(), -1);
    for (std::size_t iyl = 0; iyl < split_count; ++iyl) {
      const std::size_t iy = g.split_lo + iyl;
      for (std::size_t ix = 0; ix < lut.sx; ++ix) place(ix, iy, 0, ix, iyl, 0, iy);
    }
  }
  return lut;
}

}  // namespace

Result<MeshArrays<Kokkos::HostSpace>> generate_native_mesh(const std::string& spec, int rank,
                                                            int num_ranks) {
  const auto colon = spec.find(':');
  if (colon == std::string::npos) {
    return Failure{"E-MESH-SOURCE", fmt::format("malformed generated spec '{}'", spec)};
  }
  const std::string kind = spec.substr(0, colon);
  const std::string dims = spec.substr(colon + 1);

  if (kind == "line") {
    std::array<std::size_t, 3> n{};
    if (!parse_dims(dims, 1, n)) {
      return Failure{"E-MESH-SOURCE",
                     fmt::format("line needs one positive decimal cell count, got '{}'", dims)};
    }
    const std::size_t N = n[0];
    NativeMeshBuilder b(1);
    const auto [clo, chi] = rank_slice(N, rank, num_ranks);
    for (std::size_t v = clo; v <= chi; ++v) {
      const int owner = (v == 0) ? owner_of_index(0, N, num_ranks)
                                 : owner_of_index(v - 1, N, num_ranks);
      b.add_vertex(static_cast<std::int64_t>(v), static_cast<double>(v) / static_cast<double>(N),
                   0.0, 0.0, owner, owner == rank);
    }
    for (std::size_t c = clo; c < chi; ++c) {
      const int lv0 = static_cast<int>(c - clo);
      const int lv1 = lv0 + 1;
      b.add_cell(CellType::Segment, {lv0, lv1}, static_cast<std::int64_t>(c), rank, true);
    }
    return b.build();
  }

  if (kind == "quad" || kind == "tri") {
    std::array<std::size_t, 3> n{};
    if (!parse_dims(dims, 2, n)) {
      return Failure{"E-MESH-SOURCE",
                     fmt::format("{} needs two positive decimal counts 'NxM', got '{}'", kind,
                                 dims)};
    }
    Grid g{2, n, 0, 0, rank, num_ranks};
    std::tie(g.split_lo, g.split_hi) = rank_slice(n[1], rank, num_ranks);
    NativeMeshBuilder b(2);
    const auto lut_r = build_grid_vertices(b, g);
    if (!lut_r.ok()) return lut_r.error();
    const auto& lut = lut_r.value();
    for (std::size_t iy = g.split_lo; iy < g.split_hi; ++iy) {
      const std::size_t iyl = iy - g.split_lo;
      for (std::size_t ix = 0; ix < n[0]; ++ix) {
        const int v0 = lut.at(ix, iyl, 0), v1 = lut.at(ix + 1, iyl, 0);
        const int v2 = lut.at(ix + 1, iyl + 1, 0), v3 = lut.at(ix, iyl + 1, 0);
        const std::int64_t qgid = g.cell_gid(ix, iy, 0);
        if (kind == "quad") {
          b.add_cell(CellType::Quad, {v0, v1, v2, v3}, qgid, rank, true);
        } else {
          b.add_cell(CellType::Triangle, {v0, v1, v2}, qgid * 2, rank, true);
          b.add_cell(CellType::Triangle, {v0, v2, v3}, qgid * 2 + 1, rank, true);
        }
      }
    }
    return b.build();
  }

  if (kind == "hexbox" || kind == "tetbox" || kind == "prismpyramidmix" ||
      kind == "polyhedral") {
    std::array<std::size_t, 3> n{};
    if (!parse_dims(dims, 3, n)) {
      return Failure{"E-MESH-SOURCE",
                     fmt::format("{} needs three positive decimal counts 'NxMxP', got '{}'",
                                 kind, dims)};
    }
    Grid g{3, n, 0, 0, rank, num_ranks};
    std::tie(g.split_lo, g.split_hi) = rank_slice(n[2], rank, num_ranks);
    NativeMeshBuilder b(3);
    const auto lut_r = build_grid_vertices(b, g);
    if (!lut_r.ok()) return lut_r.error();
    const auto& lut = lut_r.value();
    const std::size_t total_verts = (n[0] + 1) * (n[1] + 1) * (n[2] + 1);

    for (std::size_t iz = g.split_lo; iz < g.split_hi; ++iz) {
      const std::size_t izl = iz - g.split_lo;
      for (std::size_t iy = 0; iy < n[1]; ++iy) {
        for (std::size_t ix = 0; ix < n[0]; ++ix) {
          int c[8] = {lut.at(ix, iy, izl),         lut.at(ix + 1, iy, izl),
                     lut.at(ix + 1, iy + 1, izl),  lut.at(ix, iy + 1, izl),
                     lut.at(ix, iy, izl + 1),      lut.at(ix + 1, iy, izl + 1),
                     lut.at(ix + 1, iy + 1, izl + 1), lut.at(ix, iy + 1, izl + 1)};
          const std::int64_t hgid = g.cell_gid(ix, iy, iz);

          if (kind == "hexbox") {
            b.add_cell(CellType::Hex, {c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]}, hgid,
                      rank, true);
          } else if (kind == "tetbox") {
            for (std::size_t s = 0; s < 6; ++s) {
              int corners[4];
              (void)simplex_corners(CellShape::Hex, 8, s, corners);
              b.add_cell(CellType::Tet,
                        {c[corners[0]], c[corners[1]], c[corners[2]], c[corners[3]]},
                        hgid * 6 + static_cast<std::int64_t>(s), rank, true);
            }
          } else if (kind == "polyhedral") {
            std::vector<std::vector<int>> faces;
            for (int f = 0; f < 6; ++f) {
              int local[4];
              (void)fixed_face_corners(CellType::Hex, f, local);
              faces.push_back({c[local[0]], c[local[1]], c[local[2]], c[local[3]]});
            }
            b.add_polyhedron_cell(faces, hgid, rank, true);
          } else {
            // prismpyramidmix: even (ix+iy+iz) -> 2 prisms along the 0-2/4-6
            // diagonal; odd -> 6 pyramids fanned to an added centroid vertex.
            if ((ix + iy + iz) % 2 == 0) {
              b.add_cell(CellType::Prism, {c[0], c[1], c[2], c[4], c[5], c[6]}, hgid * 2, rank,
                        true);
              b.add_cell(CellType::Prism, {c[0], c[2], c[3], c[4], c[6], c[7]}, hgid * 2 + 1,
                        rank, true);
            } else {
              double cx = 0.0, cy = 0.0, cz = 0.0;
              const double ux = static_cast<double>(ix) + 0.5, uy = static_cast<double>(iy) +
                                                                    0.5,
                          uz = static_cast<double>(iz) + 0.5;
              cx = ux / static_cast<double>(n[0]);
              cy = uy / static_cast<double>(n[1]);
              cz = uz / static_cast<double>(n[2]);
              const std::int64_t centroid_gid =
                  static_cast<std::int64_t>(total_verts) + hgid;
              const int apex =
                  b.add_vertex(centroid_gid, cx, cy, cz, rank, true);
              for (int f = 0; f < 6; ++f) {
                int local[4];
                (void)fixed_face_corners(CellType::Hex, f, local);
                // Reversed, so the pyramid's own outward direction at its
                // base agrees with the hex's outward direction there (the
                // hex face table lists corners outward-CCW from OUTSIDE
                // the hex; the Pyramid decomposition in cell_topology.hpp
                // wants them outward-CCW as seen from the APEX, i.e. the
                // reverse -- verified by
                // NativeGenerators.PrismPyramidMixCellsHaveThePyramidsVolume).
                b.add_cell(CellType::Pyramid,
                          {c[local[3]], c[local[2]], c[local[1]], c[local[0]], apex},
                          hgid * 6 + f, rank, true);
              }
            }
          }
        }
      }
    }
    return b.build();
  }

  if (kind == "cubedsphere") {
    std::array<std::size_t, 3> n{};
    if (!parse_dims(dims, 1, n)) {
      return Failure{"E-MESH-SOURCE",
                     fmt::format("cubedsphere needs one positive decimal edge count, got '{}'",
                                 dims)};
    }
    const std::size_t N = n[0];
    const auto [plo, phi] = rank_slice(6, rank, num_ranks);
    NativeMeshBuilder b(2);

    // Panel p's face normal axis and sign, and its two in-face axes u, v --
    // the standard six-panel cube layout.
    struct Panel { int axis; double sign; int u, v; };
    static constexpr Panel kPanels[6] = {
        {0, 1.0, 1, 2}, {0, -1.0, 1, 2}, {1, 1.0, 0, 2},
        {1, -1.0, 0, 2}, {2, 1.0, 0, 1}, {2, -1.0, 0, 1},
    };
    const std::size_t verts_per_panel = (N + 1) * (N + 1);
    const std::size_t cells_per_panel = N * N;

    for (std::size_t p = plo; p < phi; ++p) {
      const Panel& pan = kPanels[p];
      std::vector<int> vidx(verts_per_panel);
      for (std::size_t j = 0; j <= N; ++j) {
        for (std::size_t i = 0; i <= N; ++i) {
          double coord[3];
          coord[pan.axis] = pan.sign;
          coord[pan.u] = -1.0 + 2.0 * static_cast<double>(i) / static_cast<double>(N);
          coord[pan.v] = -1.0 + 2.0 * static_cast<double>(j) / static_cast<double>(N);
          const double len = std::sqrt(coord[0] * coord[0] + coord[1] * coord[1] +
                                       coord[2] * coord[2]);
          const std::int64_t gid = static_cast<std::int64_t>(p * verts_per_panel + j * (N + 1) +
                                                              i);
          vidx[j * (N + 1) + i] =
              b.add_vertex(gid, coord[0] / len, coord[1] / len, coord[2] / len, rank, true);
        }
      }
      for (std::size_t j = 0; j < N; ++j) {
        for (std::size_t i = 0; i < N; ++i) {
          const int v0 = vidx[j * (N + 1) + i], v1 = vidx[j * (N + 1) + i + 1];
          const int v2 = vidx[(j + 1) * (N + 1) + i + 1], v3 = vidx[(j + 1) * (N + 1) + i];
          const std::int64_t cgid = static_cast<std::int64_t>(p * cells_per_panel + j * N + i);
          b.add_cell(CellType::Quad, {v0, v1, v2, v3}, cgid, rank, true);
        }
      }
    }
    return b.build();
  }

  return Failure{"E-MESH-SOURCE", fmt::format("unknown generated kind '{}'", kind)};
}

}  // namespace fuse::native
