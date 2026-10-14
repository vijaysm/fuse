#include "fuse/backend/native_mesh_io.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <locale>
#include <sstream>

#include "fuse/base/error.hpp"

namespace fuse::native {

namespace {

/// The header's own contract: ids ride as `double`, exact for every
/// integer up to 2^53. A gid at or beyond that bound would be silently
/// rounded by the very representation this format relies on, so the
/// writer asserts the bound (an invariant on this backend's own id
/// assignment, never a data-dependent refusal a caller can trigger) and
/// never emits a value the reader could not reconstruct.
constexpr std::int64_t kMaxExactIdAsDouble = std::int64_t{1} << 53;

/// VTK legacy cell-type codes, per the standard (VTK_VERTEX=1, LINE=3,
/// TRIANGLE=5, POLYGON=7, QUAD=9, TETRA=10, HEXAHEDRON=12, WEDGE=13,
/// PYRAMID=14, POLYHEDRON=42).
int vtk_cell_type(CellType t) {
  switch (t) {
    case CellType::Point: return 1;
    case CellType::Segment: return 3;
    case CellType::Triangle: return 5;
    case CellType::Quad: return 9;
    case CellType::Polygon: return 7;
    case CellType::Tet: return 10;
    case CellType::Pyramid: return 14;
    case CellType::Prism: return 13;
    case CellType::Hex: return 12;
    case CellType::Polyhedron: return 42;
  }
  return -1;
}

Result<CellType> from_vtk_cell_type(int code) {
  switch (code) {
    case 1: return CellType::Point;
    case 3: return CellType::Segment;
    case 5: return CellType::Triangle;
    case 9: return CellType::Quad;
    case 7: return CellType::Polygon;
    case 10: return CellType::Tet;
    case 14: return CellType::Pyramid;
    case 13: return CellType::Prism;
    case 12: return CellType::Hex;
    case 42: return CellType::Polyhedron;
    default:
      return Failure{"E-MESH-IO", fmt::format("unrecognised VTK cell type code {}", code)};
  }
}

}  // namespace

Result<void> write_native_mesh_file(const std::string& path, const NativeMeshView& mesh,
                                    int rank) {
  std::ofstream f(path);
  if (!f) {
    return Failure{"E-MESH-IO", fmt::format("cannot open '{}' for writing", path)};
  }
  // Canonical-text rule (docs/dev/conventions.md): pin the classic locale so
  // the bytes this writes do not depend on the host's grouping/decimal
  // separator.
  f.imbue(std::locale::classic());
  // The default stream precision (6 significant digits) is for on-screen
  // display, not for bytes a reader must reconstruct exactly: a gid at or
  // past 1e6, or a coordinate with more than 6 significant digits, was
  // silently rounded on write (measured: 1234567 -> "1.23457e+06" -> read
  // back as 1234570, colliding with a different real vertex since gids
  // are dense in the generators). `max_digits10` for `double` is 17, the
  // number of decimal digits that round-trips every double exactly.
  f << std::setprecision(17);

  const std::size_t nv = mesh.vertex_count();
  const std::size_t nc = mesh.cell_count();

  f << "# vtk DataFile Version 3.0\n";
  f << "fuse native mesh, rank " << rank << "\n";
  f << "# FUSE_MESH_DIM " << mesh.dimension() << "\n";
  f << "ASCII\n";
  f << "DATASET UNSTRUCTURED_GRID\n";
  f << "POINTS " << nv << " double\n";
  for (std::size_t v = 0; v < nv; ++v) {
    f << mesh.vertex_x(v, 0) << " " << mesh.vertex_x(v, 1) << " " << mesh.vertex_x(v, 2) << "\n";
  }

  std::size_t total_ints = 0;
  for (std::size_t c = 0; c < nc; ++c) {
    const auto type = mesh.cell_type_of(c);
    if (type == CellType::Polyhedron) {
      std::size_t entry = 1;  // nfaces
      for (const auto& face : mesh.cell_faces(c)) entry += 1 + face.size();
      total_ints += 1 + entry;  // the leading "how many ints for this cell"
    } else {
      total_ints += 1 + mesh.cell_vertices(c).size();
    }
  }
  f << "CELLS " << nc << " " << total_ints << "\n";
  for (std::size_t c = 0; c < nc; ++c) {
    const auto type = mesh.cell_type_of(c);
    if (type == CellType::Polyhedron) {
      const auto& faces = mesh.cell_faces(c);
      std::size_t entry = 1;
      for (const auto& face : faces) entry += 1 + face.size();
      f << entry << " " << faces.size();
      for (const auto& face : faces) {
        f << " " << face.size();
        for (int v : face) f << " " << v;
      }
      f << "\n";
    } else {
      const auto& verts = mesh.cell_vertices(c);
      f << verts.size();
      for (int v : verts) f << " " << v;
      f << "\n";
    }
  }

  f << "CELL_TYPES " << nc << "\n";
  for (std::size_t c = 0; c < nc; ++c) f << vtk_cell_type(mesh.cell_type_of(c)) << "\n";

  f << "POINT_DATA " << nv << "\n";
  f << "SCALARS GlobalPointID double 1\nLOOKUP_TABLE default\n";
  for (std::size_t v = 0; v < nv; ++v) {
    const std::int64_t gid = mesh.vertex_gid_of(v);
    FUSE_ASSERT(gid < kMaxExactIdAsDouble,
               "vertex {} has gid {}, at or beyond 2^53: this format's double "
               "representation would not round-trip it exactly",
               v, gid);
    f << static_cast<double>(gid) << "\n";
  }
  f << "SCALARS PointOwner int 1\nLOOKUP_TABLE default\n";
  for (std::size_t v = 0; v < nv; ++v) f << mesh.vertex_owner_of(v) << "\n";
  f << "SCALARS PointIsOwned int 1\nLOOKUP_TABLE default\n";
  for (std::size_t v = 0; v < nv; ++v) f << (mesh.vertex_owned_of(v) ? 1 : 0) << "\n";

  f << "CELL_DATA " << nc << "\n";
  f << "SCALARS GlobalCellID double 1\nLOOKUP_TABLE default\n";
  for (std::size_t c = 0; c < nc; ++c) {
    const std::int64_t gid = mesh.global_id(mesh.dimension(), c);
    FUSE_ASSERT(gid < kMaxExactIdAsDouble,
               "cell {} has gid {}, at or beyond 2^53: this format's double "
               "representation would not round-trip it exactly",
               c, gid);
    f << static_cast<double>(gid) << "\n";
  }
  f << "SCALARS CellOwner int 1\nLOOKUP_TABLE default\n";
  for (std::size_t c = 0; c < nc; ++c) f << mesh.cell_owner_of(c) << "\n";
  f << "SCALARS CellIsOwned int 1\nLOOKUP_TABLE default\n";
  for (std::size_t c = 0; c < nc; ++c) f << (mesh.cell_owned_of(c) ? 1 : 0) << "\n";

  if (!f) {
    return Failure{"E-MESH-IO", fmt::format("write to '{}' failed partway through", path)};
  }
  return Result<void>::success();
}

namespace {

/// A cursor over the file's tokens, comments already stripped -- except the
/// one FUSE-specific comment (`# FUSE_MESH_DIM <n>`), captured into `dim`
/// as tokens are read. Every other line starting with `#` is dropped, same
/// as a real VTK reader does.
class TokenReader {
 public:
  explicit TokenReader(std::istream& in) {
    std::string line;
    while (std::getline(in, line)) {
      if (line.rfind("# FUSE_MESH_DIM", 0) == 0) {
        std::istringstream iss(line.substr(std::string("# FUSE_MESH_DIM").size()));
        // NM-13: conventions.md's canonical-text rule ("pin
        // std::locale::classic() on EVERY stream it creates") covers this
        // reader's own per-line streams too, not just the file stream the
        // writer pins -- only this one extracts more than a single digit
        // today (the tokenizing stream below reads whole words, never
        // numerically, so the locale was never load-bearing there either
        // until a future caller relies on it).
        iss.imbue(std::locale::classic());
        iss >> dim;
        continue;
      }
      if (!line.empty() && line[0] == '#') continue;
      std::istringstream iss(line);
      iss.imbue(std::locale::classic());
      std::string tok;
      while (iss >> tok) tokens_.push_back(tok);
    }
  }

  [[nodiscard]] bool done() const { return pos_ >= tokens_.size(); }
  [[nodiscard]] std::string word() {
    if (done()) return {};
    return tokens_[pos_++];
  }
  [[nodiscard]] Result<long long> integer() {
    if (done()) return Failure{"E-MESH-IO", "unexpected end of file (expected an integer)"};
    const std::string& t = tokens_[pos_++];
    try {
      return static_cast<long long>(std::stoll(t));
    } catch (...) {
      return Failure{"E-MESH-IO", fmt::format("expected an integer, got '{}'", t)};
    }
  }
  [[nodiscard]] Result<double> real() {
    if (done()) return Failure{"E-MESH-IO", "unexpected end of file (expected a number)"};
    const std::string& t = tokens_[pos_++];
    try {
      return std::stod(t);
    } catch (...) {
      return Failure{"E-MESH-IO", fmt::format("expected a number, got '{}'", t)};
    }
  }
  [[nodiscard]] bool peek_is(std::string_view w) const {
    return pos_ < tokens_.size() && tokens_[pos_] == w;
  }

  int dim = -1;

 private:
  std::vector<std::string> tokens_;
  std::size_t pos_ = 0;
};

}  // namespace

Result<MeshArrays<Kokkos::HostSpace>> read_native_mesh_file(const std::string& path) {
  std::ifstream f(path);
  if (!f) {
    return Failure{"E-MESH-IO", fmt::format("cannot open '{}' for reading", path)};
  }
  f.imbue(std::locale::classic());
  TokenReader tr(f);

  std::size_t nv = 0, nc = 0;
  std::vector<std::array<double, 3>> points;
  std::vector<CellType> cell_types;
  std::vector<std::vector<int>> cell_verts;               // final vertex indices per cell
  std::vector<std::vector<std::vector<int>>> cell_faces;  // only for Polyhedron

  std::vector<double> point_gid, cell_gid;
  std::vector<int> point_owner, point_owned, cell_owner, cell_owned;
  std::vector<std::vector<long long>> raw_cells_;

#define FUSE_VTK_INT(dst)                       \
  do {                                          \
    auto r_ = tr.integer();                     \
    if (!r_.ok()) return r_.error();            \
    (dst) = r_.value();                         \
  } while (false)
#define FUSE_VTK_DBL(dst)                       \
  do {                                          \
    auto r_ = tr.real();                        \
    if (!r_.ok()) return r_.error();            \
    (dst) = r_.value();                         \
  } while (false)

  while (!tr.done()) {
    const std::string tok = tr.word();
    if (tok == "ASCII" || tok == "BINARY") {
      continue;
    } else if (tok == "DATASET") {
      const std::string kind = tr.word();
      if (kind != "UNSTRUCTURED_GRID") {
        return Failure{"E-MESH-IO",
                       fmt::format("this reader understands only UNSTRUCTURED_GRID, got '{}'",
                                   kind)};
      }
    } else if (tok == "POINTS") {
      long long n = 0;
      FUSE_VTK_INT(n);
      (void)tr.word();  // the datatype name ("double"), unused: always parsed as double
      nv = static_cast<std::size_t>(n);
      points.resize(nv);
      for (std::size_t v = 0; v < nv; ++v) {
        FUSE_VTK_DBL(points[v][0]);
        FUSE_VTK_DBL(points[v][1]);
        FUSE_VTK_DBL(points[v][2]);
      }
    } else if (tok == "CELLS") {
      // Every cell's entry, plain or polyhedron, is `N followed by exactly
      // N more integers` (that is what the writer's own `entry` field
      // means for a polyhedron, and what a plain cell's leading vertex
      // count means): so the whole block can be read uniformly, one cell
      // at a time, without yet knowing any cell's type.
      long long n = 0, total = 0;
      FUSE_VTK_INT(n);
      FUSE_VTK_INT(total);
      (void)total;
      nc = static_cast<std::size_t>(n);
      raw_cells_.assign(nc, {});
      for (std::size_t c = 0; c < nc; ++c) {
        long long count = 0;
        FUSE_VTK_INT(count);
        raw_cells_[c].resize(static_cast<std::size_t>(count));
        for (std::size_t k = 0; k < raw_cells_[c].size(); ++k) {
          FUSE_VTK_INT(raw_cells_[c][k]);
        }
      }
    } else if (tok == "CELL_TYPES") {
      long long n = 0;
      FUSE_VTK_INT(n);
      if (static_cast<std::size_t>(n) != nc) {
        return Failure{"E-MESH-IO", fmt::format("CELL_TYPES declares {} but CELLS declared {}",
                                                n, nc)};
      }
      cell_types.resize(nc);
      for (std::size_t c = 0; c < nc; ++c) {
        long long code = 0;
        FUSE_VTK_INT(code);
        auto ct = from_vtk_cell_type(static_cast<int>(code));
        if (!ct.ok()) return ct.error();
        cell_types[c] = ct.value();
      }
    } else if (tok == "POINT_DATA" || tok == "CELL_DATA") {
      const bool is_point = (tok == "POINT_DATA");
      long long n = 0;
      FUSE_VTK_INT(n);
      const std::size_t count = static_cast<std::size_t>(n);
      auto& gid = is_point ? point_gid : cell_gid;
      auto& owner = is_point ? point_owner : cell_owner;
      auto& owned = is_point ? point_owned : cell_owned;
      gid.assign(count, 0.0);
      owner.assign(count, 0);
      owned.assign(count, 1);
      while (tr.peek_is("SCALARS")) {
        (void)tr.word();  // "SCALARS"
        const std::string field_name = tr.word();
        (void)tr.word();  // datatype, unused
        (void)tr.word();  // "1" (num components), unused
        (void)tr.word();  // "LOOKUP_TABLE"
        (void)tr.word();  // "default"
        for (std::size_t i = 0; i < count; ++i) {
          if (field_name == "GlobalPointID" || field_name == "GlobalCellID") {
            FUSE_VTK_DBL(gid[i]);
          } else if (field_name == "PointOwner" || field_name == "CellOwner") {
            long long v = 0; FUSE_VTK_INT(v); owner[i] = static_cast<int>(v);
          } else if (field_name == "PointIsOwned" || field_name == "CellIsOwned") {
            long long v = 0; FUSE_VTK_INT(v); owned[i] = static_cast<int>(v);
          } else {
            long long v = 0; FUSE_VTK_INT(v);  // unknown field: skip its value
          }
        }
      }
    }
  }

#undef FUSE_VTK_INT
#undef FUSE_VTK_DBL

  if (cell_types.size() != nc) {
    return Failure{"E-MESH-IO", "CELL_TYPES block missing or shorter than CELLS"};
  }
  if (point_gid.size() != nv || cell_gid.size() != nc) {
    return Failure{"E-MESH-IO",
                   "GlobalPointID/GlobalCellID scalars missing; this reader needs them to "
                   "reconstruct global ids"};
  }

  if (raw_cells_.size() != nc) {
    return Failure{"E-MESH-IO", "CELLS block missing or shorter than declared"};
  }

  NativeMeshBuilder builder(tr.dim >= 0 ? tr.dim : cell_dimension(cell_types.empty()
                                                                      ? CellType::Point
                                                                      : cell_types[0]));
  std::vector<int> vidx(nv);
  for (std::size_t v = 0; v < nv; ++v) {
    vidx[v] = builder.add_vertex(static_cast<std::int64_t>(point_gid[v]), points[v][0],
                                 points[v][1], points[v][2], point_owner[v],
                                 point_owned[v] != 0);
  }

  cell_verts.assign(nc, {});
  cell_faces.assign(nc, {});
  for (std::size_t c = 0; c < nc; ++c) {
    const auto& raw = raw_cells_[c];
    if (cell_types[c] == CellType::Polyhedron) {
      std::size_t pos = 0;
      const std::size_t nfaces = static_cast<std::size_t>(raw[pos++]);
      std::vector<std::vector<int>> faces(nfaces);
      std::vector<int> all_verts;
      for (std::size_t fi = 0; fi < nfaces; ++fi) {
        const std::size_t fn = static_cast<std::size_t>(raw[pos++]);
        faces[fi].resize(fn);
        for (std::size_t k = 0; k < fn; ++k) {
          const int local = vidx[static_cast<std::size_t>(raw[pos++])];
          faces[fi][k] = local;
          if (std::find(all_verts.begin(), all_verts.end(), local) == all_verts.end()) {
            all_verts.push_back(local);
          }
        }
      }
      cell_faces[c] = std::move(faces);
      cell_verts[c] = std::move(all_verts);
    } else {
      std::vector<int> verts(raw.size());
      for (std::size_t k = 0; k < raw.size(); ++k) {
        verts[k] = vidx[static_cast<std::size_t>(raw[k])];
      }
      cell_verts[c] = std::move(verts);
    }
  }

  for (std::size_t c = 0; c < nc; ++c) {
    if (cell_types[c] == CellType::Polyhedron) {
      builder.add_polyhedron_cell(cell_faces[c], static_cast<std::int64_t>(cell_gid[c]),
                                  cell_owner[c], cell_owned[c] != 0);
    } else {
      builder.add_cell(cell_types[c], cell_verts[c], static_cast<std::int64_t>(cell_gid[c]),
                       cell_owner[c], cell_owned[c] != 0);
    }
  }

  return builder.build();
}

}  // namespace fuse::native
