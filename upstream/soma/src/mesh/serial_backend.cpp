#include "soma/mesh/serial_backend.hpp"

#include <array>
#include <charconv>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace soma {
namespace {

/// `a * b`, or false on overflow.
///
/// `std::size_t` is unsigned, so `a * b` WRAPS silently on overflow: it is
/// defined behaviour, which means neither UBSan nor ASan reports it, and the
/// wrapped value is an ordinary plausible-looking count. A 3e9^3 box would
/// otherwise report 1.5e19 cells and a cell volume of 6e-20, and every
/// downstream size computation would inherit the lie.
[[nodiscard]] bool checked_mul(std::size_t a, std::size_t b, std::size_t& out) {
  if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) return false;
  out = a * b;
  return true;
}

[[nodiscard]] bool checked_add(std::size_t a, std::size_t b, std::size_t& out) {
  if (a > std::numeric_limits<std::size_t>::max() - b) return false;
  out = a + b;
  return true;
}

/// The six axis-aligned boundary faces of a box, in the order local face
/// indices are blocked. Values are indices into that block layout.
enum class Side { XMin = 0, XMax = 1, YMin = 2, YMax = 3, ZMin = 4, ZMax = 5 };
constexpr int kSideCount = 6;

/// Structured grid on the unit line or the unit cube.
///
/// Cells are numbered x-fastest, then y, then z; vertices likewise. Boundary
/// faces are numbered in six contiguous blocks in `Side` order, each block
/// itself numbered with its lower-axis index fastest. The ordering is an
/// implementation detail of this backend -- `MeshView` promises none -- but it
/// has to be *defined* so that `centroid` and `measure` agree with each other
/// entity by entity.
class StructuredMesh final : public MeshView {
 public:
  StructuredMesh(int dim, std::array<std::size_t, 3> n) : dim_(dim), n_(n) {}

  [[nodiscard]] int dimension() const override { return dim_; }

  [[nodiscard]] bool represents(int dim) const {
    if (dim == 0) return true;
    if (dim == dim_) return true;
    return dim == 2 && dim_ == 3;  // the boundary skin of a box
  }

  [[nodiscard]] std::size_t entity_count(int dim) const override {
    if (dim == dim_) return cells();
    if (dim == 0) return vertices();
    if (dim == 2 && dim_ == 3) return boundary_faces();
    return 0;
  }

  [[nodiscard]] std::size_t owned_count(int dim) const override {
    return entity_count(dim);  // serial: no ghosts, everything is owned
  }

  [[nodiscard]] bool is_owned(int dim, std::size_t i) const override {
    check(dim, i, "is_owned");
    return true;
  }

  [[nodiscard]] Point centroid(int dim, std::size_t i) const override {
    check(dim, i, "centroid");
    Point p;
    if (dim == dim_ && dim_ == 1) {
      p.x = (static_cast<double>(i) + 0.5) * h(0);
      return p;
    }
    if (dim == 0 && dim_ == 1) {
      p.x = static_cast<double>(i) * h(0);
      return p;
    }
    if (dim == dim_ && dim_ == 3) {
      const std::size_t ix = i % n_[0];
      const std::size_t iy = (i / n_[0]) % n_[1];
      const std::size_t iz = i / (n_[0] * n_[1]);
      p.x = (static_cast<double>(ix) + 0.5) * h(0);
      p.y = (static_cast<double>(iy) + 0.5) * h(1);
      p.z = (static_cast<double>(iz) + 0.5) * h(2);
      return p;
    }
    if (dim == 0 && dim_ == 3) {
      const std::size_t sx = n_[0] + 1;
      const std::size_t sy = n_[1] + 1;
      p.x = static_cast<double>(i % sx) * h(0);
      p.y = static_cast<double>((i / sx) % sy) * h(1);
      p.z = static_cast<double>(i / (sx * sy)) * h(2);
      return p;
    }
    // dim == 2 && dim_ == 3: a boundary face.
    const auto [side, local] = locate_face(i);
    const int axis = static_cast<int>(side) / 2;      // 0 = x, 1 = y, 2 = z
    const bool at_max = (static_cast<int>(side) % 2) == 1;
    const int u = (axis + 1) % 3;                     // in-face axes, u fastest
    const int v = (axis + 2) % 3;
    double c[3] = {0.0, 0.0, 0.0};
    c[axis] = at_max ? 1.0 : 0.0;
    c[u] = (static_cast<double>(local % n_[static_cast<std::size_t>(u)]) + 0.5) *
           h(u);
    c[v] = (static_cast<double>(local / n_[static_cast<std::size_t>(u)]) + 0.5) *
           h(v);
    p.x = c[0];
    p.y = c[1];
    p.z = c[2];
    return p;
  }

  [[nodiscard]] double measure(int dim, std::size_t i) const override {
    check(dim, i, "measure");
    if (dim == 0) {
      throw std::out_of_range(
          "StructuredMesh::measure: this backend defines no measure for "
          "vertices (dim 0); a zero weight would silently drop them from a "
          "conservation sum");
    }
    if (dim == 1 && dim_ == 1) return h(0);
    if (dim == 3 && dim_ == 3) return h(0) * h(1) * h(2);
    // dim == 2 && dim_ == 3: the area of one boundary face, which depends on
    // which of the three orientations it has. Returning a single constant
    // here -- the mistake this replaces -- makes the boundary measures sum to
    // something other than the surface area on any non-cubic box.
    const auto [side, local] = locate_face(i);
    (void)local;
    const int axis = static_cast<int>(side) / 2;
    return h((axis + 1) % 3) * h((axis + 2) % 3);
  }

  [[nodiscard]] std::size_t global_id(int dim, std::size_t i) const override {
    check(dim, i, "global_id");
    return i;  // serial: one partition, so local index == global id
  }

  /// Entity count of one named boundary set. Precondition: `dim_ == 3`.
  [[nodiscard]] std::size_t side_count(Side s) const {
    const auto axis = static_cast<std::size_t>(static_cast<int>(s) / 2);
    return n_[(axis + 1) % 3] * n_[(axis + 2) % 3];
  }

  [[nodiscard]] std::size_t boundary_faces() const {
    // Every product and sum below was proved non-overflowing at load time.
    return 2 * (n_[0] * n_[1] + n_[1] * n_[2] + n_[0] * n_[2]);
  }

 private:
  /// Cell spacing along `axis`, i.e. the edge length of every cell. The unit
  /// domain divided into n_[axis] equal parts.
  [[nodiscard]] double h(int axis) const {
    return 1.0 / static_cast<double>(n_[static_cast<std::size_t>(axis)]);
  }

  [[nodiscard]] std::size_t cells() const {
    return dim_ == 1 ? n_[0] : n_[0] * n_[1] * n_[2];
  }
  [[nodiscard]] std::size_t vertices() const {
    return dim_ == 1 ? n_[0] + 1 : (n_[0] + 1) * (n_[1] + 1) * (n_[2] + 1);
  }

  struct FaceLoc {
    Side        side;
    std::size_t local;
  };

  /// Precondition: `i < boundary_faces()`, established by `check`.
  [[nodiscard]] FaceLoc locate_face(std::size_t i) const {
    std::size_t base = 0;
    for (int s = 0; s < kSideCount; ++s) {
      const auto side = static_cast<Side>(s);
      const std::size_t m = side_count(side);
      if (i < base + m) return FaceLoc{side, i - base};
      base += m;
    }
    // Unreachable given the precondition, but this is a Release build too:
    // falling off the end would return an uninitialised index.
    throw std::out_of_range("StructuredMesh: boundary face index " +
                            std::to_string(i) + " is out of range");
  }

  /// Checked unconditionally, not with assert(): `-O3 -DNDEBUG` compiles an
  /// assert out, and every accessor here would then compute a plausible
  /// position or weight for an entity that does not exist -- `centroid(3, i)`
  /// with `i` past the end returns a point far outside the domain, with no
  /// memory error for a sanitizer to catch.
  void check(int dim, std::size_t i, const char* what) const {
    if (!represents(dim)) {
      throw std::out_of_range(std::string("StructuredMesh::") + what +
                              ": this mesh does not represent dimension " +
                              std::to_string(dim));
    }
    const std::size_t n = entity_count(dim);
    if (i >= n) {
      throw std::out_of_range(std::string("StructuredMesh::") + what +
                              ": entity index " + std::to_string(i) +
                              " is outside [0, " + std::to_string(n) +
                              ") for dimension " + std::to_string(dim));
    }
  }

  int                        dim_;
  std::array<std::size_t, 3> n_;
};

/// A plain positive decimal integer, and nothing else.
///
/// `strtol` -- what this replaces -- silently accepts leading whitespace and a
/// leading '+', and saturates to LONG_MAX on a value too large for `long`,
/// turning an out-of-range size into a merely enormous one.
[[nodiscard]] bool parse_size(std::string_view s, std::size_t& out) {
  if (s.empty()) return false;
  for (const char c : s) {
    if (c < '0' || c > '9') return false;
  }
  std::size_t v = 0;
  const char* first = s.data();
  const char* last = first + s.size();
  const auto res = std::from_chars(first, last, v);
  if (res.ec != std::errc{} || res.ptr != last) return false;  // incl. ERANGE
  if (v == 0) return false;
  out = v;
  return true;
}

/// Reject a grid whose entity counts do not fit `std::size_t` before any of
/// them is ever computed for real.
[[nodiscard]] bool counts_fit(int dim, const std::array<std::size_t, 3>& n) {
  constexpr std::size_t kMax = std::numeric_limits<std::size_t>::max();
  std::size_t scratch = 0;
  if (dim == 1) {
    return n[0] < kMax;  // n[0] cells, n[0] + 1 vertices
  }
  // Cells.
  if (!checked_mul(n[0], n[1], scratch)) return false;
  const std::size_t nxy = scratch;
  if (!checked_mul(nxy, n[2], scratch)) return false;
  // Vertices.
  if (n[0] == kMax || n[1] == kMax || n[2] == kMax) return false;
  if (!checked_mul(n[0] + 1, n[1] + 1, scratch)) return false;
  if (!checked_mul(scratch, n[2] + 1, scratch)) return false;
  // Boundary faces: 2 * (n0*n1 + n1*n2 + n0*n2).
  std::size_t yz = 0;
  std::size_t xz = 0;
  if (!checked_mul(n[1], n[2], yz)) return false;
  if (!checked_mul(n[0], n[2], xz)) return false;
  std::size_t sum = 0;
  if (!checked_add(nxy, yz, sum)) return false;
  if (!checked_add(sum, xz, sum)) return false;
  return checked_mul(2, sum, scratch);
}

/// The named entity sets a generated box defines, beyond "" and "all".
struct NamedSide {
  std::string_view name;
  Side             side;
};
constexpr std::array<NamedSide, 6> kNamedSides{{
    {"left", Side::XMin},   {"right", Side::XMax},
    {"front", Side::YMin},  {"back", Side::YMax},
    {"bottom", Side::ZMin}, {"top", Side::ZMax},
}};

/// The topological dimension every named boundary set has.
constexpr int kBoundaryDim = 2;

}  // namespace

struct SerialMeshBackend::Impl {
  std::vector<std::unique_ptr<StructuredMesh>> meshes;
  /// Parallel to `meshes`: the `MeshDecl::name` each was loaded under, so
  /// `mesh_id` can resolve a name without the caller having to know the load
  /// order. Kept as a separate vector rather than inside StructuredMesh
  /// because a name is a property of the declaration, not of the topology.
  std::vector<std::string> names;
};

SerialMeshBackend::SerialMeshBackend() : impl_(std::make_unique<Impl>()) {}
SerialMeshBackend::~SerialMeshBackend() = default;
SerialMeshBackend::SerialMeshBackend(SerialMeshBackend&&) noexcept = default;
SerialMeshBackend& SerialMeshBackend::operator=(SerialMeshBackend&&) noexcept =
    default;

Result<void> SerialMeshBackend::load(const MeshDecl& decl) {
  if (impl_ == nullptr) {
    return Error{"E-MESH-MOVED",
                 "load() on a moved-from SerialMeshBackend"};
  }
  const std::string& src = decl.source;
  constexpr std::string_view kPrefix = "generated:";
  if (src.rfind(kPrefix, 0) != 0) {
    return Error{"E-MESH-SOURCE",
                 "serial backend understands only 'generated:...', got '" +
                     src + "'"};
  }
  const std::string rest = src.substr(kPrefix.size());
  const auto colon = rest.find(':');
  if (colon == std::string::npos) {
    return Error{"E-MESH-SOURCE", "malformed generated spec '" + src + "'"};
  }
  const std::string kind = rest.substr(0, colon);
  const std::string dims = rest.substr(colon + 1);

  int                        dim = 0;
  std::array<std::size_t, 3> n{1, 1, 1};

  if (kind == "line") {
    dim = 1;
    if (!parse_size(dims, n[0])) {
      return Error{"E-MESH-SOURCE",
                   "line needs one positive decimal cell count, got '" + dims +
                       "' in '" + src + "'"};
    }
  } else if (kind == "box") {
    dim = 3;
    // Split on 'x' by hand: an istringstream loop bounded by `k < 3` stops
    // reading after three tokens and silently accepts "2x2x2x2".
    std::size_t k = 0;
    std::size_t pos = 0;
    while (true) {
      const auto sep = dims.find('x', pos);
      const std::string_view tok =
          std::string_view{dims}.substr(pos, sep == std::string::npos
                                                 ? std::string::npos
                                                 : sep - pos);
      if (k >= 3 || !parse_size(tok, n[k])) {
        return Error{"E-MESH-SOURCE",
                     "box needs exactly three positive decimal counts "
                     "'NxMxP', got '" +
                         dims + "' in '" + src + "'"};
      }
      ++k;
      if (sep == std::string::npos) break;
      pos = sep + 1;
    }
    if (k != 3) {
      return Error{"E-MESH-SOURCE",
                   "box needs exactly three positive decimal counts 'NxMxP', "
                   "got '" +
                       dims + "' in '" + src + "'"};
    }
  } else {
    return Error{"E-MESH-SOURCE", "unknown generated kind '" + kind + "'"};
  }

  if (!counts_fit(dim, n)) {
    return Error{"E-MESH-SIZE",
                 "entity counts for '" + src +
                     "' do not fit std::size_t; they would wrap silently"};
  }

  impl_->meshes.push_back(std::make_unique<StructuredMesh>(dim, n));
  // Pushed only after the mesh is in, and only on the success path, so the
  // two vectors cannot drift: `meshes[i]` is always the mesh named
  // `names[i]`. A failed load above returned before either push.
  impl_->names.push_back(decl.name);
  return Result<void>::success();
}

Result<MeshId> SerialMeshBackend::mesh_id(std::string_view name) const {
  if (impl_ == nullptr) {
    return Error{"E-MESH-MOVED", "mesh_id() on a moved-from SerialMeshBackend"};
  }
  if (name.empty()) {
    return Error{"E-MESH-NAME",
                 "the empty mesh name matches nothing; name the mesh in its "
                 "MeshDecl"};
  }
  std::size_t found = 0;
  std::size_t first = 0;
  for (std::size_t i = 0; i < impl_->names.size(); ++i) {
    if (impl_->names[i] == name) {
      if (found == 0) first = i;
      ++found;
    }
  }
  if (found == 0) {
    return Error{"E-MESH-NAME", "no loaded mesh is named '" +
                                    std::string{name} + "' (" +
                                    std::to_string(impl_->names.size()) +
                                    " loaded)"};
  }
  if (found > 1) {
    return Error{"E-MESH-NAME-AMBIGUOUS",
                 std::to_string(found) + " loaded meshes are named '" +
                     std::string{name} +
                     "'; the name cannot identify one of them"};
  }
  return MeshId{static_cast<std::uint32_t>(first)};
}

Result<const MeshView*> SerialMeshBackend::view(MeshId id) const {
  if (impl_ == nullptr) {
    return Error{"E-MESH-MOVED", "view() on a moved-from SerialMeshBackend"};
  }
  if (!id.valid() || id.value() >= impl_->meshes.size()) {
    return Error{"E-MESH-ID", "no mesh with id " + std::to_string(id.value()) +
                                  " (" + std::to_string(impl_->meshes.size()) +
                                  " loaded)"};
  }
  return static_cast<const MeshView*>(impl_->meshes[id.value()].get());
}

Result<std::size_t> SerialMeshBackend::region_size(const Region& r) const {
  if (impl_ == nullptr) {
    return Error{"E-MESH-MOVED",
                 "region_size() on a moved-from SerialMeshBackend"};
  }
  if (!r.mesh.valid() || r.mesh.value() >= impl_->meshes.size()) {
    return Error{"E-MESH-ID",
                 "region '" + r.name + "' names mesh id " +
                     std::to_string(r.mesh.value()) + ", of which " +
                     std::to_string(impl_->meshes.size()) + " are loaded"};
  }
  // Resolved through the concrete vector rather than by downcasting the
  // MeshView* that view() returns: a static_cast down an abstract base is
  // undefined the moment another MeshView implementation reaches this code.
  const StructuredMesh& mesh = *impl_->meshes[r.mesh.value()];

  if (r.dimension < -1) {
    return Error{"E-MESH-DIM",
                 "region '" + r.name + "' has dimension " +
                     std::to_string(r.dimension) +
                     "; only -1 (meaning the mesh dimension) or a "
                     "non-negative dimension is meaningful"};
  }

  if (r.entity_set.empty() || r.entity_set == "all") {
    const int dim = r.dimension >= 0 ? r.dimension : mesh.dimension();
    if (!mesh.represents(dim)) {
      return Error{"E-MESH-DIM",
                   "region '" + r.name + "' asks for dimension " +
                       std::to_string(dim) +
                       ", which this mesh does not represent (mesh dimension " +
                       std::to_string(mesh.dimension()) + ")"};
    }
    return mesh.entity_count(dim);
  }

  const bool is_wall = r.entity_set == "wall";
  const NamedSide* named = nullptr;
  for (const auto& ns : kNamedSides) {
    if (r.entity_set == ns.name) {
      named = &ns;
      break;
    }
  }
  if (!is_wall && named == nullptr) {
    return Error{"E-MESH-SET", "unknown entity set '" + r.entity_set + "'"};
  }
  if (mesh.dimension() != 3) {
    return Error{"E-MESH-SET", "boundary set '" + r.entity_set +
                                   "' needs a 3D mesh; this one is " +
                                   std::to_string(mesh.dimension()) + "D"};
  }
  // A boundary set is a set of faces. Silently ignoring a stated dimension
  // that says otherwise would hand back a face count for a region the author
  // believes is volumetric.
  if (r.dimension >= 0 && r.dimension != kBoundaryDim) {
    return Error{"E-MESH-DIM",
                 "boundary set '" + r.entity_set + "' on region '" + r.name +
                     "' holds dimension-2 faces, but the region states "
                     "dimension " +
                     std::to_string(r.dimension)};
  }
  return is_wall ? mesh.boundary_faces() : mesh.side_count(named->side);
}

RegionSizes make_region_sizes(const CouplingGraph& g,
                              const MeshBackend& backend) {
  return [&g, &backend](RegionId id) -> std::size_t {
    if (!id.valid() || id.value() >= g.region_count()) {
      throw std::out_of_range(
          "make_region_sizes: region id " + std::to_string(id.value()) +
          " is invalid or outside the graph's " +
          std::to_string(g.region_count()) + " regions");
    }
    const Region& r = g.region(id);

    // The graph's MeshId and the backend's MeshId are two independent
    // positional numberings. Resolve through the mesh's NAME rather than
    // assuming they coincide: if they do not -- a backend that loaded the
    // meshes in a different order, or loaded extra ones -- passing r.mesh
    // straight through sizes a DIFFERENT mesh's region and returns a
    // wrong-but-plausible count with no diagnostic. See
    // MeshBackend::mesh_id.
    if (!r.mesh.valid() || r.mesh.value() >= g.mesh_count()) {
      throw std::out_of_range(
          "make_region_sizes: region '" + r.name + "' names mesh id " +
          std::to_string(r.mesh.value()) + ", which is invalid or outside " +
          "the graph's " + std::to_string(g.mesh_count()) + " meshes");
    }
    const std::string& mesh_name = g.mesh(r.mesh).name;
    const auto backend_mesh = backend.mesh_id(mesh_name);
    if (!backend_mesh.ok()) {
      throw std::runtime_error(
          "make_region_sizes: region '" + r.name + "' lives on mesh '" +
          mesh_name + "', which the " + backend.name() +
          " backend cannot identify: " + backend_mesh.error().code + ": " +
          backend_mesh.error().message);
    }

    Region resolved = r;
    resolved.mesh = backend_mesh.value();
    const auto n = backend.region_size(resolved);
    if (!n.ok()) {
      throw std::runtime_error("make_region_sizes: cannot size region '" +
                               r.name + "': " + n.error().code + ": " +
                               n.error().message);
    }
    return n.value();
  };
}

}  // namespace soma
