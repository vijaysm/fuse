// A test FIXTURE GENERATOR: writes an equiangular cubed sphere into an .h5m
// carrying PARALLEL_PARTITION sets, so `MoabMeshBackend` (whose read options
// are `PARALLEL=READ_PART;PARTITION=PARALLEL_PARTITION`) can read it
// partitioned.
//
// WHY IT EXISTS. Proving anything about a PARTITIONED remap needs two meshes
// that are partitioned DIFFERENTLY. The only partitioned mesh available is
// MOAB's own spherecube.h5m, and using it on both sides gives two identical
// distributions -- a pair where a partition-local intersection happens to be
// correct, so it cannot show what a halo is for. This produces the partner:
// different resolution, different partition rule, same sphere.
//
// EXACT PARAMETERS USED BY THE TEST SUITE, so the numbers in docs/HANDOFF.md
// are reproducible:
//
//     make_partitioned_sphere 16 7996.4553620 4 <out>.h5m block
//
// 16 -> 1536 quads against spherecube.h5m's 3456; the radius is spherecube's,
// asserted by remap_test.cpp to 1e-6; 4 partition sets against its 8; and
// `block` deals contiguous index ranges, which on this panel-major numbering
// lands differently from spherecube's own partition. `rr` (round robin) is
// also accepted and gives a still more adversarial distribution -- it measures
// a different loss figure (50.4%/75.2% rather than 66.7%/87.5%), which is why
// the mode is a parameter and the suite pins one of them rather than leaving
// it to chance.
//
// A THIRD TRANSLATION UNIT THAT SEES MOAB, and deliberately so. The dependency
// rule ("only src/mesh/moab_backend.cpp and src/remap/intx_weights.cpp may
// include moab/Core.hpp") is about the LIBRARY: nothing here is linked into
// libsoma_core, nothing here appears in a public header, and this file exists
// only to write a file before the tests run. Writing meshes is not something
// MeshBackend does or should do.

#include <moab/Core.hpp>
#include <moab/Range.hpp>
#include <MBTagConventions.hpp>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <tuple>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;

/// Panel `k` of the cube, mapped to a direction in R^3.
std::array<double, 3> panel_dir(int panel, double u, double v) {
  switch (panel) {
    case 0: return {1.0, u, v};
    case 1: return {-u, 1.0, v};
    case 2: return {-1.0, -u, v};
    case 3: return {u, -1.0, v};
    case 4: return {-v, u, 1.0};
    default: return {v, u, -1.0};
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr,
                 "usage: make_partitioned_sphere <n> <radius> <nparts> "
                 "<out.h5m> [block|rr]\n");
    return 2;
  }
  const int n = std::atoi(argv[1]);
  const double r = std::atof(argv[2]);
  const int nparts = std::atoi(argv[3]);
  const std::string out = argv[4];
  const std::string mode = argc > 5 ? argv[5] : "block";
  if (n < 1 || !(r > 0.0) || nparts < 1) {
    std::fprintf(stderr, "make_partitioned_sphere: n, radius and nparts must "
                         "all be positive\n");
    return 2;
  }
  if (mode != "block" && mode != "rr") {
    std::fprintf(stderr, "make_partitioned_sphere: mode must be 'block' or "
                         "'rr', not '%s'\n", mode.c_str());
    return 2;
  }

  moab::Core mb;

  // Vertices are welded by rounded coordinate, so the six panels share their
  // seams: 6n^2 + 2 vertices, not 6(n+1)^2. The count is printed and the test
  // asserts it -- an unwelded sphere is full of cracks and would make every
  // coverage figure below meaningless.
  std::map<std::tuple<long long, long long, long long>, moab::EntityHandle> weld;
  const double quantum = 1e-9 * r;
  auto add_vertex = [&](double x, double y, double z) {
    const double s = r / std::sqrt(x * x + y * y + z * z);
    x *= s;
    y *= s;
    z *= s;
    const auto key = std::make_tuple(std::llround(x / quantum),
                                     std::llround(y / quantum),
                                     std::llround(z / quantum));
    const auto it = weld.find(key);
    if (it != weld.end()) return it->second;
    const double c[3] = {x, y, z};
    moab::EntityHandle v = 0;
    if (mb.create_vertex(c, v) != moab::MB_SUCCESS) return moab::EntityHandle{0};
    weld.emplace(key, v);
    return v;
  };

  std::vector<moab::EntityHandle> quads;
  quads.reserve(static_cast<std::size_t>(6) * n * n);
  for (int panel = 0; panel < 6; ++panel) {
    std::vector<moab::EntityHandle> node(
        static_cast<std::size_t>(n + 1) * (n + 1), 0);
    for (int j = 0; j <= n; ++j) {
      for (int i = 0; i <= n; ++i) {
        const double a = -kPi / 4.0 + (kPi / 2.0) * i / n;
        const double b = -kPi / 4.0 + (kPi / 2.0) * j / n;
        const auto d = panel_dir(panel, std::tan(a), std::tan(b));
        const moab::EntityHandle v = add_vertex(d[0], d[1], d[2]);
        if (v == 0) {
          std::fprintf(stderr, "make_partitioned_sphere: create_vertex failed\n");
          return 1;
        }
        node[static_cast<std::size_t>(j) * (n + 1) + i] = v;
      }
    }
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        const auto at = [&](int p, int q) {
          return node[static_cast<std::size_t>(q) * (n + 1) + p];
        };
        const moab::EntityHandle conn[4] = {at(i, j), at(i + 1, j),
                                            at(i + 1, j + 1), at(i, j + 1)};
        moab::EntityHandle q = 0;
        if (mb.create_element(moab::MBQUAD, conn, 4, q) != moab::MB_SUCCESS) {
          std::fprintf(stderr, "make_partitioned_sphere: create_element failed\n");
          return 1;
        }
        quads.push_back(q);
      }
    }
  }

  // GLOBAL_ID on BOTH cells and vertices. A partitioned read needs it: the
  // MOAB backend refuses a mesh whose entities are unnumbered
  // (`E-MESH-GID`-class diagnostics), and the remap halo matches a halo cell
  // to its owner by global id and by nothing else.
  // Two steps, and NOT one call with MB_TAG_CREAT: moab::Core creates
  // GLOBAL_ID itself at construction with default -1, so asking for it with a
  // different default returns MB_ALREADY_ALLOCATED rather than the handle.
  // src/mesh/moab_backend.cpp's ensure_tag() documents the same trap; this
  // file hit it on the first run.
  moab::Tag gid = nullptr;
  moab::ErrorCode gid_rval =
      mb.tag_get_handle(GLOBAL_ID_TAG_NAME, 1, moab::MB_TYPE_INTEGER, gid);
  if (gid_rval == moab::MB_TAG_NOT_FOUND) {
    const int gid_default = 0;
    gid_rval = mb.tag_get_handle(GLOBAL_ID_TAG_NAME, 1, moab::MB_TYPE_INTEGER,
                                 gid, moab::MB_TAG_DENSE | moab::MB_TAG_CREAT,
                                 &gid_default);
  }
  if (gid_rval != moab::MB_SUCCESS) {
    std::fprintf(stderr, "make_partitioned_sphere: GLOBAL_ID tag failed\n");
    return 1;
  }
  {
    std::vector<int> ids(quads.size());
    for (std::size_t i = 0; i < quads.size(); ++i) ids[i] = static_cast<int>(i) + 1;
    if (mb.tag_set_data(gid, quads.data(), static_cast<int>(quads.size()),
                        ids.data()) != moab::MB_SUCCESS) {
      std::fprintf(stderr, "make_partitioned_sphere: cell ids failed\n");
      return 1;
    }
  }
  moab::Range verts;
  if (mb.get_entities_by_dimension(0, 0, verts) != moab::MB_SUCCESS) {
    std::fprintf(stderr, "make_partitioned_sphere: vertex query failed\n");
    return 1;
  }
  {
    std::vector<int> ids(verts.size());
    for (std::size_t i = 0; i < verts.size(); ++i) ids[i] = static_cast<int>(i) + 1;
    if (mb.tag_set_data(gid, verts, ids.data()) != moab::MB_SUCCESS) {
      std::fprintf(stderr, "make_partitioned_sphere: vertex ids failed\n");
      return 1;
    }
  }

  moab::Tag ptag = nullptr;
  int part_default = -1;
  if (mb.tag_get_handle(PARALLEL_PARTITION_TAG_NAME, 1, moab::MB_TYPE_INTEGER,
                        ptag, moab::MB_TAG_SPARSE | moab::MB_TAG_CREAT,
                        &part_default) != moab::MB_SUCCESS) {
    std::fprintf(stderr, "make_partitioned_sphere: partition tag failed\n");
    return 1;
  }
  std::vector<moab::EntityHandle> psets(static_cast<std::size_t>(nparts), 0);
  for (int p = 0; p < nparts; ++p) {
    if (mb.create_meshset(moab::MESHSET_SET,
                          psets[static_cast<std::size_t>(p)]) !=
            moab::MB_SUCCESS ||
        mb.tag_set_data(ptag, &psets[static_cast<std::size_t>(p)], 1, &p) !=
            moab::MB_SUCCESS) {
      std::fprintf(stderr, "make_partitioned_sphere: partition set failed\n");
      return 1;
    }
  }
  for (std::size_t i = 0; i < quads.size(); ++i) {
    const auto p = (mode == "block")
                       ? static_cast<std::size_t>(i * static_cast<std::size_t>(nparts) /
                                                  quads.size())
                       : (i % static_cast<std::size_t>(nparts));
    if (mb.add_entities(psets[p], &quads[i], 1) != moab::MB_SUCCESS) {
      std::fprintf(stderr, "make_partitioned_sphere: add_entities failed\n");
      return 1;
    }
  }

  if (mb.write_file(out.c_str()) != moab::MB_SUCCESS) {
    std::fprintf(stderr, "make_partitioned_sphere: could not write %s\n",
                 out.c_str());
    return 1;
  }
  std::printf("make_partitioned_sphere: %s  n=%d radius=%.7f quads=%zu "
              "vertices=%zu parts=%d mode=%s\n",
              out.c_str(), n, r, quads.size(),
              static_cast<std::size_t>(verts.size()), nparts, mode.c_str());
  return 0;
}
