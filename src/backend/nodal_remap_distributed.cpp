#include "fuse/backend/nodal_remap_distributed.hpp"

#include "fuse/base/config.hpp"

#if FUSE_HAVE_MPI
#include <mpi.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <utility>

#include <fmt/format.h>

#include "fuse/search/types.hpp"

namespace fuse {
namespace {

using general_interp_detail::Cell;
using general_interp_detail::Vec3;
using general_interp_detail::basis_weights;
using general_interp_detail::cell_diameter;
using general_interp_detail::in_reference_domain;
using general_interp_detail::invert_reference_map;
using general_interp_detail::kRefEpsLoose;
using general_interp_detail::kRefEpsTight;
using general_interp_detail::kSurfaceTol;
using general_interp_detail::reference_violation;
using general_interp_detail::shape_corner_count;
using general_interp_detail::shape_supported;
using general_interp_detail::to_vec;
// kRefEpsTight/kRefEpsLoose/reference_violation are reused verbatim from
// general_interp.hpp's own namespace-scope constants and function (not
// re-derived here): the round-2 two-tier containment fix
// (build-evidence/mutations/nodal-remap.toml's "tier-a-skipped" row) applies
// exactly as much to a distributed row as to a serial one -- the failure
// mode (a lower-global-id neighbour's cell winning a shared-boundary point
// by EXTRAPOLATING past its own edge, under a single loose tolerance plus a
// tie-break that does not distinguish "genuinely contains" from "merely
// clears a curved-surface-faceting bound") does not depend on whether the
// candidate cells came from a local mesh or an Allgatherv'd replica.

/// One rank's `MPI_Allgatherv` of a homogeneous double array, `stride`
/// doubles per local "record" (a vertex, a cell, or a contribution). All
/// three assembly steps below (vertices, cells, field values) are this same
/// shape, so this is the one collective helper every one of them calls.
struct GatheredDoubles {
  std::vector<double> data;  // concatenated, `stride` doubles per record
  std::size_t records = 0;
};

// `comm_handle == fuse::search::kNullComm`: exactly one participant, so the
// caller's own local records already ARE the global list -- a plain copy,
// touching no MPI symbol at all (this is what lets every function below
// compile and run at Tier 0). Any other value is a real communicator
// (`MPI_Comm_c2f`'d by the caller): requires `FUSE_HAVE_MPI`, else refused
// with `E-RT-NODAL-DIST-NOMPI` rather than silently treated as local, which
// would give a build with MPI disabled a plausible-looking WRONG answer on
// a genuinely multi-rank communicator.
[[nodiscard]] Result<GatheredDoubles> allgatherv_records(search::CommHandle comm_handle,
                                                         const std::vector<double>& local,
                                                         int stride) {
  if (comm_handle == search::kNullComm) {
    GatheredDoubles out;
    out.records =
        stride > 0 ? local.size() / static_cast<std::size_t>(stride) : 0;
    out.data = local;
    return out;
  }
#if !FUSE_HAVE_MPI
  (void)stride;
  return Failure{"E-RT-NODAL-DIST-NOMPI",
                 "allgatherv_records: comm_handle names a real communicator, but this "
                 "build of FUSE has no MPI; configure with -DFUSE_ENABLE_MPI=ON, or pass "
                 "fuse::search::kNullComm for a single-rank caller"};
#else
  const MPI_Comm comm = MPI_Comm_f2c(static_cast<MPI_Fint>(comm_handle));
  int nranks = 0;
  if (MPI_Comm_size(comm, &nranks) != MPI_SUCCESS) {
    return Failure{"E-RT-NODAL-DIST-MPI", "allgatherv_records: MPI_Comm_size failed"};
  }
  const auto my_records =
      stride > 0 ? static_cast<int>(local.size() / static_cast<std::size_t>(stride)) : 0;
  std::vector<int> counts(static_cast<std::size_t>(nranks), 0);
  if (MPI_Allgather(&my_records, 1, MPI_INT, counts.data(), 1, MPI_INT, comm) != MPI_SUCCESS) {
    return Failure{"E-RT-NODAL-DIST-MPI", "allgatherv_records: MPI_Allgather (counts) failed"};
  }
  std::vector<int> byte_counts(static_cast<std::size_t>(nranks));
  std::vector<int> byte_displs(static_cast<std::size_t>(nranks));
  int total_records = 0;
  for (int r = 0; r < nranks; ++r) {
    byte_counts[static_cast<std::size_t>(r)] = counts[static_cast<std::size_t>(r)] * stride;
    byte_displs[static_cast<std::size_t>(r)] = total_records * stride;
    total_records += counts[static_cast<std::size_t>(r)];
  }
  GatheredDoubles out;
  out.records = static_cast<std::size_t>(total_records);
  out.data.assign(out.records * static_cast<std::size_t>(stride), 0.0);
  if (MPI_Allgatherv(local.data(), static_cast<int>(local.size()), MPI_DOUBLE, out.data.data(),
                     byte_counts.data(), byte_displs.data(), MPI_DOUBLE, comm) != MPI_SUCCESS) {
    return Failure{"E-RT-NODAL-DIST-MPI", "allgatherv_records: MPI_Allgatherv failed"};
  }
  return out;
#endif
}

/// The whole source nodal mesh, assembled identically on every rank --
/// see nodal_remap_distributed.hpp's file header for why this trades
/// scalability for an implementation whose correctness reduces entirely to
/// already-tested serial code.
struct AssembledMesh {
  std::vector<std::size_t> vertex_gid;
  std::vector<Vec3> vertex_point;
  std::unordered_map<std::size_t, std::size_t> gid_to_local;

  struct CellRec {
    std::size_t gid = 0;
    CellShape shape = CellShape::Edge;
    std::size_t n = 0;
    std::array<std::size_t, 8> corner_gid{};
  };
  std::vector<CellRec> cells;
};

[[nodiscard]] Result<AssembledMesh> gather_global_source(search::CommHandle comm_handle,
                                                          const MeshView& src) {
  const int dim = src.dimension();

  // ---- vertices: (gid, x, y, z), stride 4, OWNED only ----------------------
  std::vector<double> vbuf;
  const std::size_t nv = src.entity_count(0);
  for (std::size_t i = 0; i < nv; ++i) {
    if (!src.is_owned(0, i)) continue;
    const Point p = src.centroid(0, i);
    vbuf.push_back(static_cast<double>(src.global_id(0, i)));
    vbuf.push_back(p.x);
    vbuf.push_back(p.y);
    vbuf.push_back(p.z);
  }
  auto vg = allgatherv_records(comm_handle, vbuf, 4);
  if (!vg.ok()) return vg.error();

  // ---- cells: (gid, shape, n, corner_gid x 8), stride 11, OWNED only ------
  std::vector<double> cbuf;
  const std::size_t nc = src.entity_count(dim);
  for (std::size_t i = 0; i < nc; ++i) {
    if (!src.is_owned(dim, i)) continue;
    const auto cv = src.cell_vertices(dim, i);
    if (!cv.ok()) {
      return Failure{cv.error().code, "gather_global_source: cell " + std::to_string(i) +
                                          ": " + cv.error().message};
    }
    if (!shape_supported(cv.value().shape)) {
      return Failure{"E-INTERP-SHAPE",
                     fmt::format("gather_global_source: source cell {} has no standard "
                                 "nodal basis implemented (see general_interp.hpp)",
                                 i)};
    }
    if (cv.value().vertices.size() != shape_corner_count(cv.value().shape) ||
        cv.value().vertices.size() > 8) {
      return Failure{"E-INTERP-SHAPE",
                     "gather_global_source: cell " + std::to_string(i) +
                         " reports the wrong corner count for its shape"};
    }
    cbuf.push_back(static_cast<double>(src.global_id(dim, i)));
    cbuf.push_back(static_cast<double>(static_cast<int>(cv.value().shape)));
    cbuf.push_back(static_cast<double>(cv.value().vertices.size()));
    for (std::size_t k = 0; k < 8; ++k) {
      cbuf.push_back(k < cv.value().vertices.size()
                         ? static_cast<double>(src.global_id(0, cv.value().vertices[k]))
                         : -1.0);
    }
  }
  auto cg = allgatherv_records(comm_handle, cbuf, 11);
  if (!cg.ok()) return cg.error();

  AssembledMesh out;
  out.vertex_gid.reserve(vg.value().records);
  out.vertex_point.reserve(vg.value().records);
  for (std::size_t k = 0; k < vg.value().records; ++k) {
    const double* rec = &vg.value().data[k * 4];
    const auto gid = static_cast<std::size_t>(rec[0]);
    const Vec3 p{rec[1], rec[2], rec[3]};
    const auto it = out.gid_to_local.find(gid);
    if (it != out.gid_to_local.end()) {
      const Vec3& q = out.vertex_point[it->second];
      const double d = std::sqrt((p.x - q.x) * (p.x - q.x) + (p.y - q.y) * (p.y - q.y) +
                                 (p.z - q.z) * (p.z - q.z));
      const double scale = std::max(1.0, std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z));
      if (d > 1e-9 * scale) {
        return Failure{"E-RT-NODAL-DIST-OWNER",
                       fmt::format("gather_global_source: vertex global id {} was reported "
                                   "OWNED by two ranks at two different positions -- an "
                                   "ownership contract violation (MeshView::is_owned must "
                                   "agree with itself across ranks)",
                                   gid)};
      }
      continue;  // same vertex, reported by two ranks at the same position: harmless.
    }
    out.gid_to_local.emplace(gid, out.vertex_point.size());
    out.vertex_gid.push_back(gid);
    out.vertex_point.push_back(p);
  }

  out.cells.reserve(cg.value().records);
  std::unordered_map<std::size_t, std::size_t> seen_cell;  // gid -> position in out.cells
  for (std::size_t k = 0; k < cg.value().records; ++k) {
    const double* rec = &cg.value().data[k * 11];
    AssembledMesh::CellRec cr;
    cr.gid = static_cast<std::size_t>(rec[0]);
    cr.shape = static_cast<CellShape>(static_cast<int>(rec[1]));
    cr.n = static_cast<std::size_t>(rec[2]);
    for (std::size_t j = 0; j < 8; ++j) {
      cr.corner_gid[j] = rec[3 + j] < 0.0 ? std::numeric_limits<std::size_t>::max()
                                          : static_cast<std::size_t>(rec[3 + j]);
    }
    if (seen_cell.count(cr.gid) != 0) {
      return Failure{"E-RT-NODAL-DIST-OWNER",
                     fmt::format("gather_global_source: cell global id {} was reported "
                                 "OWNED by two ranks",
                                 cr.gid)};
    }
    seen_cell.emplace(cr.gid, out.cells.size());
    out.cells.push_back(cr);
  }
  return out;
}

[[nodiscard]] Result<std::vector<Cell>> build_cells(const AssembledMesh& mesh) {
  std::vector<Cell> cells(mesh.cells.size());
  for (std::size_t c = 0; c < mesh.cells.size(); ++c) {
    const auto& cr = mesh.cells[c];
    Cell cell;
    cell.shape = cr.shape;
    cell.n = cr.n;
    for (std::size_t k = 0; k < cr.n; ++k) {
      const auto it = mesh.gid_to_local.find(cr.corner_gid[k]);
      if (it == mesh.gid_to_local.end()) {
        return Failure{
            "E-RT-NODAL-DIST-OWNER",
            fmt::format("build_distributed_interp_rows: cell global id {} names corner "
                        "vertex global id {}, which no rank reported owning",
                        cr.gid, cr.corner_gid[k])};
      }
      cell.p[k] = mesh.vertex_point[it->second];
    }
    cells[c] = cell;
  }
  return cells;
}

}  // namespace

Result<DistributedInterpRows> build_distributed_interp_rows(const MeshView& src_nodal,
                                                             const MeshView& eval_points,
                                                             int eval_dim,
                                                             search::CommHandle comm_handle) {
  const auto assembled_r = gather_global_source(comm_handle, src_nodal);
  if (!assembled_r.ok()) return assembled_r.error();
  const AssembledMesh& mesh = assembled_r.value();
  if (mesh.cells.empty() || mesh.vertex_gid.empty()) {
    return Failure{"E-INTERP-NOGEOMETRY",
                   "build_distributed_interp_rows: the source mesh, assembled across every "
                   "rank of the communicator, has no cells or no vertices at all"};
  }
  const auto cells_r = build_cells(mesh);
  if (!cells_r.ok()) return cells_r.error();
  const std::vector<Cell>& cells = cells_r.value();

  DistributedInterpRows out;
  const std::size_t n_eval = eval_points.entity_count(eval_dim);
  for (std::size_t j = 0; j < n_eval; ++j) {
    if (!eval_points.is_owned(eval_dim, j)) continue;
    const Vec3 target = to_vec(eval_points.centroid(eval_dim, j));

    // TWO-TIER SELECTION -- identical rule to general_interp.hpp's serial
    // builder (see that file's kRefEpsTight/kRefEpsLoose comments for the
    // derivation); only the candidate list's SOURCE differs (every rank's
    // Allgatherv'd cell here vs. a broad-phase box query there). Tier (a): a
    // candidate that GENUINELY contains the point (kRefEpsTight) wins
    // outright, tie-broken by lowest source global id ONLY among tight
    // matches. Tier (b), reached only when no candidate passes tier (a): the
    // single least-violating candidate (raw, eps = 0, score), accepted if it
    // also clears kRefEpsLoose -- the genuine curved-surface faceting gap,
    // never a tie-break.
    bool have_tight = false;
    std::size_t tight_c = 0;
    std::size_t tight_gid = 0;
    std::array<double, 3> tight_u{};

    bool have_fallback = false;
    std::size_t fallback_c = 0;
    double fallback_score = std::numeric_limits<double>::max();
    std::array<double, 3> fallback_u{};
    bool fallback_passes_loose = false;

    for (std::size_t c = 0; c < cells.size(); ++c) {
      const auto inv = invert_reference_map(cells[c], target);
      if (!inv.ok()) return inv.error();
      const double diam = std::max(cell_diameter(cells[c]), 1.0);
      const bool surf_ok = inv.value().residual <= kSurfaceTol * diam;
      const std::size_t gid = mesh.cells[c].gid;

      if (surf_ok && in_reference_domain(cells[c].shape, inv.value().u, kRefEpsTight)) {
        if (!have_tight || gid < tight_gid) {
          have_tight = true;
          tight_c = c;
          tight_gid = gid;
          tight_u = inv.value().u;
        }
        continue;
      }

      const double lateral = reference_violation(cells[c].shape, inv.value().u, 0.0);
      const double normal_gap = surf_ok ? 0.0 : (inv.value().residual / diam);
      const double score = lateral + normal_gap * normal_gap;
      const bool passes_loose =
          surf_ok && in_reference_domain(cells[c].shape, inv.value().u, kRefEpsLoose);
      if (!have_fallback || score < fallback_score) {
        have_fallback = true;
        fallback_c = c;
        fallback_score = score;
        fallback_u = inv.value().u;
        fallback_passes_loose = passes_loose;
      }
    }

    std::size_t best_c;
    std::array<double, 3> best_u{};
    if (have_tight) {
      best_c = tight_c;
      best_u = tight_u;
    } else if (have_fallback && fallback_passes_loose) {
      best_c = fallback_c;
      best_u = fallback_u;
    } else {
      return Failure{"E-INTERP-DOMAIN",
                     fmt::format("build_distributed_interp_rows: evaluation entity {} (at "
                                 "({:.6g}, {:.6g}, {:.6g})) is not contained by any source "
                                 "cell assembled across the communicator; "
                                 "InterfaceDomainPolicy::ClampToBoundary is not implemented "
                                 "for the distributed builder (see the file header)",
                                 j, target.x, target.y, target.z)};
    }

    DistributedInterpRow row;
    row.local_index = j;
    const std::array<double, 8> bw = basis_weights(cells[best_c].shape, best_u);
    std::vector<std::pair<std::size_t, double>> tmp;
    tmp.reserve(cells[best_c].n);
    for (std::size_t k = 0; k < cells[best_c].n; ++k) {
      tmp.emplace_back(mesh.cells[best_c].corner_gid[k], bw[k]);
    }
    std::sort(tmp.begin(), tmp.end(),
             [](const auto& a, const auto& b) { return a.first < b.first; });
    double rowsum = 0.0;
    for (const auto& [gid, w] : tmp) {
      row.src_global_ids.push_back(gid);
      row.weights.push_back(w);
      rowsum += w;
    }
    out.max_row_defect = std::max(out.max_row_defect, std::abs(rowsum - 1.0));
    out.rows.push_back(std::move(row));
  }
  return out;
}

Result<void> apply_distributed_H(const DistributedInterpRows& rows, const MeshView& src_nodal,
                                 const std::vector<double>& local_src_values,
                                 std::size_t components, std::vector<double>& out_owned,
                                 int comm_handle) {
  if (components < 1) {
    return Failure{"E-RT-NODAL-DIST-ARG", "apply_distributed_H: components must be >= 1"};
  }
  const std::size_t nv = src_nodal.entity_count(0);
  if (local_src_values.size() != nv * components) {
    return Failure{"E-RT-NODAL-DIST-SIZE",
                   fmt::format("apply_distributed_H: local_src_values has {} entries, "
                               "expected {} ({} vertices x {} components)",
                               local_src_values.size(), nv * components, nv, components)};
  }

  const int stride = 1 + static_cast<int>(components);
  std::vector<double> buf;
  for (std::size_t i = 0; i < nv; ++i) {
    if (!src_nodal.is_owned(0, i)) continue;
    buf.push_back(static_cast<double>(src_nodal.global_id(0, i)));
    for (std::size_t c = 0; c < components; ++c) buf.push_back(local_src_values[i * components + c]);
  }
  const auto g = allgatherv_records(comm_handle, buf, stride);
  if (!g.ok()) return g.error();

  std::unordered_map<std::size_t, const double*> value_of;
  value_of.reserve(g.value().records);
  for (std::size_t k = 0; k < g.value().records; ++k) {
    const double* rec = &g.value().data[k * static_cast<std::size_t>(stride)];
    value_of[static_cast<std::size_t>(rec[0])] = rec + 1;
  }

  out_owned.assign(rows.rows.size() * components, 0.0);
  for (std::size_t r = 0; r < rows.rows.size(); ++r) {
    const DistributedInterpRow& row = rows.rows[r];
    for (std::size_t k = 0; k < row.src_global_ids.size(); ++k) {
      const auto it = value_of.find(row.src_global_ids[k]);
      if (it == value_of.end()) {
        return Failure{"E-RT-NODAL-DIST-OWNER",
                       fmt::format("apply_distributed_H: source vertex global id {} (needed "
                                   "by a row this rank owns) was not supplied as OWNED by "
                                   "any rank",
                                   row.src_global_ids[k])};
      }
      for (std::size_t c = 0; c < components; ++c) {
        out_owned[r * components + c] += row.weights[k] * it->second[c];
      }
    }
  }
  return Result<void>::success();
}

Result<void> apply_distributed_HT(const DistributedInterpRows& rows, const MeshView& src_nodal,
                                  const MeshView& eval_points, int eval_dim,
                                  const std::vector<double>& owned_loads, std::size_t components,
                                  std::vector<double>& out_src_nodal_local, int comm_handle) {
  if (components < 1) {
    return Failure{"E-RT-NODAL-DIST-ARG", "apply_distributed_HT: components must be >= 1"};
  }
  if (owned_loads.size() != rows.rows.size() * components) {
    return Failure{"E-RT-NODAL-DIST-SIZE",
                   fmt::format("apply_distributed_HT: owned_loads has {} entries, expected "
                               "{} ({} owned rows x {} components)",
                               owned_loads.size(), rows.rows.size() * components,
                               rows.rows.size(), components)};
  }

  // Pack (target_gid, contributor_eval_gid, contrib_0..contrib_{C-1}), stride
  // 2 + components -- see the header's own comment on why the contributor's
  // GLOBAL id, not its owning rank, is the sort key that makes the final sum
  // independent of rank count.
  const int stride = 2 + static_cast<int>(components);
  std::vector<double> buf;
  for (std::size_t r = 0; r < rows.rows.size(); ++r) {
    const DistributedInterpRow& row = rows.rows[r];
    const auto contributor_gid = eval_points.global_id(eval_dim, row.local_index);
    for (std::size_t k = 0; k < row.src_global_ids.size(); ++k) {
      buf.push_back(static_cast<double>(row.src_global_ids[k]));
      buf.push_back(static_cast<double>(contributor_gid));
      for (std::size_t c = 0; c < components; ++c) {
        buf.push_back(row.weights[k] * owned_loads[r * components + c]);
      }
    }
  }
  const auto g = allgatherv_records(comm_handle, buf, stride);
  if (!g.ok()) return g.error();

  // Sort every contribution by (target_gid, contributor_gid) -- a total
  // order that does not depend on which rank sent which record, so the
  // per-target summation order below is identical regardless of rank count.
  std::vector<std::size_t> order(g.value().records);
  for (std::size_t k = 0; k < order.size(); ++k) order[k] = k;
  const auto rec_at = [&](std::size_t k) { return &g.value().data[k * static_cast<std::size_t>(stride)]; };
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    const double* ra = rec_at(a);
    const double* rb = rec_at(b);
    if (ra[0] != rb[0]) return ra[0] < rb[0];
    return ra[1] < rb[1];
  });

  std::unordered_map<std::size_t, std::vector<double>> sum_by_gid;
  for (const std::size_t k : order) {
    const double* rec = rec_at(k);
    const auto gid = static_cast<std::size_t>(rec[0]);
    auto& acc = sum_by_gid[gid];
    if (acc.empty()) acc.assign(components, 0.0);
    for (std::size_t c = 0; c < components; ++c) acc[c] += rec[2 + c];
  }

  const std::size_t nv = src_nodal.entity_count(0);
  out_src_nodal_local.assign(nv * components, 0.0);
  for (std::size_t i = 0; i < nv; ++i) {
    const auto it = sum_by_gid.find(src_nodal.global_id(0, i));
    if (it == sum_by_gid.end()) continue;  // no row referenced this vertex: stays 0.
    for (std::size_t c = 0; c < components; ++c) out_src_nodal_local[i * components + c] = it->second[c];
  }
  return Result<void>::success();
}

}  // namespace fuse
