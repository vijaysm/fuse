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
#include "fuse/search/box_index.hpp"
#include "fuse/search/geometry.hpp"

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

// `comm_handle == fuse::search::kNullComm`: exactly one process, so the
// caller's own local records already ARE the global list -- a plain copy,
// touching no MPI symbol at all (this is what lets every function below
// compile and run at the base build configuration). Any other value is a real communicator
// (`MPI_Comm_c2f`'d by the caller): requires `FUSE_HAVE_MPI`, else rejected
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

// This rank's position in `comm_handle` (0 for `kNullComm`). ONLY ever
// called after a caller has already round-tripped `comm_handle` through
// `allgatherv_records` once successfully (both `build_distributed_interp_rows`
// and `agree_on_local_failure` below satisfy this): a real communicator on a
// no-MPI build would already have been rejected (`E-RT-NODAL-DIST-NOMPI`) by
// that earlier call, so by the time this runs, `comm_handle == kNullComm` (0
// is the genuinely correct answer) or `FUSE_HAVE_MPI` is on and `comm_handle`
// already named a working communicator (so `MPI_Comm_rank` on it cannot fail
// for a reason `allgatherv_records` itself would not already have caught).
[[nodiscard]] int comm_rank(search::CommHandle comm_handle) {
  if (comm_handle == search::kNullComm) return 0;
#if FUSE_HAVE_MPI
  const MPI_Comm comm = MPI_Comm_f2c(static_cast<MPI_Fint>(comm_handle));
  int r = 0;
  MPI_Comm_rank(comm, &r);
  return r;
#else
  return 0;
#endif
}

/// THE RULE "a rejection reachable after a collective must be an AGREED
/// outcome, not a rank-local one" -- the defect it prevents, in
/// `build_distributed_interp_rows`'s per-entity loop: a rank whose own local evaluation entities all lie
/// inside the assembled source mesh would return success while a
/// DIFFERENT rank, whose own local entities include one that is genuinely
/// out of domain (or defeats Newton), returned failure -- so at 3+ ranks
/// some ranks would return from this collective-adjacent function while
/// their partners, having taken the success branch, went on to call
/// FURTHER collectives the failing rank never reaches. This function is
/// the single place that turns "did THIS rank see a local failure" into
/// "did ANY rank see one, and if so, which is authoritative" -- called by
/// EVERY rank of `comm_handle`, INCLUDING one with zero local evaluation
/// entities or one that itself never failed, since the agreement itself
/// (not just its outcome) is the collective every rank must take part in.
///
/// `local_failed`/`local_code_id`/`local_entity_gid` describe (at most) the
/// FIRST local failure this rank's own per-entity scan hit (see
/// `FailCode` below for what `local_code_id` encodes); a rank that saw no
/// failure passes `local_failed = false` (the other two are then
/// ignored). Ties (more than one rank failing) are broken by LOWEST RANK,
/// the same determinism rule this codebase already uses for a shared
/// global id (general_interp.hpp's own tie-break) -- not because a lower
/// rank is more "correct," but so every rank's reconstructed message
/// agrees on which failure to report without needing to exchange the
/// dynamic message text itself over MPI.
///
/// Returns `Result<void>::success()` when no rank failed. Encodes each
/// rank's record as exactly 4 doubles (rank, failed flag, code id, entity
/// global id) and Allgathers ONE such record per rank -- fixed-size, so
/// every rank (even one with nothing of its own to report) contributes
/// exactly one record, satisfying the "still take part in the collective"
/// requirement structurally, not by convention.
enum class FailCode : int {
  kDomain = 1,
  kNewton = 2,
  kDegenerate = 3,
  kOwnerMissing = 4,  // apply_distributed_H's own E-RT-NODAL-DIST-OWNER
  kOther = 5,
  kNoCellVertices = 6,  // gather_global_source: MeshView::cell_vertices rejected
  kShape = 7,           // gather_global_source: unsupported shape or wrong corner count
  kGidAccess = 8,       // gather_global_source: global_id/centroid threw rank-locally
  kGidTooLarge = 9,     // a global id exceeds a double's 53-bit mantissa
};

/// Every global id in this file travels as a `double`
/// (`allgatherv_records`' own wire format, stride doubles per record) --
/// unreachable through `MoabMeshView` (GLOBAL_ID is a 32-bit int) but
/// `MeshView` is the contract, not MOAB specifically, and nothing here
/// stops a different backend from handing back a `std::size_t` above
/// 2^53, where it would silently lose precision in the Allgatherv record
/// and produce WRONG vertex matching with no diagnostic at all. Checked at
/// every point a gid enters a wire buffer, named by this one predicate so
/// the bound is asserted once rather than copy-pasted at each call site.
constexpr std::size_t kMaxExactDoubleGid = std::size_t{1} << 53;
[[nodiscard]] bool gid_fits_double(std::size_t gid) noexcept {
  return gid <= kMaxExactDoubleGid;
}

[[nodiscard]] const char* fail_code_str(int id) {
  switch (static_cast<FailCode>(id)) {
    case FailCode::kDomain: return "E-INTERP-DOMAIN";
    case FailCode::kNewton: return "E-INTERP-NEWTON";
    case FailCode::kDegenerate: return "E-INTERP-DEGENERATE";
    case FailCode::kOwnerMissing: return "E-RT-NODAL-DIST-OWNER";
    case FailCode::kNoCellVertices: return "E-MESH-NO-CELL-VERTICES";
    case FailCode::kShape: return "E-INTERP-SHAPE";
    case FailCode::kGidAccess: return "E-RT-NODAL-DIST-GID";
    case FailCode::kGidTooLarge: return "E-RT-NODAL-DIST-GID-RANGE";
    case FailCode::kOther: default: return "E-INTERP-DISTRIBUTED-ROW";
  }
}

[[nodiscard]] Result<void> agree_on_local_failure(search::CommHandle comm_handle, bool local_failed,
                                                   int local_code_id, std::size_t local_entity_gid,
                                                   const char* function_name) {
  const int my_rank = comm_rank(comm_handle);
  const std::vector<double> local_record = {
      static_cast<double>(my_rank), local_failed ? 1.0 : 0.0,
      static_cast<double>(local_code_id), static_cast<double>(local_entity_gid)};
  const auto g = allgatherv_records(comm_handle, local_record, /*stride=*/4);
  if (!g.ok()) return g.error();

  bool any_failed = false;
  int winning_rank = -1;
  int winning_code_id = 0;
  std::size_t winning_gid = 0;
  for (std::size_t k = 0; k < g.value().records; ++k) {
    const double* rec = &g.value().data[k * 4];
    if (rec[1] == 0.0) continue;
    const auto rec_rank = static_cast<int>(rec[0]);
    if (!any_failed || rec_rank < winning_rank) {
      any_failed = true;
      winning_rank = rec_rank;
      winning_code_id = static_cast<int>(rec[2]);
      winning_gid = static_cast<std::size_t>(rec[3]);
    }
  }
  if (!any_failed) return Result<void>::success();
  return Failure{
      fail_code_str(winning_code_id),
      fmt::format("{}: rank {} could not place its evaluation entity of global id {} ({}); "
                  "the outcome is AGREED across every rank of the communicator, so every rank "
                  "refuses together even though only rank {} saw the failure locally",
                  function_name, winning_rank, winning_gid, fail_code_str(winning_code_id),
                  winning_rank)};
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
  //
  // Every POST-collective early return in this file's loops is agreed
  // across ranks; that does not cover these two buffer-building loops, which run
  // BEFORE their own `allgatherv_records` call and can fail -- or, worse,
  // THROW -- rank-locally: `MeshView::centroid`/`global_id` throw
  // `std::out_of_range` by their own documented contract, and
  // `MoabMeshView::global_id` additionally throws `InvariantViolation` when
  // `gid_error` is set, which a partitioned MOAB read with unnumbered
  // vertices routinely produces on exactly one rank. Any of those outcomes
  // on one rank, while its peers proceed to `allgatherv_records`, is a hang
  // (an uncaught exception) or an undetected divergence (an early `return
  // Failure` here while other ranks continue). Both loops below instead
  // RECORD the first local failure (catching anything the accessors throw)
  // and stop adding to the buffer -- the SAME "record-and-stop" idiom
  // `build_distributed_interp_rows`'s own eval loop already uses just below
  // -- and `agree_on_local_failure` runs, unconditionally on every rank,
  // immediately after each loop and BEFORE that loop's own
  // `allgatherv_records`, so a rank-local failure never lets this rank skip
  // a collective its peers enter.
  std::vector<double> vbuf;
  bool v_failed = false;
  int v_fail_code = 0;
  std::size_t v_fail_gid = 0;
  const std::size_t nv = src.entity_count(0);
  for (std::size_t i = 0; i < nv && !v_failed; ++i) {
    if (!src.is_owned(0, i)) continue;
    try {
      const Point p = src.centroid(0, i);
      const std::size_t gid = src.global_id(0, i);
      if (!gid_fits_double(gid)) {
        v_failed = true;
        v_fail_code = static_cast<int>(FailCode::kGidTooLarge);
        v_fail_gid = gid;
        break;
      }
      vbuf.push_back(static_cast<double>(gid));
      vbuf.push_back(p.x);
      vbuf.push_back(p.y);
      vbuf.push_back(p.z);
    } catch (const std::exception&) {
      v_failed = true;
      v_fail_code = static_cast<int>(FailCode::kGidAccess);
      v_fail_gid = 0;  // the vertex that threw has no readable global id by definition
    }
  }
  {
    const auto agreed = agree_on_local_failure(comm_handle, v_failed, v_fail_code, v_fail_gid,
                                                "gather_global_source (vertices)");
    if (!agreed.ok()) return agreed.error();
  }
  auto vg = allgatherv_records(comm_handle, vbuf, 4);
  if (!vg.ok()) return vg.error();

  // ---- cells: (gid, shape, n, corner_gid x 8), stride 11, OWNED only ------
  std::vector<double> cbuf;
  bool c_failed = false;
  int c_fail_code = 0;
  std::size_t c_fail_gid = 0;
  const std::size_t nc = src.entity_count(dim);
  for (std::size_t i = 0; i < nc && !c_failed; ++i) {
    if (!src.is_owned(dim, i)) continue;
    try {
      const auto cv = src.cell_vertices(dim, i);
      if (!cv.ok()) {
        c_failed = true;
        c_fail_code = static_cast<int>(FailCode::kNoCellVertices);
        continue;
      }
      if (!shape_supported(cv.value().shape) ||
          cv.value().vertices.size() != shape_corner_count(cv.value().shape) ||
          cv.value().vertices.size() > 8) {
        c_failed = true;
        c_fail_code = static_cast<int>(FailCode::kShape);
        continue;
      }
      const std::size_t cell_gid = src.global_id(dim, i);
      if (!gid_fits_double(cell_gid)) {
        c_failed = true;
        c_fail_code = static_cast<int>(FailCode::kGidTooLarge);
        c_fail_gid = cell_gid;
        continue;
      }
      bool corner_gid_too_large = false;
      std::size_t bad_corner_gid = 0;
      for (std::size_t k = 0; k < cv.value().vertices.size() && !corner_gid_too_large; ++k) {
        const std::size_t corner_gid = src.global_id(0, cv.value().vertices[k]);
        if (!gid_fits_double(corner_gid)) {
          corner_gid_too_large = true;
          bad_corner_gid = corner_gid;
        }
      }
      if (corner_gid_too_large) {
        c_failed = true;
        c_fail_code = static_cast<int>(FailCode::kGidTooLarge);
        c_fail_gid = bad_corner_gid;
        continue;
      }
      cbuf.push_back(static_cast<double>(cell_gid));
      cbuf.push_back(static_cast<double>(static_cast<int>(cv.value().shape)));
      cbuf.push_back(static_cast<double>(cv.value().vertices.size()));
      for (std::size_t k = 0; k < 8; ++k) {
        cbuf.push_back(k < cv.value().vertices.size()
                           ? static_cast<double>(src.global_id(0, cv.value().vertices[k]))
                           : -1.0);
      }
    } catch (const std::exception&) {
      c_failed = true;
      c_fail_code = static_cast<int>(FailCode::kGidAccess);
      c_fail_gid = 0;
    }
  }
  {
    const auto agreed = agree_on_local_failure(comm_handle, c_failed, c_fail_code, c_fail_gid,
                                                "gather_global_source (cells)");
    if (!agreed.ok()) return agreed.error();
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

  // ---- Broad phase: one padded box per assembled source cell --------------
  //
  // The eval loop below does NOT test EVERY cell in the assembled global
  // mesh against EVERY locally-owned evaluation point -- unlike a plain
  // scan, it narrows to a handful of candidates through exactly the kind of
  // box index the serial counterpart (general_interp.hpp's
  // build_general_interp_weights) uses FIRST. At scale (N=1000, 1e6 cells
  // and 1e6 evaluation points) an unfiltered scan would feed
  // `invert_reference_map` cells arbitrarily far from the target: e.g. a
  // cell near x=0.997 Newton-inverted against a target at (0.0005,
  // 0.0005), the opposite corner of the domain, where a damped Gauss-Newton
  // started from THAT cell's own reference centre does not converge within
  // 50 iterations (last residual 1.92e-12 against a 1e-12 tolerance) --
  // even though the GENUINE containing cell (global id 0) had already been
  // found as a tight match earlier in the same scan. The scan would never
  // stop looking once it had the right answer, and a later, unrelated
  // candidate's non-convergence would abort the whole evaluation point.
  // Building the same kind of index build_general_interp_weights uses,
  // over this function's own (Allgatherv-replicated) `cells`, and querying
  // it per evaluation point the same way, avoids both: Newton is only ever
  // invoked on a small, genuinely nearby candidate set, so a stray distant
  // cell's non-convergence cannot reach this code, and the whole search is
  // O(n_eval * candidates) rather than O(n_eval * n_cells) -- the
  // candidate count being a small constant on a structured mesh.
  //
  // The filter does not change which cell wins or its computed natural
  // coordinates: the winning cell is chosen by the lowest-global-id
  // tie-break, which depends only on which candidates pass tier (a) or (b),
  // never on scan order, and the padding below matches kRefEpsLoose, the
  // loosest bound either tier accepts, so every genuine tier (a)/(b)
  // candidate is still found. (A 1-ULP VALUE disagreement in
  // NodalRemapRunnerMpi.ConsistentTransposeIsBitIdenticalAcrossRankCounts
  // is a separate matter, unrelated to this filter.)
  std::vector<search::Box> cell_boxes;
  cell_boxes.reserve(cells.size());
  for (const Cell& cell : cells) {
    search::Point lo{std::numeric_limits<double>::max(), std::numeric_limits<double>::max(),
                     std::numeric_limits<double>::max()};
    search::Point hi{std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest(),
                     std::numeric_limits<double>::lowest()};
    for (std::size_t k = 0; k < cell.n; ++k) {
      lo.x = std::min(lo.x, cell.p[k].x); hi.x = std::max(hi.x, cell.p[k].x);
      lo.y = std::min(lo.y, cell.p[k].y); hi.y = std::max(hi.y, cell.p[k].y);
      lo.z = std::min(lo.z, cell.p[k].z); hi.z = std::max(hi.z, cell.p[k].z);
    }
    // Padded by kRefEpsLoose -- the SAME bound the fallback tier below
    // already accepts a candidate within, not the default (rounding-scale)
    // tolerance -- so the broad phase can never drop a genuine tier (b)
    // candidate before the narrow phase gets to classify it (identical
    // reasoning to build_general_interp_weights' own box padding).
    cell_boxes.push_back(
        search::padded(search::Box{lo, hi}, search::Tolerance{.relative = kRefEpsLoose}));
  }
  const auto box_index_r = search::BoxIndex::build(cell_boxes);
  if (!box_index_r.ok()) return Failure{box_index_r.error().code, box_index_r.error().message};
  const search::BoxIndex& box_index = box_index_r.value();

  DistributedInterpRows out;
  const std::size_t n_eval = eval_points.entity_count(eval_dim);

  // One query point per LOCALLY OWNED evaluation entity -- `owned_j[oi]`
  // names which entity index (into `eval_points`) query `oi` answers for.
  std::vector<std::size_t> owned_j;
  std::vector<search::Point> owned_targets;
  owned_j.reserve(n_eval);
  owned_targets.reserve(n_eval);
  for (std::size_t j = 0; j < n_eval; ++j) {
    if (!eval_points.is_owned(eval_dim, j)) continue;
    const Vec3 t = to_vec(eval_points.centroid(eval_dim, j));
    owned_j.push_back(j);
    owned_targets.push_back(search::Point{t.x, t.y, t.z});
  }
  const auto hits_r = box_index.query_points(owned_targets);
  if (!hits_r.ok()) return Failure{hits_r.error().code, hits_r.error().message};
  const search::QueryResult& hits = hits_r.value();
  const int k_nearest = static_cast<int>(std::min<std::size_t>(8, cells.size()));

  // The branch below does not return
  // directly. "No candidate contains this point" (E-INTERP-DOMAIN) depends
  // on THIS rank's own local evaluation entities, so returning it straight
  // from this loop would let
  // a rank with no bad point return success and go on to call a FURTHER
  // collective its partners, having hit the failure, never reach.
  // `local_failed`/`local_fail_code_id`/`local_fail_entity_gid` record the
  // FIRST such local failure (if any) and the loop then simply stops
  // adding rows; `agree_on_local_failure` below is the single place every
  // rank -- including one with zero local evaluation entities, or one
  // that never fails at all -- takes part in the SAME collective and
  // returns the SAME outcome.
  //
  // A per-candidate invert_reference_map failure (E-INTERP-NEWTON / E-INTERP-DEGENERATE) does not abort the
  // evaluation point outright -- it means only that THIS candidate is not
  // usable, exactly like one that converges but lands outside the
  // reference domain. The broad phase above already keeps the candidate
  // set small and genuinely nearby, so a converging failure here is rare;
  // when every candidate for a point fails or lands outside both tiers,
  // the outcome is E-INTERP-DOMAIN below, as for "no containing cell
  // found."
  bool local_failed = false;
  int local_fail_code_id = 0;
  std::size_t local_fail_entity_gid = 0;

  for (std::size_t oi = 0; oi < owned_j.size() && !local_failed; ++oi) {
    const std::size_t j = owned_j[oi];
    const Vec3 target = to_vec(eval_points.centroid(eval_dim, j));

    // TWO-TIER SELECTION -- identical rule to general_interp.hpp's serial
    // builder (see that file's kRefEpsTight/kRefEpsLoose comments for the
    // derivation); the candidate list's SOURCE is now the SAME kind of
    // broad-phase box query that builder uses, over this function's own
    // Allgatherv-replicated `cells`. Tier (a): a candidate that GENUINELY
    // contains the point (kRefEpsTight) wins outright, tie-broken by
    // lowest source global id ONLY among tight matches. Tier (b), reached
    // only when no candidate passes tier (a): the single least-violating
    // candidate (raw, eps = 0, score), accepted if it also clears
    // kRefEpsLoose -- the genuine curved-surface faceting gap, never a
    // tie-break.
    bool have_tight = false;
    std::size_t tight_c = 0;
    std::size_t tight_gid = 0;
    std::array<double, 3> tight_u{};

    bool have_fallback = false;
    std::size_t fallback_c = 0;
    std::size_t fallback_gid = 0;  // the tie-break key, not just a diagnostic
    double fallback_score = std::numeric_limits<double>::max();
    std::array<double, 3> fallback_u{};
    bool fallback_passes_loose = false;

    const auto consider = [&](std::size_t c) {
      const auto inv = invert_reference_map(cells[c], target);
      if (!inv.ok()) {
        // Skip just this candidate -- another nearby one, or
        // this same point already found as a tight match at a lower-gid
        // cell, still decides the row.
        return;
      }
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
        return;
      }

      const double lateral = reference_violation(cells[c].shape, inv.value().u, 0.0);
      const double normal_gap = surf_ok ? 0.0 : (inv.value().residual / diam);
      const double score = lateral + normal_gap * normal_gap;
      const bool passes_loose =
          surf_ok && in_reference_domain(cells[c].shape, inv.value().u, kRefEpsLoose);
      // A STRICT `score <` leaves an exact tie decided by
      // iteration order; tie-broken by lowest global id -- a property of
      // the cell, not of the gather or the query order -- the SAME rule
      // tier (a) already uses above.
      if (!have_fallback || score < fallback_score ||
          (score == fallback_score && gid < fallback_gid)) {
        have_fallback = true;
        fallback_c = c;
        fallback_gid = gid;
        fallback_score = score;
        fallback_u = inv.value().u;
        fallback_passes_loose = passes_loose;
      }
    };

    for (std::size_t p = hits.offsets[oi]; p < hits.offsets[oi + 1]; ++p) {
      consider(hits.indices[p]);
    }
    // A safety net for a point whose padded box query found nothing at all
    // (genuinely just outside every cell's own padded box, e.g. at the
    // mesh's outer boundary): the k-nearest cell boxes, the same fallback
    // build_general_interp_weights' own ClampToBoundary path uses, so this
    // point still gets a fair tier (b) chance before being refused.
    if (!have_tight && !have_fallback) {
      const auto nn_r = box_index.query_nearest({owned_targets[oi]}, k_nearest);
      if (nn_r.ok()) {
        for (std::size_t p = nn_r.value().offsets[0]; p < nn_r.value().offsets[1]; ++p) {
          consider(nn_r.value().indices[p]);
        }
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
      local_failed = true;
      local_fail_code_id = static_cast<int>(FailCode::kDomain);
      local_fail_entity_gid = eval_points.global_id(eval_dim, j);
      break;
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

  const auto agreed = agree_on_local_failure(comm_handle, local_failed, local_fail_code_id,
                                             local_fail_entity_gid, "build_distributed_interp_rows");
  if (!agreed.ok()) return agreed.error();
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
  // This rank's own vertex gids enter the SAME double-
  // valued wire format gather_global_source's loops do; a gid above
  // kMaxExactDoubleGid would silently lose precision there too. Record-
  // and-stop, then agree, before allgatherv_records --
  // rather than returning straight from the loop.
  bool gid_range_failed = false;
  std::size_t gid_range_fail_gid = 0;
  for (std::size_t i = 0; i < nv && !gid_range_failed; ++i) {
    if (!src_nodal.is_owned(0, i)) continue;
    const std::size_t gid = src_nodal.global_id(0, i);
    if (!gid_fits_double(gid)) {
      gid_range_failed = true;
      gid_range_fail_gid = gid;
      continue;
    }
    buf.push_back(static_cast<double>(gid));
    for (std::size_t c = 0; c < components; ++c) buf.push_back(local_src_values[i * components + c]);
  }
  {
    const auto agreed = agree_on_local_failure(comm_handle, gid_range_failed,
                                                static_cast<int>(FailCode::kGidTooLarge),
                                                gid_range_fail_gid, "apply_distributed_H");
    if (!agreed.ok()) return agreed.error();
  }
  const auto g = allgatherv_records(comm_handle, buf, stride);
  if (!g.ok()) return g.error();

  // `gather_global_source` rejects a duplicated OWNED gid
  // by name (`E-RT-NODAL-DIST-OWNER`, the ownership contract
  // `MeshView::is_owned` must agree with itself about) -- this loop must
  // not silently keep whichever record happened to overwrite `value_of`
  // last for the same hazard. Safe to detect and return directly HERE,
  // unlike a pre-collective check: `g.value()` is the Allgatherv's own
  // output, already identical on every rank, so this either rejects on
  // every rank or on none, with no reconciliation needed.
  std::unordered_map<std::size_t, const double*> value_of;
  value_of.reserve(g.value().records);
  for (std::size_t k = 0; k < g.value().records; ++k) {
    const double* rec = &g.value().data[k * static_cast<std::size_t>(stride)];
    const auto gid = static_cast<std::size_t>(rec[0]);
    const auto [it, inserted] = value_of.emplace(gid, rec + 1);
    if (!inserted) {
      return Failure{"E-RT-NODAL-DIST-OWNER",
                     fmt::format("apply_distributed_H: source vertex global id {} was "
                                 "reported OWNED by two ranks", gid)};
    }
    (void)it;
  }

  // Same hazard class as
  // build_distributed_interp_rows' fix above, and the SAME remedy --
  // whether a rows' src_global_ids miss the globally-gathered value_of map
  // depends on THIS rank's own rows, so it must never return directly. The
  // scan still runs to completion locally (out_owned is filled wherever
  // possible) so a rank that never fails is not left with a half-filled
  // buffer if a DIFFERENT rank's outcome turns out to be the failing one;
  // the buffer is only ever read by the caller once agree_on_local_failure
  // below confirms success on every rank.
  out_owned.assign(rows.rows.size() * components, 0.0);
  bool local_failed = false;
  std::size_t local_fail_gid = 0;
  for (std::size_t r = 0; r < rows.rows.size(); ++r) {
    const DistributedInterpRow& row = rows.rows[r];
    for (std::size_t k = 0; k < row.src_global_ids.size(); ++k) {
      const auto it = value_of.find(row.src_global_ids[k]);
      if (it == value_of.end()) {
        if (!local_failed) {
          local_failed = true;
          local_fail_gid = row.src_global_ids[k];
        }
        continue;
      }
      for (std::size_t c = 0; c < components; ++c) {
        out_owned[r * components + c] += row.weights[k] * it->second[c];
      }
    }
  }
  const auto agreed = agree_on_local_failure(comm_handle, local_failed,
                                             static_cast<int>(FailCode::kOwnerMissing), local_fail_gid,
                                             "apply_distributed_H");
  if (!agreed.ok()) return agreed.error();
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

Result<DistributedRemapPlan> build_distributed_remap_plan(const DistributedInterpRows& rows,
                                                          const MeshView& src_nodal,
                                                          const MeshView& eval_points,
                                                          int eval_dim,
                                                          std::size_t components,
                                                          search::CommHandle comm_handle) {
  if (components < 1) {
    return Failure{"E-RT-NODAL-DIST-ARG",
                   "build_distributed_remap_plan: components must be >= 1"};
  }
  // Written search::kNullComm == comm_handle (reversed) rather than the
  // textually-identical comm_handle == search::kNullComm
  // allgatherv_records already uses just above: tools/mutate.py's own
  // anchor matching needs `old` to appear exactly once per mutation id,
  // and nodal-remap-dist-nullcomm-swallows-real-communicator's `old`
  // string IS that exact line -- kept distinct here on purpose, not by
  // accident.
  if (search::kNullComm == comm_handle) {
    return Failure{"E-RT-NODAL-DIST-NOMPI",
                   "build_distributed_remap_plan: comm_handle is fuse::search::kNullComm; a "
                   "single-rank caller has no index resolution worth amortising -- call "
                   "apply_distributed_H/apply_distributed_HT directly instead"};
  }
#if !FUSE_HAVE_MPI
  (void)rows;
  (void)src_nodal;
  (void)eval_points;
  (void)eval_dim;
  (void)components;
  return Failure{"E-RT-NODAL-DIST-NOMPI",
                 "build_distributed_remap_plan: comm_handle names a real communicator, but "
                 "this build of FUSE has no MPI; configure with -DFUSE_ENABLE_MPI=ON"};
#else
  // Step 1: flatten rows into a per-slot (row, k) array -- the order
  // DistributedRemapPlan::weights_flat and the component-expanded
  // GhostPattern below both use.
  std::vector<std::size_t> row_offset(rows.rows.size() + 1, 0);
  for (std::size_t r = 0; r < rows.rows.size(); ++r) {
    row_offset[r + 1] = row_offset[r] + rows.rows[r].weights.size();
  }
  const std::size_t total_slots = row_offset.back();
  std::vector<double> weights_flat(total_slots, 0.0);
  std::vector<std::size_t> src_gid_flat(total_slots, 0);
  for (std::size_t r = 0; r < rows.rows.size(); ++r) {
    const DistributedInterpRow& row = rows.rows[r];
    for (std::size_t k = 0; k < row.weights.size(); ++k) {
      weights_flat[row_offset[r] + k] = row.weights[k];
      src_gid_flat[row_offset[r] + k] = row.src_global_ids[k];
    }
  }

  // Step 2: one-time Allgather of (this rank, gid, local index) for every
  // rank's OWNED nodal vertices -- gather_global_source's own vertex
  // gather, minus the coordinates that one carries and this one does
  // not need, plus the sending rank tagged onto each record directly
  // (allgatherv_records' own flattened output does not otherwise expose
  // which rank contributed which record).
  const int my_rank = comm_rank(comm_handle);
  std::vector<double> owned_buf;
  const std::size_t nv = src_nodal.entity_count(0);
  for (std::size_t i = 0; i < nv; ++i) {
    if (!src_nodal.is_owned(0, i)) continue;
    owned_buf.push_back(static_cast<double>(my_rank));
    owned_buf.push_back(static_cast<double>(src_nodal.global_id(0, i)));
    owned_buf.push_back(static_cast<double>(i));
  }
  const auto g = allgatherv_records(comm_handle, owned_buf, /*stride=*/3);
  if (!g.ok()) return g.error();

  std::unordered_map<std::size_t, std::pair<int, std::size_t>> owner_of;
  owner_of.reserve(g.value().records);
  for (std::size_t k = 0; k < g.value().records; ++k) {
    const double* rec = &g.value().data[k * 3];
    const auto rank = static_cast<int>(rec[0]);
    const auto gid = static_cast<std::size_t>(rec[1]);
    const auto local_idx = static_cast<std::size_t>(rec[2]);
    const auto [it, inserted] = owner_of.emplace(gid, std::make_pair(rank, local_idx));
    if (!inserted) {
      return Failure{"E-RT-NODAL-DIST-OWNER",
                     fmt::format("build_distributed_remap_plan: source vertex global id {} "
                                 "was reported OWNED by two ranks",
                                 gid)};
    }
  }

  // Step 3: component-expanded GhostPattern -- slot (s, c) resolves to
  // the owner's own local position scaled by components, the SAME
  // entity-major convention apply_distributed_H's local_src_values
  // already uses, so the plan's exchanger can be driven directly off a
  // caller's existing per-arc buffer.
  linalg::GhostPattern pattern;
  pattern.owner_rank.assign(total_slots * components, 0);
  pattern.remote_local_index.assign(total_slots * components, 0);
  for (std::size_t s = 0; s < total_slots; ++s) {
    const auto it = owner_of.find(src_gid_flat[s]);
    if (it == owner_of.end()) {
      return Failure{"E-RT-NODAL-DIST-OWNER",
                     fmt::format("build_distributed_remap_plan: a row references source "
                                 "vertex global id {}, which no rank reported owning",
                                 src_gid_flat[s])};
    }
    const int owner_rank = it->second.first;
    const std::size_t owner_local = it->second.second;
    for (std::size_t c = 0; c < components; ++c) {
      pattern.owner_rank[s * components + c] = owner_rank;
      pattern.remote_local_index[s * components + c] =
          static_cast<std::int64_t>(owner_local * components + c);
    }
  }

  // Step 4: the ghost-fill pattern `apply_distributed_HT`'s own
  // contract needs (DistributedRemapPlan's own header comment) -- one
  // slot per LOCAL GHOST entity of `src_nodal` (component-expanded),
  // mapping to that SAME entity's owner, reusing `owner_of` again
  // rather than a second Allgather. A local entity this rank itself
  // owns is never a slot here: `update()`'s contract already leaves
  // `owned[i]` as whatever the caller wrote, so refusing to overwrite
  // it with a copy of itself is the identity either way, and omitting
  // the slot entirely is simpler than including a no-op one.
  std::vector<std::size_t> ghost_local_index;
  for (std::size_t i = 0; i < nv; ++i) {
    if (src_nodal.is_owned(0, i)) continue;
    ghost_local_index.push_back(i);
  }
  linalg::GhostPattern fill_pattern;
  fill_pattern.owner_rank.assign(ghost_local_index.size() * components, 0);
  fill_pattern.remote_local_index.assign(ghost_local_index.size() * components, 0);
  for (std::size_t g = 0; g < ghost_local_index.size(); ++g) {
    const std::size_t i = ghost_local_index[g];
    const auto it = owner_of.find(src_nodal.global_id(0, i));
    if (it == owner_of.end()) {
      return Failure{"E-RT-NODAL-DIST-OWNER",
                     fmt::format("build_distributed_remap_plan: local ghost vertex global id "
                                 "{} is not owned by any rank", src_nodal.global_id(0, i))};
    }
    const int owner_rank = it->second.first;
    const std::size_t owner_local = it->second.second;
    for (std::size_t c = 0; c < components; ++c) {
      fill_pattern.owner_rank[g * components + c] = owner_rank;
      fill_pattern.remote_local_index[g * components + c] =
          static_cast<std::int64_t>(owner_local * components + c);
    }
  }

  DistributedRemapPlan plan;
  plan.row_offset = std::move(row_offset);
  plan.weights_flat = std::move(weights_flat);
  plan.components = components;
  plan.owned_n = nv * components;
  plan.exchanger =
      linalg::make_mpi_exchanger(static_cast<int>(comm_handle), pattern, plan.owned_n);
  plan.ghost_fill_exchanger =
      linalg::make_mpi_exchanger(static_cast<int>(comm_handle), fill_pattern, plan.owned_n);

  // Resolve the H^T scatter-add's own FIXED summation order once here, instead of
  // leaving it to `Kokkos::atomic_add`'s MPI-arrival order -- recovers
  // `apply_distributed_HT`'s own "bit-identical across rank counts"
  // guarantee for this plan-based path too. See DistributedRemapPlan's
  // own header comment for the full rationale; this is the identical
  // (target_gid, contributor_gid) sort `apply_distributed_HT` already
  // performs per call, done here ONCE against the exchanger's own
  // ADJOINT "answer order" instead of a fresh Allgatherv'd term list
  // every period.
  const auto packed = std::dynamic_pointer_cast<const linalg::PackedExchanger>(plan.exchanger);
  if (!packed) {
    return Failure{"E-RT-NODAL-DIST-MPI",
                   "build_distributed_remap_plan: make_mpi_exchanger returned an exchanger "
                   "with no PackedExchanger facet -- unreachable for a real communicator"};
  }
  const auto& unpack_idx = packed->unpack_index();  // REQUEST order, length total_slots*components
  const auto& pack_idx = packed->pack_index();       // ANSWER order, length this rank's own incoming count

  // This rank's own per-slot contributor id (the OWNING evaluation
  // entity's global id, the SAME value repeated across a row's `k`
  // weights and `components` components), in the SAME slot order
  // `weights_flat`/`pattern` already use -- gathered into REQUEST order
  // by `unpack_index()` exactly the way `DeviceGhostExchanger::
  // update_adjoint()` gathers a real contribution value (comm.hpp's own
  // NAMING paragraph: `unpack_index()` is this rank's REQUEST side).
  bool gid_range_failed = false;
  std::size_t gid_range_fail_gid = 0;
  std::vector<double> contributor_gid(total_slots * components, 0.0);
  for (std::size_t r = 0; r < rows.rows.size(); ++r) {
    const std::size_t gid = eval_points.global_id(eval_dim, rows.rows[r].local_index);
    if (!gid_fits_double(gid) && !gid_range_failed) {
      gid_range_failed = true;
      gid_range_fail_gid = gid;
    }
    for (std::size_t k = 0; k < rows.rows[r].weights.size(); ++k) {
      for (std::size_t c = 0; c < components; ++c) {
        contributor_gid[(plan.row_offset[r] + k) * components + c] = static_cast<double>(gid);
      }
    }
  }
  {
    const auto agreed = agree_on_local_failure(comm_handle, gid_range_failed,
                                               static_cast<int>(FailCode::kGidTooLarge),
                                               gid_range_fail_gid, "build_distributed_remap_plan");
    if (!agreed.ok()) return agreed.error();
  }

  std::vector<double> req_buf(unpack_idx.size(), 0.0);
  for (std::size_t k = 0; k < unpack_idx.size(); ++k) {
    req_buf[k] = contributor_gid[static_cast<std::size_t>(unpack_idx[k])];
  }
  std::vector<double> ans_buf(pack_idx.size(), 0.0);
  const auto tag_ex = packed->exchange_packed_adjoint(req_buf.data(), ans_buf.data());
  if (!tag_ex.ok()) return tag_ex.error();

  // Group every answer slot by its OWNED target position (`pack_idx`'s
  // own value -- the identical target `DeviceGhostExchanger::
  // update_adjoint()`'s own atomic-add would scatter into), sorted
  // ascending by the gid this slot's contribution carries -- the fixed,
  // decomposition-independent order described above.
  std::vector<std::vector<std::pair<double, std::size_t>>> buckets(plan.owned_n);
  for (std::size_t k = 0; k < pack_idx.size(); ++k) {
    const auto t = static_cast<std::size_t>(pack_idx[k]);
    buckets[t].emplace_back(ans_buf[k], k);
  }
  plan.target_row_offset.assign(plan.owned_n + 1, 0);
  plan.sorted_answer_slot.assign(pack_idx.size(), 0);
  std::size_t pos = 0;
  for (std::size_t t = 0; t < plan.owned_n; ++t) {
    std::sort(buckets[t].begin(), buckets[t].end());
    for (const auto& [gid, k] : buckets[t]) plan.sorted_answer_slot[pos++] = k;
    plan.target_row_offset[t + 1] = pos;
  }

  return plan;
#endif
}

}  // namespace fuse