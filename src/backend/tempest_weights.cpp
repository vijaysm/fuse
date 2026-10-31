// The TempestRemap high-order / monotone backend.
//
// This translation unit is compiled UNCONDITIONALLY. Without TempestRemap it
// still defines every function `high_order_backend.hpp` declares; they just say
// no. An always-visible
// declaration whose definition is inside an #ifdef is the arrangement to
// avoid, so the rule here is:
//
// A NOTE ON THE GUARDS: FUSE's feature macros come from config.hpp via
// `#cmakedefine01`, so they are always DEFINED, as 0 or 1. Every test on one
// must therefore be `#if FUSE_HAVE_X`, never `#ifdef` -- which is
// unconditionally true and would compile the MOAB-only branch in a build
// without MOAB. MOAB's own macros (MOAB_HAVE_*) are `-D`-style, so `#ifndef`
// is correct for those.
// declare once, define once, both outside the guard.
//
// ---------------------------------------------------------------------------
// WHICH TEMPESTREMAP ENTRY POINTS, AND WHY NOT THE OBVIOUS ONES
// ---------------------------------------------------------------------------
//
// MOAB ships two ways in. The obvious one is `moab::TempestRemapper` plus
// `moab::TempestOnlineMap`; it is what `mbtempest` and iMOAB use. It is not
// usable from a serial build of this library:
//
//   * `moab/Remapping/Remapper.hpp` includes `moab/ParallelComm.hpp` whenever
//     MOAB was built with MPI, so it needs `mpi.h` on the include path even for
//     a serial consumer -- and MOAB's own pkg-config Cflags do NOT carry the
//     MPI include directory, so it is not even reliably available.
//   * worse, `TempestOnlineMap` dereferences `m_pcomm` unconditionally inside
//     its `#ifdef MOAB_HAVE_MPI` blocks (`TempestLinearRemap.cpp` lines 294,
//     475, 579, 762 in MOAB 5.6.0). Constructed with the documented default
//     `pcomm = nullptr`, weight generation segfaults in
//     `moab::ProcConfig::proc_comm()`. Verified under lldb.
//
// So this file calls TempestRemap's OWN API instead -- `GenerateOverlapWithMeshes`
// and `GenerateOfflineMapWithMeshes` from <TempestRemapAPI.h>. That library has
// no MPI in it, needs no `moab::Core`, and its `Mesh` face indices are exactly
// the order we hand it, which removes the whole GLOBAL_ID-versus-position
// guessing game that the MOAB `Intx2Mesh` path has to do.
//
// ---------------------------------------------------------------------------
// TEMPESTREMAP REPORTS FAILURE BY PRINTING AND RETURNING SUCCESS
// ---------------------------------------------------------------------------
//
// Both entry points end in
//
//     } catch (Exception & e) { Announce(e.ToString().c_str()); return (0); }
//     catch (...) { return (0); }
//
// (GenerateOverlapMesh.cpp:133, GenerateOfflineMap.cpp:867). Zero is also the
// success return. The same is true of the silent accuracy loss: when its
// least-squares fit is too ill-conditioned, `FiniteVolumeTools.cpp:1420` prints
// "WARNING: Poor conditioning in fit matrix (...); dropping to 1st order" and
// carries on, so a caller who asked for order 4 gets an order-1 row and no
// error. Observed here on every cell of a cubed-sphere pair at order 5, and on
// some cells of a coarse pair at order 2.
//
// Therefore this file redirects TempestRemap's `Announce` output into a
// temporary file for the duration of each call and reads it back. That text is
// the primary error channel, not a nicety. It is also why the remap path is
// serial and non-reentrant: `g_fpAnnounceOutput` is a process-wide `FILE*`.
//
// Hardening: two threads each constructing an `AnnounceCapture`
// concurrently could previously corrupt each other's diagnostics --
// thread B's constructor could capture thread A's temp file as `previous_`
// instead of the true prior buffer, and thread A's destructor could then
// restore that stolen pointer over whatever thread B had installed, leaving
// EITHER thread's captured text incomplete, wrong, or attributed to the other
// thread's build. Because printing IS the only failure channel (the header
// comment above), a corrupted capture means a real build failure or a silent
// order degradation goes unreported, which is worse than a crash. `build_high_order_weights`
// now takes a process-wide `std::mutex` (`announce_mutex()` below) for the
// entire span from before `AnnounceCapture` is constructed to after it is
// destroyed and its text consumed, so concurrent callers now SERIALIZE rather
// than corrupt each other. This does not make TempestRemap itself reentrant --
// only one build may be in flight process-wide at a time, same as before --
// it converts silent corruption under concurrent use into safe serialization.

// Before any FUSE_HAVE_* read: an undefined feature macro is read as 0 by
// the preprocessor, which would silently compile the rejecting stub.
#include "fuse/base/config.hpp"

#include "high_order_backend.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

#if FUSE_HAVE_TEMPEST
// The macro trap: MOAB spells it MOAB_HAVE_TEMPESTREMAP, not MOAB_HAVE_TEMPEST.
// FUSE_HAVE_TEMPEST is set by cmake/FindSomaMOAB.cmake only after a real
// compile-and-link of the two entry points below, but assert the header story
// agrees anyway -- a prefix change that silently drops TempestRemap would
// otherwise surface as a link error a long way from here.
#include <moab/MOABConfig.h>
#ifndef MOAB_HAVE_TEMPESTREMAP
#error "FUSE_HAVE_TEMPEST is set but MOAB's headers do not define MOAB_HAVE_TEMPESTREMAP"
#endif

#include <Announce.h>
#include <GridElements.h>
#include <OfflineMap.h>
#include <SparseMatrix.h>
#include <TempestRemapAPI.h>
#endif  // FUSE_HAVE_TEMPEST

namespace fuse::remap_backend {

#if !FUSE_HAVE_TEMPEST

bool high_order_backend_available() noexcept { return false; }

const char* high_order_backend_name() noexcept { return "none"; }

Result<HighOrderResult> build_high_order_weights(const HighOrderRequest&,
                                                 const PolygonSurface&,
                                                 const PolygonSurface&) {
  return Failure{
      "E-REMAP-NOTEMPEST",
      "high-order and monotone remap weights need TempestRemap, and this "
      "build's MOAB was configured without it (FUSE_HAVE_TEMPEST is off). "
      "Rebuild MOAB with --download-tempestremap (or --with-tempestremap) and "
      "reconfigure; see docs/BUILDING.md. The first-order conservative "
      "supermesh map is unaffected and remains available."};
}

#else  // FUSE_HAVE_TEMPEST

namespace {

/// Highest FV reconstruction order this backend will accept.
///
/// TempestRemap itself imposes no cap: `LinearRemapFVtoFV` builds an
/// `nOrder*(nOrder+1)/2`-coefficient least-squares fit for any `nOrder`, and
/// when the fit matrix is too ill-conditioned it drops that cell to first order
/// and prints a warning (`Defines.h`: FVConditionNumberThreshold = 1e7).
/// Measured on a 6x8^2 against 6x6^2 cubed-sphere pair: orders 1-4 build with
/// no warning at all; order 5 warns on essentially every cell, with condition
/// estimates from 1.5e7 to 2.2e13. An order the library cannot deliver is
/// rejected up front rather than delivered degraded -- and the per-cell warning
/// is ALSO caught below, because a mesh pair worse than the one measured can
/// degrade at order 2.
constexpr int kMaxOrder = 4;

/// A face with fewer than three distinct corners has no area.
constexpr int kMinCorners = 3;

/// Captures TempestRemap's `Announce` stream for the life of the object.
///
/// `AnnounceSetOutputBuffer(nullptr)` is NOT safe -- Announce.cpp fprintf()s to
/// the pointer without a null check -- so a failure to open the temporary file
/// leaves the stream where it was and `text()` says so, rather than crashing or
/// pretending the run was quiet.
class AnnounceCapture {
 public:
  AnnounceCapture() : previous_(AnnounceGetOutputBuffer()), file_(std::tmpfile()) {
    if (file_ != nullptr) AnnounceSetOutputBuffer(file_);
  }
  AnnounceCapture(const AnnounceCapture&) = delete;
  AnnounceCapture& operator=(const AnnounceCapture&) = delete;
  ~AnnounceCapture() {
    // Restore even if `previous_` was the default stdout: leaving a dangling
    // FILE* in a global is how the next unrelated TempestRemap user crashes.
    AnnounceSetOutputBuffer(previous_);
    if (file_ != nullptr) std::fclose(file_);
  }

  /// Everything printed so far. Non-destructive: the stream stays open and
  /// positioned at the end, so a second call after more work returns more text.
  [[nodiscard]] std::string text() const {
    if (file_ == nullptr) {
      return "(TempestRemap's diagnostic stream could not be captured: "
             "std::tmpfile() failed)";
    }
    std::fflush(file_);
    const long end = std::ftell(file_);
    if (end <= 0) return {};
    std::string out(static_cast<std::size_t>(end), '\0');
    std::rewind(file_);
    const std::size_t got =
        std::fread(out.data(), 1, static_cast<std::size_t>(end), file_);
    out.resize(got);
    std::fseek(file_, 0, SEEK_END);
    return out;
  }

 private:
  std::FILE* previous_;
  std::FILE* file_;
};

/// The last `n` lines of `log`, so an error message carries the part that
/// matters without pasting a megabyte of per-face chatter.
[[nodiscard]] std::string tail(const std::string& log, std::size_t n) {
  if (log.empty()) return "(no diagnostics)";
  std::size_t pos = log.size();
  std::size_t seen = 0;
  while (pos > 0 && seen < n) {
    const std::size_t nl = log.rfind('\n', pos - 1);
    if (nl == std::string::npos) return log;
    pos = nl;
    ++seen;
  }
  return log.substr(pos == 0 ? 0 : pos + 1);
}

[[nodiscard]] bool contains(const std::string& hay, const char* needle) {
  return hay.find(needle) != std::string::npos;
}

/// The mutex serializing every high-order build against `AnnounceCapture`'s
/// process-wide `FILE*` hazard. Function-local static so it is
/// constructed exactly once, on first use, with no static-init-order question
/// against anything else in this translation unit.
std::mutex& announce_mutex() {
  static std::mutex m;
  return m;
}

/// Build a TempestRemap `Mesh` from a polygon soup.
///
/// Three things happen here that are specific to THIS backend and are therefore
/// not done to the caller's data:
///
///   1. coordinates are divided by `radius`. TempestRemap works on the unit
///      sphere -- its areas are spherical excesses of great-circle polygons and
///      a radius other than 1 makes `CalculateFaceAreas` disagree with its own
///      overlap areas. Weights are dimensionless ratios, so this is invisible in
///      the result; the areas we return are multiplied back by `radius^2`.
///   2. consecutive duplicate corners are dropped. MOAB pads a polygon by
///      repeating its last node; TempestRemap's `RemoveZeroEdges` handles that
///      too, but doing it here keeps the face's corner count honest for the
///      `kMinCorners` check.
///   3. faces are oriented outward from the origin. TempestRemap's overlap
///      requires it, and MOAB's own spherecube.h5m is the
///      other way round (signed area -4*pi*R^2), so this is not hypothetical.
///      Unlike the MOAB supermesh path, which reverses connectivity IN the
///      moab::Core, this reverses only the copy.
[[nodiscard]] Result<void> to_tempest(const PolygonSurface& s, double radius,
                                      ::Mesh& out) {
  const std::size_t nn = s.node_count();
  const std::size_t nf = s.face_count();
  out.nodes.reserve(nn);
  const double inv_r = 1.0 / radius;
  for (std::size_t v = 0; v < nn; ++v) {
    out.nodes.push_back(::Node(s.node_xyz[3 * v] * inv_r,
                             s.node_xyz[3 * v + 1] * inv_r,
                             s.node_xyz[3 * v + 2] * inv_r));
  }

  std::vector<int> corner;
  out.faces.reserve(nf);
  for (std::size_t f = 0; f < nf; ++f) {
    const std::int32_t begin = s.face_start[f];
    const std::int32_t end = s.face_start[f + 1];
    corner.clear();
    for (std::int32_t k = begin; k < end; ++k) {
      const std::int32_t v = s.face_node[static_cast<std::size_t>(k)];
      if (v < 0 || static_cast<std::size_t>(v) >= nn) {
        return Failure{"E-REMAP-GEOMETRY",
                     s.label + " face " + std::to_string(f) +
                         " refers to node " + std::to_string(v) + ", outside " +
                         "the " + std::to_string(nn) + " nodes it declares"};
      }
      if (!corner.empty() && corner.back() == v) continue;  // padded polygon
      corner.push_back(v);
    }
    while (corner.size() > 1 && corner.front() == corner.back()) {
      corner.pop_back();
    }
    if (corner.size() < static_cast<std::size_t>(kMinCorners)) {
      return Failure{"E-REMAP-GEOMETRY",
                   s.label + " face " + std::to_string(f) + " has only " +
                       std::to_string(corner.size()) +
                       " distinct corners; a surface polygon needs at least " +
                       std::to_string(kMinCorners)};
    }

    // Newell normal dotted with the (unnormalised) centroid direction. Robust
    // for a non-planar quad, which is exactly what a cubed-sphere cell is.
    double nx = 0.0, ny = 0.0, nz = 0.0, cx = 0.0, cy = 0.0, cz = 0.0;
    for (std::size_t k = 0; k < corner.size(); ++k) {
      const ::Node& a = out.nodes[static_cast<std::size_t>(corner[k])];
      const ::Node& b =
          out.nodes[static_cast<std::size_t>(corner[(k + 1) % corner.size()])];
      nx += a.y * b.z - a.z * b.y;
      ny += a.z * b.x - a.x * b.z;
      nz += a.x * b.y - a.y * b.x;
      cx += a.x;
      cy += a.y;
      cz += a.z;
    }
    if (nx * cx + ny * cy + nz * cz < 0.0) {
      std::reverse(corner.begin(), corner.end());
    }

    ::Face face(static_cast<int>(corner.size()));
    for (std::size_t k = 0; k < corner.size(); ++k) {
      face.SetNode(static_cast<int>(k), corner[k]);
    }
    out.faces.push_back(face);
  }

  // The three preprocessing steps TempestRemap's own file-loading path does
  // before handing a mesh to the overlap. Skipping ConstructEdgeMap costs an
  // "EdgeMap in meshSource must be constructed prior to GenerateOverlapFace"
  // exception -- which, per the header comment, is returned as success.
  out.RemoveZeroEdges();
  out.ConstructEdgeMap(/*fVerbose=*/false);
  out.ConstructReverseNodeArray();
  return Result<void>::success();
}

}  // namespace

bool high_order_backend_available() noexcept { return true; }

const char* high_order_backend_name() noexcept { return "TempestRemap"; }

Result<HighOrderResult> build_high_order_weights(const HighOrderRequest& request,
                                                 const PolygonSurface& src,
                                                 const PolygonSurface& tgt) {
  // --- what this backend can and cannot do -----------------------------------
  //
  // Stated in terms of the axis that is missing, not as "high order is
  // unavailable": high-order remap on a plane or in a volume is unimplemented
  // HERE, and a backend for it would slot in behind the same interface.
  if (request.kind != SurfaceKind::Sphere) {
    const char* what = (request.kind == SurfaceKind::Plane)
                           ? "a planar surface"
                           : "neither a sphere nor a plane (or is not a "
                             "dimension-2 surface at all)";
    return Failure{
        "E-REMAP-UNSUPPORTED",
        std::string("the TempestRemap backend remaps SPHERICAL SURFACES only: "
                    "its cell edges are great-circle arcs, its measures are "
                    "spherical excesses and its reconstruction stencils are "
                    "built in gnomonic coordinates. This pair is ") +
            what +
            ". High-order and monotone remap on a plane, or between volume "
            "meshes, is NOT IMPLEMENTED in this library -- it is not "
            "impossible, and no backend for it exists yet. The first-order "
            "conservative supermesh map (method \"intx\") does handle "
            "axis-aligned volumes and planar surfaces."};
  }
  if (!(request.radius > 0.0)) {
    return Failure{"E-REMAP-GEOMETRY",
                 "the TempestRemap backend was handed sphere radius " +
                     std::to_string(request.radius)};
  }
  if (request.order < 1 || request.order > kMaxOrder) {
    return Failure{
        "E-REMAP-UNSUPPORTED",
        "TempestRemap's finite-volume reconstruction is used here at orders 1 "
        "to " +
            std::to_string(kMaxOrder) + "; the plan asks for order " +
            std::to_string(request.order) +
            ". TempestRemap accepts higher orders but does not deliver them: "
            "its least-squares fit exceeds the built-in conditioning threshold "
            "(1e7) and it drops the offending cell to FIRST order with only a "
            "printed warning. Measured on a cubed-sphere pair, order 5 "
            "degraded on essentially every cell (condition estimates 1.5e7 to "
            "2.2e13), so it is refused rather than returned mislabelled."};
  }
  if (request.monotone && request.order > 1) {
    return Failure{
        "E-REMAP-UNSUPPORTED",
        "a conservative map that is BOTH bound-preserving and of order " +
            std::to_string(request.order) +
            " is not available. Bound preservation for every input field is "
            "equivalent to `all weights >= 0 and all row sums 1`, and the "
            "least-squares reconstruction that makes a map exact for "
            "polynomials puts negative entries in the stencil (measured "
            "minimum weight -0.049 at order 2, -0.075 at order 4). "
            "TempestRemap resolves the conflict by ignoring the order: "
            "GenerateOfflineMap.cpp calls LinearRemapFVtoFV with order 1 "
            "whenever fMonotone is set. Ask for order 1 with "
            "RemapFlags::Monotone to get the bound-preserving map, or drop "
            "Monotone to get the high-order one -- this library will not "
            "return the first under the second's name."};
  }
  if (src.face_count() == 0 || tgt.face_count() == 0) {
    return Failure{"E-REMAP-GEOMETRY",
                 "the TempestRemap backend was handed an empty surface (" +
                     std::to_string(src.face_count()) + " source faces, " +
                     std::to_string(tgt.face_count()) + " target faces)"};
  }
  if (src.face_count() >
          static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
      tgt.face_count() >
          static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
    return Failure{"E-REMAP-SIZE",
                 "the surfaces hold more faces than int32 COO indices can "
                 "address"};
  }

  ::Mesh mesh_src;
  ::Mesh mesh_tgt;
  {
    auto rs = to_tempest(src, request.radius, mesh_src);
    if (!rs.ok()) return rs.error();
    auto rt = to_tempest(tgt, request.radius, mesh_tgt);
    if (!rt.ok()) return rt.error();
  }

  HighOrderResult out;
  out.n_rows = tgt.face_count();
  out.n_cols = src.face_count();
  out.order = request.order;
  out.monotone = request.monotone;

  // Held from before `capture` is constructed to after it is destroyed
  // (declared first in this scope, so it is destroyed LAST on every path out
  // of this function -- normal return or early `return Failure{...}` alike,
  // per C++'s reverse-order-of-construction guarantee for locals). Every
  // `capture.text()` call below therefore happens while this lock is held.
  std::lock_guard<std::mutex> announce_lock(announce_mutex());
  AnnounceCapture capture;

  // --- the supermesh ---------------------------------------------------------
  ::Mesh mesh_ovl;
  {
    // The return value is checked, but it is not the real signal: see the file
    // header. `fAllowNoOverlap = false` so a genuinely disjoint pair raises
    // inside TempestRemap and lands in the captured text.
    const int rc = GenerateOverlapWithMeshes(
        mesh_src, mesh_tgt, mesh_ovl, /*strOverlapMesh=*/"",
        /*strOutputFormat=*/"Netcdf4", /*strMethod=*/"exact",
        /*fHasConcaveFacesA=*/false, /*fHasConcaveFacesB=*/false,
        /*fAllowNoOverlap=*/false, /*fVerbose=*/false);
    const std::string log = capture.text();
    if (rc != 0 || contains(log, "EXCEPTION") || mesh_ovl.faces.empty() ||
        mesh_ovl.vecSourceFaceIx.size() != mesh_ovl.faces.size() ||
        mesh_ovl.vecTargetFaceIx.size() != mesh_ovl.faces.size()) {
      return Failure{"E-REMAP-INTX",
                   "TempestRemap could not build the overlap mesh between " +
                       src.label + " and " + tgt.label + " (return code " +
                       std::to_string(rc) + ", " +
                       std::to_string(mesh_ovl.faces.size()) +
                       " overlap faces). It reports failure by printing and "
                       "returning success, so its own words follow:\n" +
                       tail(log, 12)};
    }
  }

  // --- coverage, BEFORE the map generation ------------------------------------
  //
  // `GenerateOfflineMapWithMeshes` rewrites `vecFaceArea` on both parents (it
  // corrects them to the overlap's areas when they agree to 1e-10), so the
  // coverage question -- how much of each parent the supermesh actually
  // accounted for -- has to be asked first, or the answer is 1 by construction.
  const double r2 = request.radius * request.radius;
  {
    mesh_src.CalculateFaceAreas(false);
    mesh_tgt.CalculateFaceAreas(false);
    mesh_ovl.CalculateFaceAreas(false);
    out.src_covered.assign(out.n_cols, 0.0);
    out.tgt_covered.assign(out.n_rows, 0.0);
    for (std::size_t k = 0; k < mesh_ovl.faces.size(); ++k) {
      const int i = mesh_ovl.vecSourceFaceIx[k];
      const int j = mesh_ovl.vecTargetFaceIx[k];
      if (i < 0 || static_cast<std::size_t>(i) >= out.n_cols || j < 0 ||
          static_cast<std::size_t>(j) >= out.n_rows) {
        return Failure{"E-REMAP-INTX",
                     "TempestRemap attributed overlap face " +
                         std::to_string(k) + " to parents (" +
                         std::to_string(i) + ", " + std::to_string(j) +
                         "), outside the " + std::to_string(out.n_rows) + "x" +
                         std::to_string(out.n_cols) + " matrix"};
      }
      out.src_covered[static_cast<std::size_t>(i)] += mesh_ovl.vecFaceArea[k] * r2;
      out.tgt_covered[static_cast<std::size_t>(j)] += mesh_ovl.vecFaceArea[k] * r2;
    }
  }

  // --- the weights -----------------------------------------------------------
  ::OfflineMap map;
  {
    ::GenerateOfflineMapAlgorithmOptions options;
    options.nPin = request.order;
    options.nPout = request.order;
    options.fMonotone = request.monotone;
    // fNoCheck: TempestRemap's own CheckMap only prints, and it prints into the
    // stream we are scanning for the word EXCEPTION. The audit that decides
    // anything is done by the caller, on the matrix, against MeshView measures.
    options.fNoCheck = true;

    const int rc = GenerateOfflineMapWithMeshes(mesh_src, mesh_tgt, mesh_ovl,
                                                "fv", "fv", options, map);
    out.log = capture.text();
    if (rc != 0 || contains(out.log, "EXCEPTION")) {
      return Failure{"E-REMAP-INTX",
                   "TempestRemap could not build the remap weights (return "
                   "code " +
                       std::to_string(rc) +
                       "). It reports failure by printing and returning "
                       "success, so its own words follow:\n" +
                       tail(out.log, 12)};
    }
    // The silent one. Not a warning to be relayed: the map that came back is a
    // first-order map wearing an order-N label, which is precisely the thing
    // this check exists to prevent.
    if (contains(out.log, "dropping to 1st order")) {
      return Failure{
          "E-REMAP-DEGRADED",
          "TempestRemap was asked for order " + std::to_string(request.order) +
              " and silently built a lower-order map: its least-squares fit "
              "exceeded the built-in conditioning threshold on at least one "
              "cell of this mesh pair, and it drops such cells to FIRST order "
              "with only a printed warning and a success return. The map is "
              "discarded rather than returned mislabelled. Try a lower order, "
              "or a less distorted mesh. TempestRemap's own words:\n" +
              tail(out.log, 6)};
    }
  }

  const ::SparseMatrix<double>& smat = map.GetSparseMatrix();
  if (smat.GetRows() != static_cast<int>(out.n_rows) ||
      smat.GetColumns() != static_cast<int>(out.n_cols)) {
    return Failure{"E-REMAP-INTX",
                 "TempestRemap returned a " + std::to_string(smat.GetRows()) +
                     "x" + std::to_string(smat.GetColumns()) +
                     " matrix for a " + std::to_string(out.n_rows) + "x" +
                     std::to_string(out.n_cols) + " mesh pair"};
  }

  ::DataArray1D<int> rows;
  ::DataArray1D<int> cols;
  ::DataArray1D<double> vals;
  smat.GetEntries(rows, cols, vals);
  const std::size_t nnz = vals.GetRows();
  if (nnz == 0) {
    return Failure{"E-REMAP-INTX",
                 "TempestRemap returned an all-zero weight matrix for " +
                     std::to_string(mesh_ovl.faces.size()) +
                     " overlap faces"};
  }
  out.row.reserve(nnz);
  out.col.reserve(nnz);
  out.val.reserve(nnz);
  for (std::size_t k = 0; k < nnz; ++k) {
    if (rows[k] < 0 || static_cast<std::size_t>(rows[k]) >= out.n_rows ||
        cols[k] < 0 || static_cast<std::size_t>(cols[k]) >= out.n_cols) {
      return Failure{"E-REMAP-INTX",
                   "TempestRemap emitted triplet (" + std::to_string(rows[k]) +
                       ", " + std::to_string(cols[k]) + "), outside the " +
                       std::to_string(out.n_rows) + "x" +
                       std::to_string(out.n_cols) + " matrix"};
    }
    out.row.push_back(static_cast<std::int32_t>(rows[k]));
    out.col.push_back(static_cast<std::int32_t>(cols[k]));
    out.val.push_back(vals[k]);
  }

  // The areas the map is conservative in: TempestRemap's own, corrected to the
  // overlap's during generation, scaled back out of the unit sphere.
  const ::DataArray1D<double>& sa = map.GetSourceAreas();
  const ::DataArray1D<double>& ta = map.GetTargetAreas();
  if (sa.GetRows() != out.n_cols || ta.GetRows() != out.n_rows) {
    return Failure{"E-REMAP-INTX",
                 "TempestRemap returned " + std::to_string(sa.GetRows()) +
                     " source and " + std::to_string(ta.GetRows()) +
                     " target areas for a " + std::to_string(out.n_cols) +
                     " by " + std::to_string(out.n_rows) + " mesh pair"};
  }
  out.src_area.resize(out.n_cols);
  out.tgt_area.resize(out.n_rows);
  for (std::size_t i = 0; i < out.n_cols; ++i) out.src_area[i] = sa[i] * r2;
  for (std::size_t j = 0; j < out.n_rows; ++j) out.tgt_area[j] = ta[j] * r2;

  return out;
}

#endif  // FUSE_HAVE_TEMPEST

}  // namespace fuse::remap_backend
