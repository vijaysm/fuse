#pragma once

#include "fuse/base/ids.hpp"

namespace fuse {

/// KNOWN_ISSUES D82 (MOAB half, round 11): how one intersection measure
/// becomes one remap weight entry W(target_cell, source_cell), given the
/// channel's own `Conservation` kind (`RemapPlan::conservation`).
///
/// `overlap` is the measure of the (source, target) cell pair's intersection,
/// in whatever quadrature the supermesh/intersection itself used -- for the
/// MOAB sphere path, `spherical_polygon_area` (exact great-circle excess).
/// `src_parent_measure` is the SAME source cell's own measure in that
/// identical quadrature (so `overlap / src_parent_measure` is an exact
/// geometric fraction of the source cell, independent of any other measure
/// convention). `src_measure`/`tgt_measure` are the source/target cell's
/// measure in the backend's own reporting convention (`MeshView::measure`,
/// `RemapWeights::src_measure_total`'s own measure) -- for a MOAB spherical
/// pair this is the bilinear-patch (chord) area, NOT the spherical one, and
/// the two differ by O(h^2) (`remap_plan.hpp`'s own "WHICH SUM IS WHICH"
/// header measured 9.88e-4 on one mesh pair; round 11 measured 4.4e-3 mean,
/// 5.1e-3 max on wholeATM_T.h5m's 726 cells directly, and 1.8e-4 max on
/// recMeshOcn.h5m's finer 7153 cells -- see KNOWN_ISSUES D82).
///
///   * `Conservation::Extensive`: `w = overlap / src_measure`. The field
///     value is an AMOUNT, not a density, so no density-to-density
///     conversion is applied at all -- source cell i's column sums to
///     `src_parent_measure_i / src_measure_i`, which is exactly 1 wherever
///     the two measure conventions coincide (planar or matched-measure
///     pairs), and off by the O(h^2) discrepancy above on a spherical pair
///     where they do not. This is what closes D82's reported symptom: the
///     PRE-EXISTING (Intensive) formula's density-conversion factor
///     `src_measure / tgt_measure` inflates or deflates an amount-valued
///     field whenever a source cell is split across target cells of a
///     different measure, independent of any sphere-specific discrepancy
///     (a matched-measure planar example with one source cell split evenly
///     across two half-measure target cells shows the Intensive formula
///     doubling the delivered total; round 11 read this algebraically
///     before writing the fix, not just the sphere-specific residual).
///   * Everything else (`Conservation::Intensive`, `Conservation::None`):
///     the pre-existing form, UNCHANGED -- `(overlap / src_parent_measure) *
///     (src_measure / tgt_measure)`. Target cell j's row sums to 1 under
///     full target coverage (constant preservation), to the same O(h^2)
///     residual, reported in `RemapWeights::max_row_defect`.
[[nodiscard]] inline double normalize_remap_weight(double overlap,
                                                    double src_parent_measure,
                                                    double src_measure,
                                                    double tgt_measure,
                                                    Conservation kind) {
  if (kind == Conservation::Extensive) {
    return overlap / src_measure;
  }
  return (overlap / src_parent_measure) * (src_measure / tgt_measure);
}

}  // namespace fuse
