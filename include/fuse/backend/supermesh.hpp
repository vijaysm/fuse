#pragma once

/// Lowest-order conservative remap weights from an exact simplex supermesh.
///
/// This is the path that is ALWAYS available: it needs no MOAB, no
/// TempestRemap and no spherical geometry, only the simplex decomposition
/// every `MeshView` provides. Because it is always present it is also the
/// differential oracle for the optional paths -- on a pair all of them can
/// express, they are required to agree.
///
/// WHAT IT COMPUTES. For every target cell `j` and source cell `i` it computes
/// the exact measure of their intersection, `|T_j intersect S_i|`, and
/// normalises by ONE of two conventions, chosen by the `Conservation`
/// argument (KNOWN_ISSUES.md D82, owner decision 2026-10-19):
///
///     Intensive: W_ji = |T_j intersect S_i| / m_j   (m_j = target cell j's own measure)
///     Extensive: W_ji = |T_j intersect S_i| / n_i   (n_i = source cell i's own measure)
///
/// `Intensive` (the field's value is a DENSITY, the conserved amount is
/// `value * measure`) makes `sum_i W_ji = 1` (every target row), so
/// `g_j = sum_i W_ji f_i` is the measure-weighted average of the source field
/// over the target cell -- constant-preserving and conservative in the
/// integral sense (`sum_j g_j m_j = sum_i f_i n_i`) together, which is why
/// there was no separate "conservative" variant before this field existed: a
/// first-order supermesh map is conservative under THIS convention by
/// construction. `Extensive` (the field's value directly IS the conserved
/// amount) instead makes `sum_j W_ji = 1` (every source COLUMN), so the whole
/// amount `f_i` in source cell `i` is redistributed across the target cells
/// it overlaps (`sum_i f_i = sum_j g_j`) rather than reinterpreted as a
/// density -- the two conventions agree only when every overlapping pair
/// shares the same measure. `RemapWeights`'s own audit fields (`min_src_
/// coverage`, `max_row_defect`, ...) report how nearly the CHOSEN convention
/// achieved its own guarantee on the pair it was given, not the other one's.
///
/// EXACTNESS. The intersection measures are exact up to floating-point
/// rounding, not quadrature: the geometric kernel clips polytopes and computes
/// analytic moments (r3d, Powell & Abel 2015). One dimension does not use r3d
/// at all -- an interval overlap is a subtraction, and calling a clipper for it
/// would add error rather than remove it.
///
/// WHAT IT IS NOT. Order 1. The reconstruction is piecewise constant per
/// source cell, so the map is exact for constants and first order for anything
/// else. That is what the ruling asked for as the floor; higher order on the
/// sphere is what the TempestRemap tier adds.

#include "fuse/backend/mesh_view.hpp"
#include "fuse/backend/remap_plan.hpp"
#include "fuse/base/result.hpp"

namespace fuse {

/// Builds the weights for the pair, at the given entity dimension.
///
/// Rejects, rather than returning an empty or partial map, when either view
/// reports no geometry (`simplex_count() == 0`), when the dimension is outside
/// 1..3, or when a cell's simplices do not cover it. A silently empty
/// supermesh would look like a legitimate zero-overlap answer.
[[nodiscard]] Result<RemapWeights> build_supermesh_weights(const MeshView& src,
                                                           const MeshView& tgt,
                                                           int dim,
                                                           Conservation conservation = Conservation::None);

}  // namespace fuse
