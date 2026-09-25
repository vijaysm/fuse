#pragma once

#include <algorithm>
#include <cmath>

#include "fuse/search/types.hpp"

/// Pure geometry on `fuse::search::Box`/`Point` -- no Kokkos, no ArborX, no
/// Result<>. `BoxIndex` (box_index.hpp) and `element_boxes`
/// (mesh_adapter.hpp) both build on exactly these functions, and so does
/// every test that checks either against a brute-force reference: comparing
/// against a SEPARATELY invented notion of "padded" or "contains" would not
/// be testing this library's actual tolerance rule.
namespace fuse::search {

/// Euclidean length of `b`'s diagonal. Degenerate axes (an embedded-lower-
/// dimension box's unused coordinates, always exactly zero per `Point`'s
/// convention) contribute zero, so this is the right reference length
/// regardless of whether `b` is genuinely 1D, 2D or 3D.
[[nodiscard]] inline double diagonal(const Box& b) noexcept {
  const double dx = b.hi.x - b.lo.x;
  const double dy = b.hi.y - b.lo.y;
  const double dz = b.hi.z - b.lo.z;
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

/// The dimensionality `b` ITSELF exhibits: 3 if it has nonzero extent on z,
/// else 2 if nonzero on y, else 1. Exact equality, not a tolerance
/// comparison -- an unused axis is meant to be EXACTLY zero (Point's
/// convention; mesh_view.hpp's Point doc says the same), so a box that is
/// merely very thin but not exactly flat correctly classifies as the higher
/// dimension rather than being silently rounded down.
[[nodiscard]] inline int classify_dim(const Box& b) noexcept {
  if (b.hi.z != b.lo.z) return 3;
  if (b.hi.y != b.lo.y) return 2;
  return 1;
}

/// `b` grown by `max(tol.relative * diagonal(b), tol.absolute_floor)` on
/// every side of every ACTIVE axis -- one with `hi != lo` -- and left
/// EXACTLY unchanged on a degenerate one. The exact-unchanged half is not an
/// optimisation: `classify_dim` (and `BoxIndex::build`'s per-box consistency
/// check built on it) requires an unused axis to stay EXACTLY zero
/// (Point's own convention), and growing it symmetrically by an amount that
/// is not exactly representable would turn `lo.z == hi.z` into
/// `lo.z == -p, hi.z == +p` -- two DIFFERENT floating point values -- and
/// silently reclassify a 1D or 2D box as one dimension higher the next time
/// anything calls `classify_dim` on the padded result (caught by
/// mesh_adapter_test.cpp's `GeneratedLineYieldsOneBoxPerCellAtDimension1`,
/// which failed with `dimension() == 3` for a 1D line mesh before this rule
/// existed). This is the ONE place the "shared face finds both cells"
/// tolerance is applied; a caller that wants that property calls this
/// before `BoxIndex::build` (`BoxIndex` itself is tolerance-agnostic -- it
/// indexes whatever boxes it is given). `mesh_adapter.hpp`'s
/// `element_boxes` calls it internally.
[[nodiscard]] inline Box padded(const Box& b, const Tolerance& tol) noexcept {
  const double p = std::max(tol.relative * diagonal(b), tol.absolute_floor);
  Box out = b;
  if (b.hi.x != b.lo.x) {
    out.lo.x -= p;
    out.hi.x += p;
  }
  if (b.hi.y != b.lo.y) {
    out.lo.y -= p;
    out.hi.y += p;
  }
  if (b.hi.z != b.lo.z) {
    out.lo.z -= p;
    out.hi.z += p;
  }
  return out;
}

/// Exact (no tolerance) containment test: every coordinate of `p` within
/// `[lo, hi]`. Called against an ALREADY-`padded()` box when the tolerant
/// version is wanted -- kept separate rather than folded together so a
/// caller can also use it for an exact post-filter.
[[nodiscard]] inline bool contains(const Box& b, const Point& p) noexcept {
  return p.x >= b.lo.x && p.x <= b.hi.x && p.y >= b.lo.y && p.y <= b.hi.y && p.z >= b.lo.z &&
         p.z <= b.hi.z;
}

}  // namespace fuse::search
