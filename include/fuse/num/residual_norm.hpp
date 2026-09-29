#pragma once

/// Residual norms for a coupled iteration, including one that has units.
///
/// WHAT IS WRONG WITH THE USUAL NORM. A coupled iteration is declared
/// converged when the residual over the exchanged interface values is small.
/// Both parents of this library, and every coupler we looked at, computed that
/// as a plain Euclidean norm over the array of values. That quantity depends
/// on things that have nothing to do with the physics:
///
///   * Mesh resolution. Refine the interface and the same physical error gives
///     a larger norm, because there are more entries to square and add. A
///     tolerance tuned on a coarse mesh is a different tolerance on a fine one,
///     and nobody is told.
///   * Cell size variation. A graded mesh weights its small cells exactly as
///     heavily as its large ones, so a tiny sliver cell can hold up an
///     iteration that has converged everywhere that matters.
///   * Units. A norm over a temperature field and a norm over a heat flux
///     field are added together as though they were commensurable.
///   * Halo duplication. Sum over all local entries and a value owned by one
///     rank but mirrored on two others is counted three times, so the norm
///     depends on the partition.
///
/// WHAT THIS COMPUTES INSTEAD. The measure-weighted root-mean-square: each
/// entry's square is weighted by the measure (length, area or volume) of the
/// entity it lives on, summed over OWNED entities only, and divided by the
/// total measure. The result has the units of the field and is, to leading
/// order, independent of resolution, of grading, and of how the interface was
/// partitioned -- so a tolerance means the same thing across all three.
///
/// It is not a drop-in replacement for the plain norm and does not try to be:
/// the recorded iteration counts of the inherited test suite were produced
/// with the plain norm, and changing it would change them. Both are available
/// and which one an implicit coupling group uses is declared.

#include <cstddef>
#include <vector>

#include "fuse/base/result.hpp"
#include "fuse/base/units.hpp"

namespace fuse::num {

/// How a residual over one field's entries becomes one number.
enum class NormKind {
  /// sqrt(sum r_i^2) over every supplied entry. What the inherited iteration
  /// counts were recorded with; dimensionless by convention, which is part of
  /// the problem it has.
  Euclidean,
  /// sqrt(sum w_i r_i^2 / sum w_i) over owned entries, with `w_i` the entity
  /// measure. Carries the field's units.
  MeasureWeightedRms,
};

[[nodiscard]] constexpr const char* to_cstring(NormKind k) noexcept {
  switch (k) {
    case NormKind::Euclidean:          return "euclidean";
    case NormKind::MeasureWeightedRms: return "measure-weighted-rms";
  }
  return "unknown";
}

/// One field's contribution: its residual entries, the measure of the entity
/// each lives on, and which entries this rank owns.
///
/// `owned` is a mask rather than a count because ownership is not contiguous
/// in a general partition. An empty mask means every entry is owned, which is
/// the serial case and the common one.
struct ResidualField {
  const double*            residual = nullptr;
  const double*            measure  = nullptr;   ///< nullptr = every weight 1
  const bool*              owned    = nullptr;   ///< nullptr = all owned
  std::size_t              count    = 0;
  Units                    units{};
};

/// The weighted norm of one field, with its units.
struct NormValue {
  double value = 0.0;
  Units  units{};
  /// Total measure the norm was divided by, and the number of owned entries
  /// it covered. Reported because a norm over zero owned entries is not zero,
  /// it is undefined, and a caller has to be able to tell the difference.
  double total_measure = 0.0;
  std::size_t owned_count = 0;
};

/// Computes `kind` over `f`.
///
/// A field with no owned entries on this rank returns a zero norm with
/// `owned_count == 0`; combining that across ranks is the caller's job, and
/// `combine` below is how.
[[nodiscard]] Result<NormValue> field_norm(NormKind kind,
                                           const ResidualField& f);

/// Combines per-field norms into the single number a convergence measure is
/// tested against.
///
/// The fields must agree on units, and disagreement is an error rather than a
/// silent sum: adding a temperature norm to a flux norm produces a number with
/// no meaning, and a tolerance on it cannot be reasoned about. A caller that
/// genuinely wants one number over unlike fields must scale them itself and
/// say so.
[[nodiscard]] Result<NormValue> combine(NormKind kind,
                                        const std::vector<NormValue>& parts);

}  // namespace fuse::num
