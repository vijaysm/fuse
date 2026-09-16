#include "fuse/num/residual_norm.hpp"

#include <cmath>
#include <cstddef>

#include "fuse/base/error.hpp"

namespace fuse::num {

Result<NormValue> field_norm(NormKind kind, const ResidualField& f) {
  if (f.count > 0 && f.residual == nullptr) {
    return Failure{"E-NORM-NULL",
                   "field_norm: a field with entries has no residual array"};
  }

  NormValue out;
  out.units = kind == NormKind::Euclidean ? Units{} : f.units;

  double sum = 0.0;
  double wsum = 0.0;
  std::size_t owned = 0;
  for (std::size_t i = 0; i < f.count; ++i) {
    // Ownership first: an unowned entry contributes to neither the numerator
    // nor the denominator. Counting it in one but not the other is how a norm
    // ends up depending on the partition.
    if (kind == NormKind::MeasureWeightedRms && f.owned != nullptr && !f.owned[i]) {
      continue;
    }
    ++owned;
    const double r = f.residual[i];
    if (!std::isfinite(r)) {
      return Failure{"E-NORM-NONFINITE",
                     "field_norm: residual entry " + std::to_string(i) +
                         " is not finite; the iteration has already diverged "
                         "and a norm over it would hide that"};
    }
    if (kind == NormKind::Euclidean) {
      sum += r * r;
      continue;
    }
    const double w = f.measure != nullptr ? f.measure[i] : 1.0;
    if (!(w > 0.0) || !std::isfinite(w)) {
      return Failure{"E-NORM-MEASURE",
                     "field_norm: entity " + std::to_string(i) +
                         " has measure " + std::to_string(w) +
                         "; a non-positive or non-finite cell measure is a mesh "
                         "defect, not something to weight by"};
    }
    sum += w * r * r;
    wsum += w;
  }

  out.owned_count = owned;
  out.total_measure = kind == NormKind::Euclidean ? static_cast<double>(owned) : wsum;

  if (kind == NormKind::Euclidean) {
    out.value = std::sqrt(sum);
    return out;
  }
  if (owned == 0) {
    // Not zero: undefined. Zero would read as "converged" on a rank that owns
    // no part of this interface, which is the wrong answer to publish into a
    // reduction. The caller sees owned_count == 0 and excludes it.
    out.value = 0.0;
    return out;
  }
  out.value = std::sqrt(sum / wsum);
  return out;
}

Result<NormValue> combine(NormKind kind, const std::vector<NormValue>& parts) {
  NormValue out;
  if (parts.empty()) return out;

  double sum = 0.0;
  double wsum = 0.0;
  std::size_t owned = 0;
  bool have_units = false;
  for (const NormValue& p : parts) {
    if (p.owned_count == 0) continue;   // undefined here, not zero
    if (kind == NormKind::MeasureWeightedRms) {
      if (!have_units) {
        out.units = p.units;
        have_units = true;
      } else if (!out.units.same_dimensions(p.units)) {
        return Failure{"E-NORM-UNITS",
                       "combine: cannot add a norm in " + out.units.to_string() +
                           " to one in " + p.units.to_string() +
                           "; scale them to a common unit and say so, rather "
                           "than summing unlike quantities into one tolerance"};
      }
      // Un-average, add, re-average: the combined norm is the weighted RMS
      // over the union of the entities, which is NOT the RMS of the per-field
      // RMS values unless every field carries the same total measure.
      sum += p.value * p.value * p.total_measure;
      wsum += p.total_measure;
    } else {
      sum += p.value * p.value;
      wsum += p.total_measure;
    }
    owned += p.owned_count;
  }

  out.owned_count = owned;
  out.total_measure = wsum;
  if (owned == 0) return out;
  out.value = kind == NormKind::Euclidean ? std::sqrt(sum)
                                          : std::sqrt(sum / wsum);
  return out;
}

}  // namespace fuse::num
