#ifndef KARMA_COUPLING_TRANSFER_MAP_H
#define KARMA_COUPLING_TRANSFER_MAP_H

#include "karma_field.h"

#include <cstddef>
#include <map>
#include <span>
#include <vector>

namespace KARMA
{
enum class TransferMapRepresentation
{
    SparseInterpolation,
    ProjectionFactors,
    ExplicitProjection
};

enum class CoverageClass
{
    Full,
    Partial,
    Uncovered
};

struct SparseInterpolationRow
{
    FieldEntryKey target;
    std::vector< FieldEntryKey > source;
    std::vector< Real > weights;
    CoverageClass coverage = CoverageClass::Uncovered;
};

class KARMATransferMap
{
  public:
    KARMATransferMap( TransferMapRepresentation representation, unsigned components );

    [[nodiscard]] TransferMapRepresentation representation() const noexcept { return _representation; }
    [[nodiscard]] unsigned components() const noexcept { return _components; }
    [[nodiscard]] const std::vector< SparseInterpolationRow >& rows() const noexcept { return _rows; }

    void add_row( SparseInterpolationRow row );
    void verify() const;
    [[nodiscard]] KARMATransferMap select_targets(std::span<const FieldEntryKey> owned_targets) const;
    void validate_distribution(std::span<const FieldEntryKey> target_owned,
                               std::span<const FieldEntryKey> source_local) const;
    [[nodiscard]] std::map<FieldEntryKey, Real> apply_b(
        const std::map<FieldEntryKey, Real>& source_values,
        std::span<const FieldEntryKey> target_owned,
        std::span<const FieldEntryKey> source_local) const;

  private:
    TransferMapRepresentation _representation;
    unsigned _components;
    std::vector< SparseInterpolationRow > _rows;
};
}  // namespace KARMA

#endif
