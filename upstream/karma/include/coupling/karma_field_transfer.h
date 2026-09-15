#ifndef KARMA_COUPLING_FIELD_TRANSFER_H
#define KARMA_COUPLING_FIELD_TRANSFER_H

#include "karma_field.h"

#include <string>
#include <variant>
#include <vector>

namespace KARMA
{
enum class TransferFamily
{
    PointwiseInterpolation,
    QuasiInterpolation,
    VariationalProjection,
    ConservativeRemap
};

enum class TransferScheme
{
    ElementConstant,
    FiniteElement,
    Bilinear,
    Trilinear,
    NaturalNeighbor,
    Shepard,
    RadialBasisFunction,
    MovingLeastSquares,
    HierarchicalBSpline,
    L2MatchingMesh,
    L2Intersection,
    ConservativeSupermesh
};

enum class TransferState
{
    Uninitialized,
    Ready,
    Failed
};

struct FiniteElementTransferOptions
{
    unsigned order = 1;
};
struct ShepardTransferOptions
{
    double exponent = 2.0;
    unsigned neighbors = 0;
};
struct RadialBasisTransferOptions
{
    double shape = 1.0;
};
struct MovingLeastSquaresOptions
{
    unsigned degree = 1;
};
struct MatchingL2Options
{
    double coordinate_tolerance = 1.0e-12;
};
struct IntersectionL2Options
{
    bool project_covered_region = false;
};
struct ConservativeRemapOptions
{
    bool require_local_conservation = true;
};
using TransferOptions = std::variant< FiniteElementTransferOptions, ShepardTransferOptions,
                                      RadialBasisTransferOptions, MovingLeastSquaresOptions,
                                      MatchingL2Options, IntersectionL2Options, ConservativeRemapOptions >;

struct TransferCapabilities
{
    bool available = false;
    bool supports_parallel = false;
    bool supports_coverage = false;
    bool can_build_map = false;
    bool guarantees_conservation = false;
    std::string unavailable_reason;
};

class KARMAFieldTransfer
{
  public:
    KARMAFieldTransfer( TransferFamily family, TransferScheme scheme, TransferOptions options );

    [[nodiscard]] TransferFamily family() const noexcept { return _family; }
    [[nodiscard]] TransferScheme scheme() const noexcept { return _scheme; }
    [[nodiscard]] const TransferCapabilities& capabilities() const noexcept { return _capabilities; }
    [[nodiscard]] TransferState state() const noexcept { return _state; }
    [[nodiscard]] const std::string& diagnostic() const noexcept { return _diagnostic; }

    void setup( const KARMAField& source, const KARMAField& target );
    void apply() const;

    [[nodiscard]] static TransferCapabilities fem_interpolation_capabilities() noexcept
    { return {true, true, true, true, false, {}}; }
    [[nodiscard]] static TransferCapabilities local_projection_capabilities() noexcept
    { return {true, true, true, false, false, {}}; }
    [[nodiscard]] static TransferCapabilities global_cg_projection_capabilities() noexcept
    { return {true, true, true, false, false, {}}; }
    [[nodiscard]] static TransferCapabilities projection_map_persistence_capabilities() noexcept
    { return {false, false, false, false, false, "projection-factor persistence is not implemented"}; }

  private:
    void validate_options();
    void set_unavailable( std::string reason );

    TransferFamily _family;
    TransferScheme _scheme;
    TransferOptions _options;
    TransferCapabilities _capabilities;
    TransferState _state = TransferState::Uninitialized;
    std::string _diagnostic;
};
}  // namespace KARMA

#endif
