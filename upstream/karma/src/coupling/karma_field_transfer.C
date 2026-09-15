#include "karma_field_transfer.h"

#include "karma_config.h"
#include "karma_exceptions.h"

#include <cmath>
#include <utility>

namespace KARMA
{
KARMAFieldTransfer::KARMAFieldTransfer(TransferFamily family, TransferScheme scheme, TransferOptions options)
    : _family(family), _scheme(scheme), _options(std::move(options))
{
    validate_options();
    if (_diagnostic.empty())
    {
        switch (_scheme)
        {
        case TransferScheme::ElementConstant:
        case TransferScheme::FiniteElement:
        case TransferScheme::Bilinear:
        case TransferScheme::Trilinear:
            _capabilities = {true, true, true, true, false, {}};
            break;
        case TransferScheme::L2MatchingMesh:
            _capabilities = {false, false, false, false, false, "matching L2 backend is not implemented"};
            break;
        case TransferScheme::L2Intersection:
            _capabilities = {false, false, true, false, false, "validated intersection provider is unavailable"};
            break;
        case TransferScheme::ConservativeSupermesh:
            _capabilities = {false, false, false, false, false,
                              "conservative supermesh backend requires a validated volumetric remapper"};
            break;
        default:
            _capabilities = {false, false, false, false, false, "requested quasi-interpolant is not implemented"};
            break;
        }
    }
}

void KARMAFieldTransfer::validate_options()
{
    if (const auto* finite_element = std::get_if<FiniteElementTransferOptions>(&_options))
    {
        if (finite_element->order == 0 || finite_element->order > 2) set_unavailable("finite-element order must be 1 or 2");
    }
    else if (const auto* shepard = std::get_if<ShepardTransferOptions>(&_options))
    {
        if (!std::isfinite(shepard->exponent) || shepard->exponent <= 0.0)
            set_unavailable("Shepard exponent must be finite and positive");
    }
    else if (const auto* radial_basis = std::get_if<RadialBasisTransferOptions>(&_options))
    {
        if (!std::isfinite(radial_basis->shape) || radial_basis->shape <= 0.0)
            set_unavailable("RBF shape must be finite and positive");
    }
    else if (const auto* moving_least_squares = std::get_if<MovingLeastSquaresOptions>(&_options))
    {
        if (moving_least_squares->degree == 0) set_unavailable("MLS degree must be positive");
    }
}

void KARMAFieldTransfer::set_unavailable(std::string reason)
{
    _diagnostic = std::move(reason);
    _capabilities.available = false;
    _capabilities.unavailable_reason = _diagnostic;
}

void KARMAFieldTransfer::setup(const KARMAField& source, const KARMAField& target)
{
    const auto previous_state = _state;
    const auto previous_diagnostic = _diagnostic;
    try
    {
        source.validate_values();
        target.validate_values();
        if (!_diagnostic.empty() || !_capabilities.available)
            KARMA_THROW(InvalidArgument(_diagnostic.empty() ? _capabilities.unavailable_reason : _diagnostic));
        if (source.components() != target.components())
            KARMA_THROW(InvalidArgument("source and target component counts differ"));
        _state = TransferState::Ready;
    }
    catch (...)
    {
        _state = previous_state;
        _diagnostic = previous_diagnostic;
        throw;
    }
}

void KARMAFieldTransfer::apply() const
{
    if (_state != TransferState::Ready) KARMA_THROW(InvalidArgument("transfer has no valid setup"));
}
}  // namespace KARMA
