#include "gray_scott_model.h"

#include "karma_exceptions.h"

#include <cmath>

namespace KARMA
{
GrayScottModel::GrayScottModel( GrayScottParameters parameters ) : _parameters( parameters )
{
    if( !std::isfinite( _parameters.diffusion_u ) || _parameters.diffusion_u < 0 ||
        !std::isfinite( _parameters.diffusion_v ) || _parameters.diffusion_v < 0 )
        KARMA_THROW( InvalidArgument( "Gray-Scott diffusion coefficients must be finite and nonnegative" ) );
    if( !std::isfinite( _parameters.feed_rate ) || _parameters.feed_rate < 0 ||
        !std::isfinite( _parameters.kill_rate ) || _parameters.kill_rate < 0 )
        KARMA_THROW( InvalidArgument( "Gray-Scott feed and kill rates must be finite and nonnegative" ) );
}

Real GrayScottModel::diffusion( GrayScottSpecies species ) const noexcept
{
    return species == GrayScottSpecies::U ? _parameters.diffusion_u : _parameters.diffusion_v;
}

GrayScottReaction GrayScottModel::reaction( Real u, Real v ) const noexcept
{
    const Real uvv = u * v * v;
    return { -uvv + _parameters.feed_rate * ( 1 - u ),
             uvv - ( _parameters.feed_rate + _parameters.kill_rate ) * v };
}

GrayScottModel::Jacobian GrayScottModel::reaction_jacobian( Real u, Real v ) const noexcept
{
    return { { { -v * v - _parameters.feed_rate, -2 * u * v },
               { v * v, 2 * u * v - _parameters.feed_rate - _parameters.kill_rate } } };
}
}  // namespace KARMA
