#include "diffusion_reaction_system_property_manager.h"

#include "karma_exceptions.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace KARMA
{
void DiffusionReactionSystemPropertyManager::register_model( int material_id, DiffusionReactionCoefficients model )
{
    if( material_id < 0 ) KARMA_THROW( InvalidArgument( "material ID must be nonnegative" ) );
    if( !model.diffusion || !model.reaction || !model.source )
        KARMA_THROW( InvalidArgument( "diffusion-reaction model requires diffusion, reaction, and source callbacks" ) );
    if( !_models.emplace( material_id, std::move( model ) ).second )
        KARMA_THROW( InvalidArgument( "material " + std::to_string( material_id ) + " is already registered" ) );
}

bool DiffusionReactionSystemPropertyManager::has_model( int material_id ) const noexcept
{
    return _models.contains( material_id );
}

const DiffusionReactionCoefficients& DiffusionReactionSystemPropertyManager::model( int material_id ) const
{
    const auto found = _models.find( material_id );
    if( found == _models.end() )
        KARMA_THROW( InvalidArgument( "no diffusion-reaction model registered for material " +
                                      std::to_string( material_id ) ) );
    return found->second;
}

std::vector< int > DiffusionReactionSystemPropertyManager::material_ids() const
{
    std::vector< int > result;
    result.reserve( _models.size() );
    for( const auto& [material, model] : _models )
    {
        (void)model;
        result.push_back( material );
    }
    return result;
}

DiffusionReactionCoefficients DiffusionReactionSystemPropertyManager::coefficients() const
{
    DiffusionReactionCoefficients result;
    result.diffusion = [this]( const Point& point, Real field, int material ) {
        return model( material ).diffusion( point, field, material );
    };
    result.reaction = [this]( const Point& point, Real field, int material ) {
        return model( material ).reaction( point, field, material );
    };
    result.source = [this]( const Point& point, Real field, int material ) {
        return model( material ).source( point, field, material );
    };
    const auto derivative = [this]( auto member, auto coefficient ) {
        return [this, member, coefficient]( const Point& point, Real field, int material ) {
            const auto& selected = model( material );
            const auto& callback = selected.*member;
            if( callback ) return callback( point, field, material );
            const Real step = std::cbrt( std::numeric_limits< Real >::epsilon() ) *
                              std::max( Real{ 1 }, std::abs( field ) );
            const auto& value = selected.*coefficient;
            return ( value( point, field + step, material ) - value( point, field - step, material ) ) / ( 2 * step );
        };
    };
    result.diffusion_derivative = derivative( &DiffusionReactionCoefficients::diffusion_derivative,
                                              &DiffusionReactionCoefficients::diffusion );
    result.reaction_derivative = derivative( &DiffusionReactionCoefficients::reaction_derivative,
                                             &DiffusionReactionCoefficients::reaction );
    result.source_derivative = derivative( &DiffusionReactionCoefficients::source_derivative,
                                           &DiffusionReactionCoefficients::source );
    return result;
}
}  // namespace KARMA
