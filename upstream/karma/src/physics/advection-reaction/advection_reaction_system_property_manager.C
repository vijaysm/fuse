#include "advection_reaction_system_property_manager.h"

#include "karma_exceptions.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>

namespace KARMA
{
void AdvectionReactionSystemPropertyManager::register_model( int material_id, AdvectionReactionCoefficients model )
{
    if( material_id < 0 ) KARMA_THROW( InvalidArgument( "material ID must be nonnegative" ) );
    if( !model.velocity || !model.reaction || !model.source || !model.inflow )
        KARMA_THROW( InvalidArgument( "advection-reaction model requires velocity, reaction, source, and inflow callbacks" ) );
    if( !_models.emplace( material_id, std::move( model ) ).second )
        KARMA_THROW( InvalidArgument( "material " + std::to_string( material_id ) + " is already registered" ) );
}

bool AdvectionReactionSystemPropertyManager::has_model( int material_id ) const noexcept
{
    return _models.contains( material_id );
}

const AdvectionReactionCoefficients& AdvectionReactionSystemPropertyManager::model( int material_id ) const
{
    const auto found = _models.find( material_id );
    if( found == _models.end() )
        KARMA_THROW( InvalidArgument( "no advection-reaction model registered for material " +
                                      std::to_string( material_id ) ) );
    return found->second;
}

std::vector< int > AdvectionReactionSystemPropertyManager::material_ids() const
{
    std::vector< int > ids;
    ids.reserve( _models.size() );
    for( const auto& [id, model] : _models ) { (void)model; ids.push_back( id ); }
    return ids;
}

AdvectionReactionCoefficients AdvectionReactionSystemPropertyManager::coefficients() const
{
    AdvectionReactionCoefficients result;
    result.velocity = [this]( const Point& point, int material ) { return model( material ).velocity( point, material ); };
    result.reaction = [this]( const Point& point, Real field, int material ) {
        return model( material ).reaction( point, field, material );
    };
    result.source = [this]( const Point& point, Real field, int material ) {
        return model( material ).source( point, field, material );
    };
    result.inflow = [this]( const Point& point, int material ) { return model( material ).inflow( point, material ); };
    const auto derivative = [this]( auto derivative_member, auto value_member ) {
        return [this, derivative_member, value_member]( const Point& point, Real field, int material ) {
            const auto& selected = model( material );
            const auto& exact = selected.*derivative_member;
            if( exact ) return exact( point, field, material );
            const Real step = std::cbrt( std::numeric_limits< Real >::epsilon() ) *
                              std::max( Real{ 1 }, std::abs( field ) );
            const auto& value = selected.*value_member;
            return ( value( point, field + step, material ) - value( point, field - step, material ) ) / ( 2 * step );
        };
    };
    result.reaction_derivative = derivative( &AdvectionReactionCoefficients::reaction_derivative,
                                             &AdvectionReactionCoefficients::reaction );
    result.source_derivative = derivative( &AdvectionReactionCoefficients::source_derivative,
                                           &AdvectionReactionCoefficients::source );
    return result;
}
}  // namespace KARMA
