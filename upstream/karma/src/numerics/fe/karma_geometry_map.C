// Mesh-backed geometric element mappings, independent of field bases.
#include "karma_geometry_map.h"

#include "karma_exceptions.h"
#include "karma_mesh.h"

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace
{
constexpr KARMA::Real scaled_tolerance = 256.0 * std::numeric_limits< KARMA::Real >::epsilon();

bool in_reference_cell( const KARMA::KARMAReferenceTopology& reference, const KARMA::Point& point, KARMA::Real tolerance )
{
    const auto type = reference.topology();
    if( type == moab::MBPYRAMID )
        return point( 2 ) >= -tolerance && point( 2 ) <= 1 + tolerance &&
               std::abs( point( 0 ) ) <= 1 - point( 2 ) + tolerance && std::abs( point( 1 ) ) <= 1 - point( 2 ) + tolerance;
    for( unsigned d = 0; d < reference.dimension(); ++d )
        if( point( d ) < -tolerance || point( d ) > 1 + tolerance ) return false;
    if( type == moab::MBTRI || type == moab::MBTET || type == moab::MBPRISM )
        if( point( 0 ) + point( 1 ) + ( type == moab::MBTET ? point( 2 ) : 0.0 ) > 1 + tolerance ) return false;
    return true;
}

KARMA::Point reference_outward_normal( moab::EntityType topology, unsigned side )
{
    using KARMA::Point;
    switch( topology )
    {
        case moab::MBTRI: return std::array{ Point(0,-1,0), Point(1,1,0), Point(-1,0,0) }[side];
        case moab::MBQUAD: return std::array{ Point(0,-1,0), Point(1,0,0), Point(0,1,0), Point(-1,0,0) }[side];
        case moab::MBTET: return std::array{ Point(0,-1,0), Point(1,1,1), Point(-1,0,0), Point(0,0,-1) }[side];
        case moab::MBHEX: return std::array{ Point(0,-1,0), Point(1,0,0), Point(0,1,0), Point(-1,0,0), Point(0,0,-1), Point(0,0,1) }[side];
        case moab::MBPRISM: return std::array{ Point(0,-1,0), Point(1,1,0), Point(-1,0,0), Point(0,0,-1), Point(0,0,1) }[side];
        case moab::MBPYRAMID: return std::array{ Point(0,-1,1), Point(1,0,1), Point(0,1,1), Point(-1,0,1), Point(0,0,-1) }[side];
        default: KARMA_THROW( KARMA::InvalidArgument( "unsupported topology for side normal" ) );
    }
}

KARMA::Point from_sampling_coordinates( const KARMA::KARMAReferenceTopology& reference,
                                         const KARMA::Point& coordinate )
{
    if( reference.topology() != moab::MBPYRAMID ) return coordinate;
    const KARMA::Real height = 1.0 - coordinate( 2 );
    return KARMA::Point( ( 2.0 * coordinate( 0 ) - 1.0 ) * height,
                         ( 2.0 * coordinate( 1 ) - 1.0 ) * height, coordinate( 2 ) );
}
}  // namespace

namespace KARMA
{
KARMAGeometryMap::KARMAGeometryMap( KARMAReferenceTopology reference, std::span< const Point > nodes )
    : _reference( std::move( reference ) )
{
    reinit( nodes );
}

KARMAGeometryMap::KARMAGeometryMap( KARMAReferenceTopology reference ) : _reference( std::move( reference ) ) {}

void KARMAGeometryMap::reinit( std::span< const Point > nodes )
{
    if( nodes.size() != _reference.node_count() )
        KARMA_THROW( InvalidArgument( "geometry node count does not match variant" ) );
    std::copy( nodes.begin(), nodes.end(), _nodes.begin() );
    _node_count = nodes.size();
    update_affine_quad4_cache();
    validate();
}

void KARMAGeometryMap::update_affine_quad4_cache()
{
    _has_affine_quad4_cache = false;
    if( _reference.variant() != GeometryVariant::Quad4 ||
        _nodes[0] - _nodes[1] + _nodes[2] - _nodes[3] != Point::Zero() )
        return;

    _affine_quad4_origin = _nodes[0];
    _affine_quad4_r = _nodes[1] - _affine_quad4_origin;
    _affine_quad4_s = _nodes[3] - _affine_quad4_origin;
    _affine_quad4_jacobian = Matrix3::Zero();
    _affine_quad4_jacobian.col( 0 ) = _affine_quad4_r;
    _affine_quad4_jacobian.col( 1 ) = _affine_quad4_s;
    _affine_quad4_metric = Matrix3::Zero();
    _affine_quad4_metric.topLeftCorner< 2, 2 >() =
        _affine_quad4_jacobian.leftCols< 2 >().transpose() * _affine_quad4_jacobian.leftCols< 2 >();
    _affine_quad4_measure = cross( _affine_quad4_r, _affine_quad4_s ).norm();
    _has_affine_quad4_cache = true;
}

KARMAGeometryMap::KARMAGeometryMap( const KARMAMesh& mesh, moab::EntityHandle element )
    : _reference( [&] {
          if( element == 0 ) KARMA_THROW( InvalidArgument( "geometry element handle must be nonzero" ) );
          const moab::EntityHandle* connectivity = nullptr;
          int count = 0;
          if( mesh.moab()->get_connectivity( element, connectivity, count ) != moab::MB_SUCCESS ) KARMA_THROW( LogicError() );
          return KARMAReferenceTopology( KARMAReferenceTopology::from_moab( mesh.moab()->type_from_handle( element ),
                                                                           static_cast< std::size_t >( count ) ) );
      }() )
{
    std::array< Real, 81 > coordinates{};
    const std::size_t count = mesh.element_coordinates( element, coordinates );
    _node_count = count;
    for( std::size_t i = 0; i < count; ++i ) _nodes[i] = point_at( coordinates.data(), i );
    update_affine_quad4_cache();
    validate();
}

Point KARMAGeometryMap::map( const Point& reference_point ) const
{
    if( _has_affine_quad4_cache )
        return _affine_quad4_origin + reference_point( 0 ) * _affine_quad4_r + reference_point( 1 ) * _affine_quad4_s;
    _reference.evaluate_values( reference_point, std::span< Real >( _values.data(), _node_count ) );
    Point result = Point::Zero();
    for( std::size_t i = 0; i < _node_count; ++i ) result += _values[i] * _nodes[i];
    return result;
}

Matrix3 KARMAGeometryMap::jacobian( const Point& reference_point ) const
{
    if( _reference.topology() == moab::MBPYRAMID && reference_point == Point( 0, 0, 1 ) )
        KARMA_THROW( InvalidArgument( "pyramid Jacobian is path-dependent at the exact apex" ) );
    if( _has_affine_quad4_cache ) return _affine_quad4_jacobian;
    _reference.evaluate( reference_point, std::span< Real >( _values.data(), _node_count ),
                         std::span< Point >( _derivatives.data(), _node_count ) );
    Matrix3 result = Matrix3::Zero();
    for( std::size_t i = 0; i < _node_count; ++i )
        for( unsigned d = 0; d < _reference.dimension(); ++d ) result.col( d ) += _derivatives[i]( d ) * _nodes[i];
    return result;
}

Matrix3 KARMAGeometryMap::metric( const Point& reference_point ) const
{
    if( _has_affine_quad4_cache ) return _affine_quad4_metric;
    const Matrix3 j = jacobian( reference_point );
    Matrix3 result = Matrix3::Zero();
    result.topLeftCorner( _reference.dimension(), _reference.dimension() ) =
        j.leftCols( _reference.dimension() ).transpose() * j.leftCols( _reference.dimension() );
    return result;
}

Real KARMAGeometryMap::measure( const Point& reference_point ) const
{
    if( _has_affine_quad4_cache ) return _affine_quad4_measure;
    const Matrix3 j = jacobian( reference_point );
    if( _reference.dimension() == 1 ) return j.col( 0 ).norm();
    if( _reference.dimension() == 2 ) return cross( j.col( 0 ), j.col( 1 ) ).norm();
    return j.determinant();
}

InverseMapResult KARMAGeometryMap::try_inverse_map( const Point& physical_point, Real reference_tolerance,
                                                       Real residual_tolerance ) const noexcept
{
    try
    {
        if( !physical_point.allFinite() || reference_tolerance < 0.0 || residual_tolerance < 0.0 )
            return { InverseMapStatus::InvalidGeometry, Point::Zero(), std::numeric_limits< Real >::infinity() };
        Point lower = _nodes[0];
        Point upper = _nodes[0];
        for( std::size_t node = 1; node < _node_count; ++node )
        {
            lower = lower.cwiseMin( _nodes[node] );
            upper = upper.cwiseMax( _nodes[node] );
        }
        const Real box_tolerance = std::max( reference_tolerance, residual_tolerance );
        for( unsigned dimension = 0; dimension < 3; ++dimension )
            if( physical_point( static_cast<Eigen::Index>( dimension ) ) < lower( static_cast<Eigen::Index>( dimension ) ) - box_tolerance ||
                physical_point( static_cast<Eigen::Index>( dimension ) ) > upper( static_cast<Eigen::Index>( dimension ) ) + box_tolerance )
                return { InverseMapStatus::NotContained, Point::Zero(), std::numeric_limits< Real >::infinity() };
        const Point reference = inverse_map( physical_point );
        if( !in_reference_cell( _reference, reference, reference_tolerance ) )
            return { InverseMapStatus::NotContained, reference, ( map( reference ) - physical_point ).norm() };
        const Real residual = ( map( reference ) - physical_point ).norm();
        if( !std::isfinite( residual ) || residual > residual_tolerance )
            return { InverseMapStatus::NotContained, reference, residual };
        if( !std::isfinite( measure( reference ) ) || measure( reference ) <= 0.0 )
            return { InverseMapStatus::InvalidGeometry, reference, residual };
        return { InverseMapStatus::Located, reference, residual };
    }
    catch( ... )
    {
        return { InverseMapStatus::InvalidGeometry, Point::Zero(), std::numeric_limits< Real >::infinity() };
    }
}

Point KARMAGeometryMap::inverse_map( const Point& physical_point ) const
{
    if( !physical_point.allFinite() ) KARMA_THROW( InvalidArgument( "inverse-map point must be finite" ) );
    const Real scale = [&] {
        Real value = 0.0;
        for( std::size_t i = 0; i < _node_count; ++i )
            for( std::size_t j = i + 1; j < _node_count; ++j ) value = std::max( value, ( _nodes[i] - _nodes[j] ).norm() );
        return value;
    }();
    if( _reference.topology() == moab::MBPYRAMID && ( physical_point - _nodes[4] ).norm() <= 2.e-12 * scale )
        return Point( 0, 0, 1 );
    Point reference = _reference.center();
    constexpr int maximum_iterations = 40;
    const Real tolerance = 2.e-12;
    for( int iteration = 0; iteration < maximum_iterations; ++iteration )
    {
        const Point residual = map( reference ) - physical_point;
        if( residual.norm() <= tolerance * scale ) return reference;
        const Matrix3 j = jacobian( reference );
        const unsigned dimension = _reference.dimension();
        const Matrix3 metric = j.transpose() * j;
        Matrix3 inverse = Matrix3::Zero();
        const Real metric_scale = metric.topLeftCorner( dimension, dimension ).cwiseAbs().maxCoeff();
        Real determinant = 0.0;
        if( dimension == 1 )
        {
            determinant = metric( 0, 0 );
            inverse( 0, 0 ) = 1.0 / determinant;
        }
        else if( dimension == 2 )
        {
            determinant = metric.topLeftCorner< 2, 2 >().determinant();
            inverse.topLeftCorner< 2, 2 >() = metric.topLeftCorner< 2, 2 >().inverse();
        }
        else
        {
            determinant = metric.determinant();
            inverse = metric.inverse();
        }
        if( !std::isfinite( determinant ) || metric_scale <= 0.0 ||
            determinant <= scaled_tolerance * std::pow( metric_scale, static_cast< int >( dimension ) ) )
            KARMA_THROW( InvalidArgument( "inverse map encountered a rank-deficient or ill-conditioned Jacobian" ) );
        const Vector3 correction = inverse * j.transpose() * residual;
        if( !correction.allFinite() ) KARMA_THROW( InvalidArgument( "inverse map correction is nonfinite" ) );
        Real damping = 1.0;
        Point candidate = reference;
        bool accepted = false;
        while( damping >= 1.0 / 1024.0 )
        {
            candidate = reference;
            candidate.head( _reference.dimension() ) -= damping * correction;
            if( in_reference_cell( _reference, candidate, 0.0 ) &&
                ( map( candidate ) - physical_point ).squaredNorm() < residual.squaredNorm() )
            {
                accepted = true;
                break;
            }
            damping *= 0.5;
        }
        if( !accepted ) KARMA_THROW( InvalidArgument( "inverse map line search failed" ) );
        reference = candidate;
    }
    KARMA_THROW( InvalidArgument( "inverse map failed to converge" ) );
}

SideGeometry KARMAGeometryMap::side( unsigned side_index, const Point& side_point ) const
{
    if( side_index >= _reference.side_count() ) KARMA_THROW( InvalidArgument( "side index is out of range" ) );
    return side_from_parent( side_index, _reference.side_reference_map( side_index, side_point ),
                             _reference.side_reference_jacobian( side_index, side_point ) );
}

SideGeometry KARMAGeometryMap::side_from_parent( unsigned side_index, const Point& parent,
                                                 const Matrix3& parent_jacobian ) const
{
    if( side_index >= _reference.side_count() ) KARMA_THROW( InvalidArgument( "side index is out of range" ) );
    const Matrix3 physical_jacobian = jacobian( parent );
    SideGeometry result;
    result.point = map( parent );
    result.tangents = physical_jacobian * parent_jacobian;
    if( _reference.dimension() == 2 )
    {
        const Vector3 tangent = result.tangents.col( 0 );
        const Vector3 surface_normal = cross( physical_jacobian.col( 0 ), physical_jacobian.col( 1 ) ).normalized();
        result.normal = cross( tangent, surface_normal );
        result.measure = result.normal.norm();
    }
    else if( _reference.dimension() == 3 )
    {
        result.normal = cross( result.tangents.col( 0 ), result.tangents.col( 1 ) );
        result.measure = result.normal.norm();
    }
    else
        KARMA_THROW( InvalidArgument( "zero-dimensional edge endpoints have no unique spatial normal" ) );
    Real side_scale = 0.0;
    for( std::size_t i = 0; i < _node_count; ++i )
        for( std::size_t j = i + 1; j < _node_count; ++j )
            side_scale = std::max( side_scale, ( _nodes[i] - _nodes[j] ).norm() );
    const Real side_limit = scaled_tolerance * std::pow( side_scale, static_cast< int >( _reference.dimension() - 1 ) );
    if( !std::isfinite( result.measure ) || side_scale <= 0.0 || result.measure <= side_limit )
        KARMA_THROW( InvalidArgument( "side mapping is degenerate" ) );
    result.normal /= result.measure;
    const unsigned dimension = _reference.dimension();
    const Matrix3 metric = physical_jacobian.transpose() * physical_jacobian;
    Matrix3 inverse = Matrix3::Zero();
    if( dimension == 2 ) inverse.topLeftCorner< 2, 2 >() = metric.topLeftCorner< 2, 2 >().inverse();
    else inverse = metric.inverse();
    const Vector3 reference_covector = reference_outward_normal( _reference.topology(), side_index );
    const Vector3 canonical = physical_jacobian * inverse * reference_covector;
    if( !canonical.allFinite() || canonical.norm() == 0.0 )
        KARMA_THROW( InvalidArgument( "side orientation produced a nonfinite canonical co-normal" ) );
    if( dot( result.normal, canonical ) < 0.0 ) result.normal = -result.normal;
    return result;
}

void KARMAGeometryMap::validate() const
{
    if( _node_count != _reference.node_count() ) KARMA_THROW( InvalidArgument( "geometry node count does not match variant" ) );
    for( std::size_t i = 0; i < _node_count; ++i )
        if( !_nodes[i].allFinite() ) KARMA_THROW( InvalidArgument( "geometry nodes must be finite" ) );
    Real scale = 0.0;
    for( std::size_t i = 0; i < _node_count; ++i )
        for( std::size_t j = i + 1; j < _node_count; ++j ) scale = std::max( scale, ( _nodes[i] - _nodes[j] ).norm() );
    if( !std::isfinite( scale ) || scale <= 0.0 ) KARMA_THROW( InvalidArgument( "geometry has zero or nonfinite scale" ) );

    if( _reference.variant() == GeometryVariant::Edge3 )
    {
        // x'(r) = a*r+b. Minimize |x'(r)|^2 exactly on [0,1].
        const Vector3 a = 4.0 * _nodes[0] + 4.0 * _nodes[1] - 8.0 * _nodes[2];
        const Vector3 b = -3.0 * _nodes[0] - _nodes[1] + 4.0 * _nodes[2];
        const Real stationary = a.squaredNorm() == 0.0 ? 0.0 : std::clamp( -dot( a, b ) / a.squaredNorm(), 0.0, 1.0 );
        const Real minimum_speed = std::min( { b.norm(), ( a + b ).norm(), ( stationary * a + b ).norm() } );
        if( !std::isfinite( minimum_speed ) || minimum_speed <= scaled_tolerance * scale )
            KARMA_THROW( InvalidArgument( "EDGE3 geometry has zero or ill-conditioned derivative" ) );
        // This local injectivity test does not certify absence of global self-intersection.
        return;
    }

    // Recursively sample each parameter cell at corners, edge/face midpoints,
    // and center. Cells with high metric variation refine through depth 3.
    // Pyramids recurse in collapsed coordinates. This is not a formal certificate.
    const Real limit = scaled_tolerance * std::pow( scale, static_cast< int >( _reference.dimension() ) );
    Vector3 surface_orientation = Vector3::Zero();
    Matrix3 center_jacobian = Matrix3::Zero();
    if( _reference.dimension() == 2 )
    {
        center_jacobian = jacobian( _reference.center() );
        surface_orientation = cross( center_jacobian.col( 0 ), center_jacobian.col( 1 ) );
        if( !surface_orientation.allFinite() || surface_orientation.norm() <= limit )
            KARMA_THROW( InvalidArgument( "surface orientation anchor is degenerate" ) );
        surface_orientation.normalize();
    }
    if( _has_affine_quad4_cache )
    {
        // x(r,s) = x0 + (x1-x0)r + (x3-x0)s + (x0-x1+x2-x3)rs.
        // A vanishing rs coefficient makes the Jacobian constant, so one exact
        // metric check is equivalent to the sampled checks below.
        const Eigen::Matrix2d metric = ( center_jacobian.leftCols< 2 >().transpose() * center_jacobian.leftCols< 2 >() );
        const Real trace = metric.trace();
        const Real discriminant = std::sqrt( std::pow( metric( 0, 0 ) - metric( 1, 1 ), 2 ) + 4.0 * std::pow( metric( 0, 1 ), 2 ) );
        const Real largest = 0.5 * ( trace + discriminant );
        const Real smallest = metric.determinant() / largest;
        const Real value = dot( cross( center_jacobian.col( 0 ), center_jacobian.col( 1 ) ), surface_orientation );
        if( !std::isfinite( value ) || value <= limit || !std::isfinite( largest ) || !std::isfinite( smallest ) ||
            smallest <= scaled_tolerance * scaled_tolerance * largest )
            KARMA_THROW( InvalidArgument( "geometry is degenerate, inverted, or ill-conditioned at a validation point" ) );
        return;
    }
    const auto checked_geometry = [&]( const Point& sampling_point ) {
        const Point point = from_sampling_coordinates( _reference, sampling_point );
        if( _reference.topology() == moab::MBPYRAMID && point( 2 ) == 1.0 ) return std::pair{ std::numeric_limits< Real >::quiet_NaN(), 0.0 };
        if( !in_reference_cell( _reference, point, 0.0 ) ) return std::pair{ std::numeric_limits< Real >::quiet_NaN(), 0.0 };
        const Matrix3 j = jacobian( point );
        const Matrix3 metric = j.transpose() * j;
        const unsigned dimension = _reference.dimension();
        Real smallest = 0.0;
        Real largest = 0.0;
        if( dimension == 1 ) smallest = largest = metric( 0, 0 );
        else if( dimension == 2 )
        {
            const Eigen::SelfAdjointEigenSolver< Eigen::Matrix2d > solver( metric.topLeftCorner< 2, 2 >(),
                                                                           Eigen::EigenvaluesOnly );
            smallest = solver.eigenvalues()( 0 );
            largest = solver.eigenvalues()( 1 );
        }
        else
        {
            const Eigen::SelfAdjointEigenSolver< Matrix3 > solver( metric, Eigen::EigenvaluesOnly );
            smallest = solver.eigenvalues()( 0 );
            largest = solver.eigenvalues()( 2 );
        }
        const Real metric_determinant = dimension == 1 ? metric( 0, 0 )
                                        : dimension == 2 ? metric.topLeftCorner< 2, 2 >().determinant()
                                                         : metric.determinant();
        Real value = std::sqrt( metric_determinant );
        if( _reference.dimension() == 2 )
            value = dot( cross( j.col( 0 ), j.col( 1 ) ), surface_orientation );
        else if( _reference.dimension() == 3 )
            value = j.determinant();
        if( !std::isfinite( value ) || value <= limit || !std::isfinite( largest ) ||
            smallest <= scaled_tolerance * scaled_tolerance * largest )
            KARMA_THROW( InvalidArgument( "geometry is degenerate, inverted, or ill-conditioned at a validation point" ) );
        return std::pair{ value, std::sqrt( largest / smallest ) };
    };
    constexpr unsigned initial_subdivisions = 8;
    constexpr unsigned maximum_depth = 3;
    const auto inspect = [&]( const auto& self, const Point& low, const Point& high, unsigned depth ) -> void {
        Real minimum = std::numeric_limits< Real >::max(), maximum = 0.0, maximum_condition = 1.0;
        unsigned valid_count = 0;
        const Point middle = .5 * ( low + high );
        unsigned sample_count = 1;
        for( unsigned d = 0; d < _reference.dimension(); ++d ) sample_count *= 3;
        for( unsigned code = 0; code < sample_count; ++code )
        {
            Point sample = low;
            unsigned digits = code;
            for( unsigned d = 0; d < _reference.dimension(); ++d )
            {
                const unsigned location = digits % 3;
                digits /= 3;
                sample( d ) = location == 0 ? low( d ) : location == 1 ? middle( d ) : high( d );
            }
            const auto [value, condition] = checked_geometry( sample );
            if( std::isfinite( value ) )
            {
                minimum = std::min( minimum, value );
                maximum = std::max( maximum, value );
                maximum_condition = std::max( maximum_condition, condition );
                ++valid_count;
            }
        }
        if( depth == maximum_depth || valid_count == 0 ||
            ( maximum <= 4.0 * minimum && maximum_condition <= 16.0 ) ) return;
        const unsigned combinations = 1u << _reference.dimension();
        for( unsigned mask = 0; mask < combinations; ++mask )
        {
            Point child_low = low, child_high = high;
            for( unsigned d = 0; d < _reference.dimension(); ++d )
                if( mask & ( 1u << d ) ) child_low( d ) = middle( d ); else child_high( d ) = middle( d );
            self( self, child_low, child_high, depth + 1 );
        }
    };
    for( unsigned k = 0; k < ( _reference.dimension() == 3 ? initial_subdivisions : 1 ); ++k )
        for( unsigned j = 0; j < ( _reference.dimension() >= 2 ? initial_subdivisions : 1 ); ++j )
            for( unsigned i = 0; i < initial_subdivisions; ++i )
            {
                Point low( i / 8.0, j / 8.0, k / 8.0 );
                Point high( ( i + 1 ) / 8.0,
                            _reference.dimension() >= 2 ? ( j + 1 ) / 8.0 : 0.0,
                            _reference.dimension() == 3 ? ( k + 1 ) / 8.0 : 0.0 );
                inspect( inspect, low, high, 0 );
            }
    if( _reference.topology() == moab::MBPYRAMID )
        for( unsigned level = 4; level <= 40; ++level )
        {
            const Real height = std::ldexp( 1.0, -static_cast< int >( level ) );
            for( const Real a : { .125, .5, .875 } )
                for( const Real b : { .125, .5, .875 } ) (void)checked_geometry( Point( a, b, 1.0 - height ) );
        }
}
}  // namespace KARMA
