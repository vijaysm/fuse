// KARMA finite-element basis evaluation.

#include "karma_fe_base.h"

#include "karma_exceptions.h"
#include "karma_mesh.h"

#include "moab/CN.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>

namespace
{
void check_petsc( PetscErrorCode ierr, const char* operation )
{
    if( ierr == PETSC_SUCCESS ) return;
    const char* text = nullptr;
    PetscErrorMessage( ierr, &text, nullptr );
    std::ostringstream message;
    message << operation << " failed with PETSc error " << ierr;
    if( text ) message << ": " << text;
    std::string detail = message.str();
    KARMA_THROW( KARMA::LogicError( detail ) );
}

bool has_triangular_sides( moab::EntityType topology )
{
    return topology == moab::MBTET || topology == moab::MBPRISM || topology == moab::MBPYRAMID;
}

bool has_quadrilateral_sides( moab::EntityType topology )
{
    return topology == moab::MBHEX || topology == moab::MBPRISM || topology == moab::MBPYRAMID;
}

moab::EntityType side_topology( moab::EntityType topology )
{
    switch( topology )
    {
        case moab::MBEDGE:
            return moab::MBVERTEX;
        case moab::MBTRI:
        case moab::MBQUAD:
            return moab::MBEDGE;
        case moab::MBTET:
            return moab::MBTRI;
        case moab::MBHEX:
        case moab::MBPRISM:
            return moab::MBQUAD;
        case moab::MBPYRAMID:
            return moab::MBQUAD;
        default:
            KARMA_THROW( KARMA::InvalidArgument( "unsupported finite-element topology" ) );
    }
}
}  // namespace

namespace KARMA
{
KARMAFEBase::KARMAFEBase( const KARMAMesh& mesh, GeometryVariant geometry_variant, KARMABasis basis,
                           int quadrature_degree, FEIntegrationDomain domain )
    : _mesh( mesh ), _geometry_variant( geometry_variant ), _geometry_reference( geometry_variant ),
      _basis( std::move( basis ) ), _geometry( _geometry_reference ), _topology( _geometry_reference.topology() ),
      _quadrature( domain == FEIntegrationDomain::Volume ? _topology : side_topology( _topology ), quadrature_degree ),
      _triangle_quadrature( domain == FEIntegrationDomain::Face && has_triangular_sides( _topology ) &&
                                    has_quadrilateral_sides( _topology )
                                ? std::make_unique< KARMAQuadrature >( moab::MBTRI, quadrature_degree )
                                : nullptr ),
      _domain( domain ), _n_basis( _basis.size() ),
      _capacity( std::max( _quadrature.n_points(), _triangle_quadrature ? _triangle_quadrature->n_points() : 0 ) ),
      _active_qpoints( domain == FEIntegrationDomain::Volume ? _quadrature.n_points() : _capacity ), _jxw( _capacity ),
      _phi( _n_basis, std::vector< Real >( _capacity ) ),
      _dphi( _n_basis, std::vector< Gradient >( _capacity, Gradient::Zero() ) ), _xyz( _capacity, Point::Zero() ),
      _normals( _capacity, Vector3::Zero() ), _coordinates( 3 * _geometry_reference.node_count() ),
      _reference_phi( _n_basis ), _reference_gradient( _n_basis ),
      _side_parent_points( _domain == FEIntegrationDomain::Face ? _geometry_reference.side_count() * _capacity : 0 ),
      _side_parent_jacobians( _side_parent_points.size(), Matrix3::Zero() ),
      _physical_point_scratch( _domain == FEIntegrationDomain::Face ? _capacity : 0 ),
      _reference_point_scratch( _physical_point_scratch.size() )
{
    if( reference_cell( geometry_variant ) != _basis.key().cell )
        KARMA_THROW( InvalidArgument( "geometry and field basis reference cells must match" ) );
    if( static_cast< int >( mesh.dimension() ) < moab::CN::Dimension( _topology ) )
        KARMA_THROW( InvalidArgument( "finite-element topology dimension exceeds physical mesh dimension" ) );

    invalidate();
    if( _domain == FEIntegrationDomain::Face )
        for( unsigned side = 0; side < _geometry_reference.side_count(); ++side )
        {
            const auto actual = moab::CN::SubEntityType( _topology,
                static_cast< int >( _geometry_reference.dimension() - 1 ), static_cast< int >( side ) );
            const KARMAQuadrature& rule = actual == moab::MBTRI && _triangle_quadrature
                                              ? *_triangle_quadrature
                                              : _quadrature;
            PetscInt side_dimension = 0;
            PetscInt side_count = 0;
            const PetscReal* side_points = nullptr;
            check_petsc( PetscQuadratureGetData( rule.petsc_quadrature(), &side_dimension, nullptr, &side_count,
                                                 &side_points, nullptr ), "query face quadrature" );
            for( PetscInt q = 0; q < side_count; ++q )
            {
                const std::size_t index = side * _capacity + static_cast< std::size_t >( q );
                if( _geometry_reference.dimension() == 1 )
                {
                    _side_parent_points[index] = Point( static_cast< Real >( side ), 0, 0 );
                    continue;
                }
                Point point = Point::Zero();
                for( PetscInt d = 0; d < side_dimension; ++d ) point( d ) = side_points[q * side_dimension + d];
                _side_parent_points[index] = _geometry_reference.side_reference_map( side, point );
                _side_parent_jacobians[index] = _geometry_reference.side_reference_jacobian( side, point );
            }
        }
}

void KARMAFEBase::set_active_size( std::size_t count )
{
    if( count > _capacity ) KARMA_THROW( InvalidArgument( "active quadrature count exceeds fixed FE capacity" ) );
    _active_qpoints = count;
    _jxw.resize( count );
    _xyz.resize( count );
    _normals.resize( count );
    for( auto& row : _phi ) row.resize( count );
    for( auto& row : _dphi ) row.resize( count );
}

void KARMAFEBase::invalidate() noexcept
{
    _valid             = false;
    const Real invalid = std::numeric_limits< Real >::quiet_NaN();
    std::fill( _jxw.begin(), _jxw.end(), invalid );
    for( auto& row : _phi )
        std::fill( row.begin(), row.end(), invalid );
    for( auto& row : _dphi )
        for( Gradient& value : row )
            value.setConstant( invalid );
    for( Point& point : _xyz )
        point.setConstant( invalid );
    for( Vector3& normal : _normals )
        normal.setConstant( invalid );
}

void KARMAFEBase::reinit( moab::EntityHandle element )
{
    invalidate();
    try
    {
        if( _domain != FEIntegrationDomain::Volume )
            KARMA_THROW( InvalidArgument( "volume reinit called on face finite element" ) );
        if( element == 0 || _mesh.moab()->type_from_handle( element ) != _topology )
            KARMA_THROW( InvalidArgument( "finite-element object cannot change topology" ) );

        const std::size_t nconn = _mesh.element_coordinates( element, _coordinates );
        if( nconn != _geometry_reference.node_count() )
            KARMA_THROW( InvalidArgument( "element connectivity does not match bound geometry variant" ) );
        for( std::size_t i = 0; i < nconn; ++i ) _geometry_nodes[i] = point_at( _coordinates.data(), i );
        _geometry.reinit( std::span< const Point >( _geometry_nodes.data(), nconn ) );
        evaluate_reference_points( _geometry, _quadrature.petsc_quadrature(), true );
        _valid = true;
    }
    catch( ... )
    {
        invalidate();
        throw;
    }
}

void KARMAFEBase::evaluate_reference_points( const KARMAGeometryMap& geometry, PetscQuadrature quadrature,
                                              bool compute_weights )
{
    PetscInt quadrature_dimension = 0;
    PetscInt count                = 0;
    const PetscReal* points       = nullptr;
    const PetscReal* weights      = nullptr;
    check_petsc( PetscQuadratureGetData( quadrature, &quadrature_dimension, nullptr, &count, &points, &weights ),
                 "query finite-element quadrature" );
    const unsigned reference_dimension = geometry.reference_dimension();
    if( quadrature_dimension != static_cast< PetscInt >( reference_dimension ) ||
        count != static_cast< PetscInt >( _active_qpoints ) )
        KARMA_THROW( InvalidArgument( "finite-element quadrature shape does not match fixed storage" ) );

    constexpr Real condition_tolerance = 256.0 * std::numeric_limits< Real >::epsilon();
    for( std::size_t q = 0; q < _active_qpoints; ++q )
    {
        Point natural = Point::Zero();
        for( unsigned d = 0; d < reference_dimension; ++d )
            natural( d ) = points[q * reference_dimension + d];
        _basis.evaluate( natural, _reference_phi, _reference_gradient );

        const Matrix3 jacobian = geometry.jacobian( natural );
        const Matrix3 metric = geometry.metric( natural );
        Matrix3 metric_inverse = Matrix3::Zero();
        const Real scale = metric.topLeftCorner( reference_dimension, reference_dimension ).cwiseAbs().maxCoeff();
        Real determinant = 0.0;
        if( reference_dimension == 1 )
        {
            determinant = metric( 0, 0 );
            metric_inverse( 0, 0 ) = 1.0 / determinant;
        }
        else if( reference_dimension == 2 )
        {
            determinant = metric.topLeftCorner< 2, 2 >().determinant();
            metric_inverse.topLeftCorner< 2, 2 >() = metric.topLeftCorner< 2, 2 >().inverse();
        }
        else
        {
            determinant = metric.determinant();
            metric_inverse = metric.inverse();
        }
        if( !std::isfinite( determinant ) || scale <= 0.0 ||
            determinant <= condition_tolerance * std::pow( scale, static_cast< int >( reference_dimension ) ) )
            KARMA_THROW( InvalidArgument( "finite-element Jacobian is rank-deficient or ill-conditioned" ) );
        const Matrix3 gradient_transform = jacobian * metric_inverse;

        _xyz[q] = geometry.map( natural );
        for( std::size_t i = 0; i < _n_basis; ++i )
        {
            _phi[i][q] = _reference_phi[i];
            _dphi[i][q] = gradient_transform * _reference_gradient[i];
        }
        if( compute_weights ) _jxw[q] = weights[q] * geometry.measure( natural );
    }
}

void KARMAFEBase::evaluate_reference_points( const KARMAGeometryMap& geometry,
                                              std::span< const Point > points )
{
    constexpr Real condition_tolerance = 256.0 * std::numeric_limits< Real >::epsilon();
    const unsigned reference_dimension = geometry.reference_dimension();
    for( std::size_t q = 0; q < points.size(); ++q )
    {
        const Point& natural = points[q];
        _basis.evaluate( natural, _reference_phi, _reference_gradient );
        const Matrix3 jacobian = geometry.jacobian( natural );
        const Matrix3 metric = geometry.metric( natural );
        Matrix3 metric_inverse = Matrix3::Zero();
        const Real scale = metric.topLeftCorner( reference_dimension, reference_dimension ).cwiseAbs().maxCoeff();
        Real determinant = 0.0;
        if( reference_dimension == 1 )
        {
            determinant = metric( 0, 0 );
            metric_inverse( 0, 0 ) = 1.0 / determinant;
        }
        else if( reference_dimension == 2 )
        {
            determinant = metric.topLeftCorner< 2, 2 >().determinant();
            metric_inverse.topLeftCorner< 2, 2 >() = metric.topLeftCorner< 2, 2 >().inverse();
        }
        else
        {
            determinant = metric.determinant();
            metric_inverse = metric.inverse();
        }
        if( !std::isfinite( determinant ) || scale <= 0.0 ||
            determinant <= condition_tolerance * std::pow( scale, static_cast< int >( reference_dimension ) ) )
            KARMA_THROW( InvalidArgument( "finite-element Jacobian is rank-deficient or ill-conditioned" ) );
        const Matrix3 gradient_transform = jacobian * metric_inverse;
        _xyz[q] = geometry.map( natural );
        for( std::size_t i = 0; i < _n_basis; ++i )
        {
            _phi[i][q] = _reference_phi[i];
            _dphi[i][q] = gradient_transform * _reference_gradient[i];
        }
    }
}

void KARMAFEBase::reinit( moab::EntityHandle element, unsigned side )
{
    invalidate();
    try
    {
        if( _domain != FEIntegrationDomain::Face )
            KARMA_THROW( InvalidArgument( "face reinit called on volume finite element" ) );
        if( element == 0 || _mesh.moab()->type_from_handle( element ) != _topology )
            KARMA_THROW( InvalidArgument( "finite-element object cannot change topology" ) );
        if( side >= _geometry_reference.side_count() )
            KARMA_THROW( InvalidArgument( "finite-element side index is out of range" ) );

        const std::size_t nconn = _mesh.element_coordinates( element, _coordinates );
        if( nconn != _geometry_reference.node_count() )
            KARMA_THROW( InvalidArgument( "element connectivity does not match bound geometry variant" ) );
        for( std::size_t i = 0; i < nconn; ++i ) _geometry_nodes[i] = point_at( _coordinates.data(), i );
        _geometry.reinit( std::span< const Point >( _geometry_nodes.data(), nconn ) );

        const moab::EntityType actual_side_topology =
            moab::CN::SubEntityType( _topology, static_cast< int >( _geometry_reference.dimension() - 1 ),
                                     static_cast< int >( side ) );
        const KARMAQuadrature& rule = actual_side_topology == moab::MBTRI && _triangle_quadrature
                                          ? *_triangle_quadrature
                                          : _quadrature;
        PetscInt qdim            = 0;
        PetscInt count           = 0;
        const PetscReal* points  = nullptr;
        const PetscReal* weights = nullptr;
        check_petsc( PetscQuadratureGetData( rule.petsc_quadrature(), &qdim, nullptr, &count, &points, &weights ),
                     "query face quadrature" );
        set_active_size( static_cast< std::size_t >( count ) );
        invalidate();
        for( std::size_t q = 0; q < _active_qpoints; ++q )
        {
            Point side_point = Point::Zero();
            for( PetscInt d = 0; d < qdim; ++d ) side_point( d ) = points[q * qdim + d];
            const Real reference_weight = weights[q];
            const std::size_t side_offset = side * _capacity + q;
            if( _geometry_reference.dimension() == 1 )
            {
                const Matrix3 jacobian = _geometry.jacobian( _side_parent_points[side_offset] );
                _xyz[q] = _geometry.map( _side_parent_points[side_offset] );
                _normals[q] = ( side == 0 ? -1.0 : 1.0 ) * jacobian.col( 0 ).normalized();
                _jxw[q] = reference_weight;
            }
            else
            {
                const SideGeometry side_geometry = _geometry.side_from_parent(
                    side, _side_parent_points[side_offset], _side_parent_jacobians[side_offset] );
                _xyz[q] = side_geometry.point;
                _normals[q] = side_geometry.normal;
                _jxw[q] = reference_weight * side_geometry.measure;
            }
        }
        evaluate_physical_points( _geometry, _xyz, true );
        _valid = true;
    }
    catch( ... )
    {
        invalidate();
        throw;
    }
}

void KARMAFEBase::reinit_at_physical_points( moab::EntityHandle element, std::span< const Point > points )
{
    if( _domain != FEIntegrationDomain::Face )
        KARMA_THROW( InvalidArgument( "physical-point reinit requires face finite element" ) );
    if( points.empty() || points.size() > _capacity )
        KARMA_THROW( InvalidArgument( "physical-point count must be nonzero and within fixed FE capacity" ) );
    std::copy( points.begin(), points.end(), _physical_point_scratch.begin() );
    set_active_size( points.size() );
    invalidate();
    try
    {
        if( element == 0 || _mesh.moab()->type_from_handle( element ) != _topology )
            KARMA_THROW( InvalidArgument( "finite-element object cannot change topology" ) );

        const std::size_t nconn = _mesh.element_coordinates( element, _coordinates );
        if( nconn != _geometry_reference.node_count() )
            KARMA_THROW( InvalidArgument( "element connectivity does not match bound geometry variant" ) );
        for( std::size_t i = 0; i < nconn; ++i ) _geometry_nodes[i] = point_at( _coordinates.data(), i );
        _geometry.reinit( std::span< const Point >( _geometry_nodes.data(), nconn ) );
        evaluate_physical_points( _geometry,
                                  std::span< const Point >( _physical_point_scratch.data(), _active_qpoints ), false );
        _valid = true;
    }
    catch( ... )
    {
        invalidate();
        throw;
    }
}

void KARMAFEBase::evaluate_physical_points( const KARMAGeometryMap& geometry, std::span< const Point > points,
                                             bool preserve_face_geometry )
{
    if( _domain != FEIntegrationDomain::Face )
        KARMA_THROW( InvalidArgument( "physical-point reinit requires face finite element" ) );
    if( points.size() != _active_qpoints )
        KARMA_THROW( InvalidArgument( "physical-point count does not match fixed FE storage" ) );
    for( const Point& point : points )
        if( !point.allFinite() ) KARMA_THROW( InvalidArgument( "physical point coordinates must be finite" ) );

    for( std::size_t q = 0; q < _active_qpoints; ++q )
        _reference_point_scratch[q] = geometry.inverse_map( points[q] );

    evaluate_reference_points( geometry,
                               std::span< const Point >( _reference_point_scratch.data(), _active_qpoints ) );

    if( !preserve_face_geometry )
    {
        const Real invalid = std::numeric_limits< Real >::quiet_NaN();
        std::fill( _jxw.begin(), _jxw.end(), invalid );
        for( Vector3& normal : _normals )
            normal.setConstant( invalid );
    }
}
}  // namespace KARMA
