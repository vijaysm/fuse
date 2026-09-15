// Tests for quadrature and finite-element evaluation.

#include "karma_basis.h"
#include "karma_exceptions.h"
#include "karma_fe_base.h"
#include "karma_mesh.h"
#include "karma_quadrature.h"

#include <petsc.h>

#include <array>
#include <atomic>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <new>
#include <type_traits>

static_assert( !std::is_copy_constructible_v< KARMA::KARMAQuadrature > );
static_assert( !std::is_copy_assignable_v< KARMA::KARMAQuadrature > );
static_assert( !std::is_move_constructible_v< KARMA::KARMAQuadrature > );
static_assert( !std::is_move_assignable_v< KARMA::KARMAQuadrature > );
static_assert( std::is_nothrow_destructible_v< KARMA::KARMAQuadrature > );
static_assert( std::is_nothrow_destructible_v< KARMA::KARMAFEBase > );

namespace
{
std::atomic< std::size_t > allocation_count{ 0 };
std::atomic< bool > count_allocations{ false };
}

void* operator new( std::size_t size )
{
    if( count_allocations.load( std::memory_order_relaxed ) )
        allocation_count.fetch_add( 1, std::memory_order_relaxed );
    if( void* memory = std::malloc( size ) ) return memory;
    throw std::bad_alloc();
}

void* operator new[]( std::size_t size )
{
    if( count_allocations.load( std::memory_order_relaxed ) )
        allocation_count.fetch_add( 1, std::memory_order_relaxed );
    if( void* memory = std::malloc( size ) ) return memory;
    throw std::bad_alloc();
}

void operator delete( void* memory ) noexcept { std::free( memory ); }
void operator delete( void* memory, std::size_t ) noexcept { std::free( memory ); }
void operator delete[]( void* memory ) noexcept { std::free( memory ); }
void operator delete[]( void* memory, std::size_t ) noexcept { std::free( memory ); }
void* operator new( std::size_t size, std::align_val_t alignment )
{
    if( count_allocations.load( std::memory_order_relaxed ) )
        allocation_count.fetch_add( 1, std::memory_order_relaxed );
    void* memory = nullptr;
    if( posix_memalign( &memory, static_cast< std::size_t >( alignment ), size ) != 0 ) throw std::bad_alloc();
    return memory;
}
void* operator new[]( std::size_t size, std::align_val_t alignment ) { return ::operator new( size, alignment ); }
void operator delete( void* memory, std::align_val_t ) noexcept { std::free( memory ); }
void operator delete( void* memory, std::size_t, std::align_val_t ) noexcept { std::free( memory ); }
void operator delete[]( void* memory, std::align_val_t ) noexcept { std::free( memory ); }
void operator delete[]( void* memory, std::size_t, std::align_val_t ) noexcept { std::free( memory ); }

#include <string>
#include <utility>
#include <vector>

namespace
{
int failures = 0;

void check( bool ok, const std::string& what )
{
    if( !ok )
    {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

void close( KARMA::Real got, KARMA::Real expected, const std::string& what, KARMA::Real tol = 1.e-11 )
{
    if( !std::isfinite( got ) || !std::isfinite( expected ) || std::abs( got - expected ) > tol )
    {
        std::cerr << "FAIL: " << what << " (got " << got << ", expected " << expected << ")\n";
        ++failures;
    }
}

template < typename F >
bool throws( F&& fn )
{
    try
    {
        fn();
    }
    catch( const std::logic_error& )
    {
        return true;
    }
    catch( ... )
    {
        return false;
    }
    return false;
}

KARMA::GeometryVariant linear_variant( moab::EntityType type )
{
    using KARMA::GeometryVariant;
    if( type == moab::MBEDGE ) return GeometryVariant::Edge2;
    if( type == moab::MBTRI ) return GeometryVariant::Tri3;
    if( type == moab::MBQUAD ) return GeometryVariant::Quad4;
    if( type == moab::MBTET ) return GeometryVariant::Tet4;
    if( type == moab::MBHEX ) return GeometryVariant::Hex8;
    if( type == moab::MBPRISM ) return GeometryVariant::Prism6;
    return GeometryVariant::Pyramid5;
}

const char* variant_name( KARMA::GeometryVariant variant )
{
    using KARMA::GeometryVariant;
    switch( variant )
    {
        case GeometryVariant::Prism6: return "Prism6";
        case GeometryVariant::Pyramid5: return "Pyramid5";
        case GeometryVariant::Pyramid13: return "Pyramid13";
        case GeometryVariant::Pyramid14: return "Pyramid14";
        default: return "other";
    }
}

KARMA::KARMABasis p1_basis( KARMA::GeometryVariant variant )
{
    using namespace KARMA;
    return KARMABasisRegistry().create( { reference_cell( variant ), FieldBasisFamily::LagrangeNodal, 1,
                                          BasisContinuity::cG } );
}

moab::EntityHandle add_element( KARMA::KARMAMesh& mesh, moab::EntityType type,
                                const std::vector< KARMA::Point >& points )
{
    std::vector< double > coords;
    coords.reserve( 3 * points.size() );
    for( const auto& point : points )
        for( int d = 0; d < 3; ++d )
            coords.push_back( point( d ) );

    moab::Range vertices;
    check( mesh.moab()->create_vertices( coords.data(), static_cast< int >( points.size() ), vertices ) ==
               moab::MB_SUCCESS,
           "create FE test vertices" );
    std::vector< moab::EntityHandle > connectivity( vertices.begin(), vertices.end() );
    moab::EntityHandle element = 0;
    check( mesh.moab()->create_element( type, connectivity.data(), static_cast< int >( connectivity.size() ),
                                        element ) == moab::MB_SUCCESS,
           "create FE test element" );
    mesh.assign_ids();
    mesh.update();
    return element;
}

void check_affine_tensor_gradients( moab::EntityType type, const std::vector< KARMA::Point >& points,
                                    const KARMA::KARMAFEBase& fe )
{
    using namespace KARMA;
    if( type != moab::MBQUAD && type != moab::MBHEX ) return;

    Matrix3 jacobian   = Matrix3::Zero();
    jacobian.col( 0 )  = points[1] - points[0];
    jacobian.col( 1 )  = points[3] - points[0];
    const unsigned dim = type == moab::MBQUAD ? 2 : 3;
    if( dim == 3 ) jacobian.col( 2 ) = points[4] - points[0];
    Matrix3 inverse_transpose                   = Matrix3::Zero();
    inverse_transpose.topLeftCorner( dim, dim ) = jacobian.topLeftCorner( dim, dim ).inverse().transpose();

    KARMAQuadrature quadrature( type, 2 );
    PetscInt qdim            = 0;
    PetscInt count           = 0;
    const PetscReal* natural = nullptr;
    check( PetscQuadratureGetData( quadrature.petsc_quadrature(), &qdim, nullptr, &count, &natural, nullptr ) ==
               PETSC_SUCCESS,
           "query affine gradient quadrature" );
    for( PetscInt q = 0; q < count; ++q )
    {
        const Real r = natural[q * qdim];
        const Real s = natural[q * qdim + 1];
        const Real t = dim == 3 ? natural[q * qdim + 2] : 0.0;
        const std::array< Gradient, 8 > reference =
            dim == 2 ? std::array< Gradient, 8 >{ Gradient( -( 1 - s ), -( 1 - r ), 0 ), Gradient( 1 - s, -r, 0 ),
                                                  Gradient( s, r, 0 ), Gradient( -s, 1 - r, 0 ) }
                     : std::array< Gradient, 8 >{ Gradient( -( 1 - s ) * ( 1 - t ), -( 1 - r ) * ( 1 - t ),
                                                            -( 1 - r ) * ( 1 - s ) ),
                                                  Gradient( ( 1 - s ) * ( 1 - t ), -r * ( 1 - t ), -r * ( 1 - s ) ),
                                                  Gradient( s * ( 1 - t ), r * ( 1 - t ), -r * s ),
                                                  Gradient( -s * ( 1 - t ), ( 1 - r ) * ( 1 - t ), -( 1 - r ) * s ),
                                                  Gradient( -( 1 - s ) * t, -( 1 - r ) * t, ( 1 - r ) * ( 1 - s ) ),
                                                  Gradient( ( 1 - s ) * t, -r * t, r * ( 1 - s ) ),
                                                  Gradient( s * t, r * t, r * s ),
                                                  Gradient( -s * t, ( 1 - r ) * t, ( 1 - r ) * s ) };
        for( std::size_t i = 0; i < points.size(); ++i )
        {
            const Gradient expected = inverse_transpose * reference[i];
            for( unsigned d = 0; d < dim; ++d )
                close( fe.get_dphi()[i][static_cast< std::size_t >( q )]( d ), expected( d ),
                       "individual affine tensor gradient component" );
        }
    }
}

void test_volume( moab::EntityType type, unsigned dim, const std::vector< KARMA::Point >& points, KARMA::Real measure,
                  KARMA::Real integral_x2, KARMA::Real integral_xy, bool check_affine_gradients = false )
{
    using namespace KARMA;
    KARMAMesh mesh( dim, PETSC_COMM_WORLD );
    const auto element = add_element( mesh, type, points );
    const auto variant = linear_variant( type );
    KARMAFEBase fe( mesh, variant, p1_basis( variant ), 2 );

    const auto& jxw       = fe.get_JxW();
    const auto& phi       = fe.get_phi();
    const auto& dphi      = fe.get_dphi();
    const auto& xyz       = fe.get_xyz();
    const auto& normals   = fe.get_normals();
    const Real* jxw_data  = jxw.data();
    const Point* xyz_data = xyz.data();
    check( !fe.is_valid(), "volume FE starts invalid before first reinit" );
    for( const Real weight : jxw )
        check( !std::isfinite( weight ), "initial volume JxW is poisoned" );
    for( const auto& row : phi )
        for( const Real value : row )
            check( !std::isfinite( value ), "initial volume phi is poisoned" );
    for( const auto& row : dphi )
        for( const Gradient& value : row )
            check( !value.allFinite(), "initial volume dphi is poisoned" );
    for( const Point& point : xyz )
        check( !point.allFinite(), "initial volume xyz is poisoned" );
    for( const Vector3& normal : normals )
        check( !normal.allFinite(), "initial volume normals are poisoned" );
    std::vector< const Real* > phi_rows;
    std::vector< const Gradient* > dphi_rows;
    for( std::size_t i = 0; i < phi.size(); ++i )
    {
        phi_rows.push_back( phi[i].data() );
        dphi_rows.push_back( dphi[i].data() );
    }

    fe.reinit( element );
    check( fe.is_valid(), "successful volume reinit marks state valid" );
    check( &jxw == &fe.get_JxW(), "JxW outer reference remains stable" );
    check( &phi == &fe.get_phi(), "phi outer reference remains stable" );
    check( &dphi == &fe.get_dphi(), "dphi outer reference remains stable" );
    check( &xyz == &fe.get_xyz(), "xyz outer reference remains stable" );
    for( std::size_t i = 0; i < phi.size(); ++i )
    {
        check( phi_rows[i] == fe.get_phi()[i].data(), "every phi row remains stable" );
        check( dphi_rows[i] == fe.get_dphi()[i].data(), "every dphi row remains stable" );
    }

    const auto first_jxw  = jxw;
    const auto first_dphi = dphi;
    const auto first_xyz  = xyz;
    for( std::size_t q = 0; q < jxw.size(); ++q )
    {
        const Point reconstructed = [&] {
            Point value = Point::Zero();
            for( std::size_t i = 0; i < phi.size(); ++i )
                value += phi[i][q] * points[i];
            return value;
        }();
        close( ( reconstructed - xyz[q] ).norm(), 0.0, "direct nodal basis values reproduce physical point" );
    }

    Real integrated_measure = 0.0;
    Real integrated_x2      = 0.0;
    Real integrated_xy      = 0.0;
    for( std::size_t q = 0; q < jxw.size(); ++q )
    {
        Real phi_sum          = 0.0;
        Gradient gradient_sum = Gradient::Zero();
        for( std::size_t i = 0; i < phi.size(); ++i )
        {
            phi_sum += phi[i][q];
            gradient_sum += dphi[i][q];
        }
        close( phi_sum, 1.0, "partition of unity" );
        close( gradient_sum.norm(), 0.0, "zero gradient sum" );
        for( unsigned component = 0; component < dim; ++component )
        {
            Gradient coordinate_gradient = Gradient::Zero();
            for( std::size_t i = 0; i < phi.size(); ++i )
                coordinate_gradient += points[i]( component ) * dphi[i][q];
            for( unsigned direction = 0; direction < dim; ++direction )
                close( coordinate_gradient( direction ), component == direction ? 1.0 : 0.0,
                       "individual dphi obeys physical coordinate chain rule" );
        }
        check( mesh.bounding_box().contains( xyz[q], 1.e-12 ), "physical quadrature point lies in element bounds" );
        integrated_measure += jxw[q];
        integrated_x2 += jxw[q] * xyz[q]( 0 ) * xyz[q]( 0 );
        integrated_xy += jxw[q] * xyz[q]( 0 ) * xyz[q]( 1 );
    }
    close( integrated_measure, measure, "analytic element measure" );
    close( integrated_x2, integral_x2, "degree-two x^2 integral" );
    close( integrated_xy, integral_xy, "mixed xy integral" );
    if( check_affine_gradients ) check_affine_tensor_gradients( type, points, fe );

    std::vector< Real > shifted_coordinates( 3 * points.size() );
    for( std::size_t i = 0; i < points.size(); ++i )
    {
        shifted_coordinates[3 * i]     = 1.5 * points[i]( 0 ) + 0.25;
        shifted_coordinates[3 * i + 1] = points[i]( 1 );
        shifted_coordinates[3 * i + 2] = points[i]( 2 );
    }
    moab::Range shifted_vertices;
    check( mesh.moab()->create_vertices( shifted_coordinates.data(), static_cast< int >( points.size() ),
                                         shifted_vertices ) == moab::MB_SUCCESS,
           "create second FE test element vertices" );
    std::vector< moab::EntityHandle > shifted_conn( shifted_vertices.begin(), shifted_vertices.end() );
    moab::EntityHandle shifted_element = 0;
    check( mesh.moab()->create_element( type, shifted_conn.data(), static_cast< int >( shifted_conn.size() ),
                                        shifted_element ) == moab::MB_SUCCESS,
           "create second FE test element" );
    fe.reinit( shifted_element );
    check( jxw_data == jxw.data() && xyz_data == xyz.data(), "outer data storage survives a distinct element reinit" );
    check( ( first_xyz.front() - xyz.front() ).norm() > 0.0, "physical values change on a distinct element reinit" );
    bool gradient_changed = false;
    for( std::size_t i = 0; i < dphi.size(); ++i )
        for( std::size_t q = 0; q < dphi[i].size(); ++q )
            gradient_changed = gradient_changed || ( first_dphi[i][q] - dphi[i][q] ).norm() > 1.e-12;
    check( gradient_changed, "gradients change on a scaled element reinit" );
    check( first_jxw != jxw, "weights change on a scaled element reinit" );
    for( std::size_t i = 0; i < phi.size(); ++i )
    {
        check( phi_rows[i] == fe.get_phi()[i].data(), "phi rows survive multiple reinits" );
        check( dphi_rows[i] == fe.get_dphi()[i].data(), "dphi rows survive multiple reinits" );
    }
    check( throws( [&] { fe.reinit( 0 ); } ), "invalid element is rejected" );
    check( !fe.is_valid(), "failed volume reinit marks state invalid" );
    for( const Real weight : jxw )
        check( !std::isfinite( weight ), "failed volume reinit poisons JxW" );
    for( const auto& row : phi )
        for( const Real value : row )
            check( !std::isfinite( value ), "failed volume reinit poisons phi" );
    for( const auto& row : dphi )
        for( const Gradient& value : row )
            check( !value.allFinite(), "failed volume reinit poisons dphi" );
    for( const Point& point : xyz )
        check( !point.allFinite(), "failed volume reinit poisons xyz" );
    for( const Vector3& normal : normals )
        check( !normal.allFinite(), "failed volume reinit poisons normals" );
    fe.reinit( element );
    check( fe.is_valid(), "successful volume reinit restores validity after failure" );
}
}  // namespace

int main( int argc, char** argv )
{
    PetscErrorCode ierr = PetscInitialize( &argc, &argv, nullptr, nullptr );
    if( ierr ) return ierr;

    using namespace KARMA;
    int rank = 0;
    int communicator_size = 1;
    check( MPI_Comm_rank( PETSC_COMM_WORLD, &rank ) == MPI_SUCCESS, "query replicated-SPMD FE rank" );
    check( MPI_Comm_size( PETSC_COMM_WORLD, &communicator_size ) == MPI_SUCCESS,
           "query replicated-SPMD FE size" );
    check( rank >= 0 && rank < communicator_size, "replicated-SPMD FE numerical checks execute on this rank" );

    // Sections let CTest schedule this file's independent blocks in parallel
    // instead of running them as one long serial test. The blocks below share
    // no state - each builds its own meshes, bases and elements; the only
    // things crossing between them are `rank`/`size` above and the global
    // `failures` counter, which is per-process either way. With no argument
    // every section runs, so an interactive invocation is unchanged.
    const std::string karma_section = argc > 1 ? argv[1] : "--all";
    bool karma_section_recognized = ( karma_section == "--all" );
    const auto selected = [&]( const char* name ) {
        if( karma_section == "--all" ) return true;
        if( karma_section == name )
        {
            karma_section_recognized = true;
            return true;
        }
        return false;
    };


    if( selected( "--quadrature" ) )
    {
    for( const auto [type, dim] :
         { std::pair{ moab::MBEDGE, 1u }, std::pair{ moab::MBTRI, 2u }, std::pair{ moab::MBQUAD, 2u },
           std::pair{ moab::MBTET, 3u }, std::pair{ moab::MBHEX, 3u } } )
    {
        KARMAQuadrature quadrature( type, 2 );
        check( quadrature.dimension() == dim, "quadrature dimension follows topology" );
        check( quadrature.degree() == 2, "quadrature retains integer degree" );
        check( quadrature.topology() == type, "quadrature retains topology" );
        check( quadrature.n_points() > 0, "quadrature has points" );
        check( quadrature.petsc_quadrature() != nullptr, "quadrature owns PETSc object" );
        PetscInt qdim           = 0;
        PetscInt count          = 0;
        const PetscReal* points = nullptr;
        check( PetscQuadratureGetData( quadrature.petsc_quadrature(), &qdim, nullptr, &count, &points, nullptr ) ==
                   PETSC_SUCCESS,
               "query native quadrature data" );
        DMPolytopeType cell_type = DM_POLYTOPE_UNKNOWN;
        check( PetscQuadratureGetCellType( quadrature.petsc_quadrature(), &cell_type ) == PETSC_SUCCESS,
               "query native quadrature topology" );
        const DMPolytopeType expected_cell = type == moab::MBEDGE   ? DM_POLYTOPE_SEGMENT
                                             : type == moab::MBTRI  ? DM_POLYTOPE_TRIANGLE
                                             : type == moab::MBQUAD ? DM_POLYTOPE_QUADRILATERAL
                                             : type == moab::MBTET  ? DM_POLYTOPE_TETRAHEDRON
                                                                    : DM_POLYTOPE_HEXAHEDRON;
        check( cell_type == expected_cell, "native quadrature retains topology" );
        for( PetscInt i = 0; i < qdim * count; ++i )
            check( points[i] >= 0.0 && points[i] <= 1.0, "quadrature points use DMMoab natural coordinates" );
        if( type == moab::MBTRI )
        {
            const PetscReal* weights = nullptr;
            check( PetscQuadratureGetData( quadrature.petsc_quadrature(), nullptr, nullptr, nullptr, nullptr,
                                           &weights ) == PETSC_SUCCESS,
                   "query triangle quadrature weights" );
            Real total_weight = 0.0;
            for( PetscInt q = 0; q < count; ++q )
                total_weight += weights[q];
            close( total_weight, 0.5, "triangle rule integrates unit-simplex measure" );
        }
    }

    for( const auto type : { moab::MBPRISM, moab::MBPYRAMID } )
    for( int requested_degree = 0; requested_degree <= 6; ++requested_degree )
    {
        KARMAQuadrature quadrature( type, requested_degree );
        PetscInt dimension = 0;
        PetscInt count = 0;
        const PetscReal* points = nullptr;
        const PetscReal* weights = nullptr;
        check( PetscQuadratureGetData( quadrature.petsc_quadrature(), &dimension, nullptr, &count, &points,
                                       &weights ) == PETSC_SUCCESS,
               "query product/collapsed quadrature" );
        Real constant = 0.0;
        Real x2 = 0.0;
        for( PetscInt q = 0; q < count; ++q )
        {
            constant += weights[q];
            x2 += weights[q] * points[q * dimension] * points[q * dimension];
            if( type == moab::MBPYRAMID )
                check( points[q * dimension + 2] < 1.0, "pyramid quadrature avoids the apex" );
        }
        close( constant, type == moab::MBPRISM ? 0.5 : 4.0 / 3.0,
               "product/collapsed quadrature integrates reference measure" );
        if( requested_degree >= 2 )
            close( x2, type == moab::MBPRISM ? 1.0 / 12.0 : 4.0 / 15.0,
                   "product/collapsed quadrature integrates x squared" );
        const auto factorial = []( unsigned n ) {
            Real value = 1.0;
            for( unsigned i = 2; i <= n; ++i ) value *= i;
            return value;
        };
        for( unsigned a = 0; a <= static_cast< unsigned >( requested_degree ); ++a )
            for( unsigned b = 0; a + b <= static_cast< unsigned >( requested_degree ); ++b )
                for( unsigned c = 0; a + b + c <= static_cast< unsigned >( requested_degree ); ++c )
                {
                    Real integral = 0.0;
                    for( PetscInt q = 0; q < count; ++q )
                        integral += weights[q] * std::pow( points[q * dimension], static_cast< int >( a ) ) *
                                    std::pow( points[q * dimension + 1], static_cast< int >( b ) ) *
                                    std::pow( points[q * dimension + 2], static_cast< int >( c ) );
                    const Real expected = type == moab::MBPRISM
                        ? factorial( a ) * factorial( b ) /
                              ( factorial( a + b + 2 ) * static_cast< Real >( c + 1 ) )
                        : ( a % 2 || b % 2 ) ? 0.0
                        : 4.0 * factorial( c ) * factorial( a + b + 2 ) /
                              ( static_cast< Real >( a + 1 ) * static_cast< Real >( b + 1 ) *
                                factorial( a + b + c + 3 ) );
                    close( integral, expected, "requested-degree product/collapsed monomial exactness", 5.e-12 );
                }
    }
    for( const auto parent_variant : { GeometryVariant::Prism6, GeometryVariant::Pyramid5 } )
        for( int degree = 0; degree <= 7; ++degree )
        {
            const KARMAReferenceTopology reference( parent_variant );
            KARMAMesh mesh( 3, PETSC_COMM_WORLD );
            const auto element = add_element( mesh, reference.topology(), reference.reference_nodes() );
            KARMAFEBase face( mesh, parent_variant, p1_basis( parent_variant ), degree,
                              FEIntegrationDomain::Face );
            const unsigned triangular_side = parent_variant == GeometryVariant::Prism6 ? 3u : 0u;
            face.reinit( element, triangular_side );
            for( unsigned a = 0; a <= static_cast< unsigned >( degree ); ++a )
                for( unsigned b = 0; a + b <= static_cast< unsigned >( degree ); ++b )
                {
                    Real integral = 0.0;
                    const KARMAGeometryMap geometry( reference, reference.reference_nodes() );
                    for( std::size_t q = 0; q < face.get_JxW().size(); ++q )
                    {
                        const Point natural = geometry.inverse_map( face.get_xyz()[q] );
                        const Matrix3 side_jacobian = reference.side_reference_jacobian( triangular_side,
                            Point( natural( 0 ), natural( 1 ), 0 ) );
                        const Point side_origin = reference.side_reference_map( triangular_side, Point::Zero() );
                        const Eigen::Vector2d local =
                            side_jacobian.leftCols< 2 >().colPivHouseholderQr().solve( natural - side_origin ).eval();
                        const Real physical_scale = geometry.side( triangular_side,
                            Point( local( 0 ), local( 1 ), 0 ) ).measure;
                        integral += face.get_JxW()[q] / physical_scale *
                                    std::pow( local( 0 ), static_cast< int >( a ) ) *
                                    std::pow( local( 1 ), static_cast< int >( b ) );
                    }
                    Real numerator = 1.0;
                    for( unsigned i = 2; i <= a; ++i ) numerator *= i;
                    for( unsigned i = 2; i <= b; ++i ) numerator *= i;
                    Real denominator = 1.0;
                    for( unsigned i = 2; i <= a + b + 2; ++i ) denominator *= i;
                    close( integral, numerator / denominator,
                           "mixed-cell triangular face requested-degree exactness", 5.e-12 );
                }
        }
    check( throws( [] { KARMAQuadrature q( moab::MBTRI, -1 ); } ), "negative degree is rejected" );
    check( throws( [] { KARMAQuadrature q( moab::MBHEX, INT_MAX ); } ),
           "unreasonably large quadrature degree is rejected before arithmetic" );
    }
    if( selected( "--pyramid-gradients" ) )
    {
        // Does the reduced quadrature resolve GRADIENT terms on pyramids?
        //
        // The monomial exactness check above cannot answer that. Under the
        // pyramid's collapsed map the chain rule introduces 1/h factors that
        // no polynomial argument covers, and the concern is that the point
        // per axis removed from this rule was margin for exactly them.
        //
        // The divergence theorem answers it without needing a closed form for
        // the rational basis. For any basis function,
        //
        //     int_Omega grad N_i dV = int_dOmega N_i n dS
        //
        // is an identity of calculus, so it holds whatever the basis and
        // whatever the continuity. The two sides are computed by different
        // machinery here: the left by the volume rule against the rational
        // gradients, where the 1/h factors live, and the right by the face
        // rules against basis *traces*, which pyramid bases are constructed
        // to keep polynomial precisely so they conform to neighbouring
        // tetrahedra and hexahedra. The right side is therefore effectively
        // exact, and the gap between them measures the one thing in question.
        //
        // The prism is included as a control. It has no collapsed map and so
        // no 1/h factors at all, so if a pyramid column ever looks bad, the
        // prism column distinguishes "the collapsed map is under-resolved"
        // from "something common to both topologies regressed".
        //
        // The sweep over quadrature degree is the diagnostic: if the gap at
        // the production degree already sits at rounding, the reduced rule
        // resolves gradients and the removed point was not margin. If instead
        // it fell steadily with degree, it was.
        std::cout << "pyramid/prism gradient consistency"
                     "  (max_i |int grad N_i dV - int N_i n dS| / surface area)\n";
        for( const auto variant : { GeometryVariant::Prism6, GeometryVariant::Pyramid5,
                                    GeometryVariant::Pyramid13, GeometryVariant::Pyramid14 } )
        {
            const KARMAReferenceTopology reference( variant );
            for( int degree = 2; degree <= 8; degree += 2 )
            {
                KARMAMesh mesh( 3, PETSC_COMM_WORLD );
                const auto element = add_element( mesh, reference.topology(), reference.reference_nodes() );
                KARMAFEBase volume( mesh, variant, p1_basis( variant ), degree );
                KARMAFEBase face( mesh, variant, p1_basis( variant ), degree, FEIntegrationDomain::Face );
                volume.reinit( element );

                const std::size_t basis_count = volume.get_phi().size();
                std::vector< Vector3 > from_volume( basis_count, Vector3::Zero() );
                for( std::size_t i = 0; i < basis_count; ++i )
                    for( std::size_t q = 0; q < volume.get_JxW().size(); ++q )
                        from_volume[i] += volume.get_dphi()[i][q] * volume.get_JxW()[q];

                std::vector< Vector3 > from_surface( basis_count, Vector3::Zero() );
                Real surface_area = 0.0;
                for( unsigned side = 0; side < reference.side_count(); ++side )
                {
                    face.reinit( element, side );
                    for( std::size_t q = 0; q < face.get_JxW().size(); ++q )
                    {
                        surface_area += face.get_JxW()[q];
                        for( std::size_t i = 0; i < basis_count; ++i )
                            from_surface[i] += face.get_phi()[i][q] * face.get_normals()[q] * face.get_JxW()[q];
                    }
                }
                check( surface_area > 0.0, "the element boundary has positive area" );

                Real worst = 0.0;
                for( std::size_t i = 0; i < basis_count; ++i )
                    worst = std::max( worst, ( from_volume[i] - from_surface[i] ).norm() );
                worst /= surface_area;

                std::cout << "  " << variant_name( variant ) << " degree " << degree
                          << " (" << volume.get_JxW().size() << " points): " << worst << '\n';
                // Loose on purpose. A rule that under-resolved the gradients
                // would miss by orders of magnitude, not by a few ulp, and a
                // threshold tightened to today's measurement would then fail
                // for distorted elements without telling anyone anything new.
                close( worst, 0.0, "gradient integral matches the surface integral on " +
                                       std::string( variant_name( variant ) ), 1.e-10 );
            }
        }
    }

    if( selected( "--quadrature-exactness" ) )
    {
        // Pins the property the pyramid and prism rules were re-sized on.
        //
        // The pyramid rule integrates over a collapsed map whose h^2 Jacobian
        // now lives in a Gauss-Jacobi weight rather than in the integrand. If
        // that Jacobian is ever applied twice, or the collapsed axis reverts
        // to Gauss-Legendre at the reduced count, the rule silently stops
        // being exact - silently, because partition-of-unity and positive-JxW
        // checks elsewhere still pass. This integrates the monomial basis
        // directly against the closed form and so fails loudly instead.
        //
        // Reference pyramid: apex (0,0,1), base [-1,1]^2 at z=0, volume 4/3.
        const auto exact_pyramid = []( unsigned p, unsigned q, unsigned r ) -> Real {
            if( p % 2 || q % 2 ) return 0.0;
            return ( 2.0 / ( p + 1 ) ) * ( 2.0 / ( q + 1 ) ) * std::tgamma( r + 1.0 ) *
                   std::tgamma( p + q + 3.0 ) / std::tgamma( r + p + q + 4.0 );
        };

        for( int degree = 1; degree <= 6; ++degree )
        {
            KARMAQuadrature rule( moab::MBPYRAMID, degree );
            PetscInt count = 0;
            const PetscReal* qpoints = nullptr;
            const PetscReal* qweights = nullptr;
            check( PetscQuadratureGetData( rule.petsc_quadrature(), nullptr, nullptr, &count, &qpoints,
                                           &qweights ) == PETSC_SUCCESS,
                   "query pyramid quadrature for the exactness check" );

            // The reduced rule must also be the one actually in use: a
            // regression to (degree+4)/2 would still be exact, just slower,
            // and would pass every assertion below.
            const PetscInt expected_per_axis = std::max( 1, ( degree + 2 ) / 2 );
            check( count == expected_per_axis * expected_per_axis * expected_per_axis,
                   "pyramid quadrature uses the reduced Gauss-Jacobi point count" );

            for( unsigned p = 0; p <= static_cast< unsigned >( degree ); ++p )
                for( unsigned q = 0; p + q <= static_cast< unsigned >( degree ); ++q )
                    for( unsigned r = 0; p + q + r <= static_cast< unsigned >( degree ); ++r )
                    {
                        Real integral = 0.0;
                        for( PetscInt s = 0; s < count; ++s )
                            integral += qweights[s] * std::pow( qpoints[3 * s], p ) *
                                        std::pow( qpoints[3 * s + 1], q ) * std::pow( qpoints[3 * s + 2], r );
                        // Normalized by the cell volume so a monomial whose
                        // exact integral is zero is compared on an absolute
                        // scale rather than a relative one.
                        close( ( integral - exact_pyramid( p, q, r ) ) / ( 4.0 / 3.0 ), 0.0,
                               "pyramid quadrature integrates the monomial basis exactly", 1.e-13 );
                    }
        }

        // The prism shares the sizing but is a plain triangle-by-line product
        // with no Jacobian weight, so it only needs the standard line count.
        for( int degree = 1; degree <= 6; ++degree )
        {
            KARMAQuadrature prism( moab::MBPRISM, degree );
            KARMAQuadrature hex( moab::MBHEX, degree );
            check( prism.n_points() > 0 && hex.n_points() > 0, "prism and hex quadratures are non-empty" );
            Real prism_volume = 0.0;
            PetscInt prism_count = 0;
            const PetscReal* prism_weights = nullptr;
            check( PetscQuadratureGetData( prism.petsc_quadrature(), nullptr, nullptr, &prism_count, nullptr,
                                           &prism_weights ) == PETSC_SUCCESS,
                   "query prism quadrature for the exactness check" );
            for( PetscInt s = 0; s < prism_count; ++s ) prism_volume += prism_weights[s];
            // Reference prism: unit triangle cross [0,1], volume 1/2.
            close( prism_volume, 0.5, "prism quadrature weights sum to the reference volume", 1.e-13 );
        }
    }

    if( selected( "--allocation" ) )
    {
    {
        KARMAQuadrature warmup( moab::MBPYRAMID, 7 );
        PetscLogDouble before = 0.0;
        PetscLogDouble after = 0.0;
        check( PetscMallocGetCurrentUsage( &before ) == PETSC_SUCCESS,
               "query PETSc bytes before quadrature ownership stress" );
        for( int repetition = 0; repetition < 32; ++repetition )
        {
            KARMAQuadrature point( moab::MBVERTEX, 0 );
            KARMAQuadrature simplex( moab::MBTET, 5 );
            KARMAQuadrature product( moab::MBPRISM, 7 );
            KARMAQuadrature collapsed( moab::MBPYRAMID, 7 );
        }
        check( PetscMallocGetCurrentUsage( &after ) == PETSC_SUCCESS,
               "query PETSc bytes after quadrature ownership stress" );
        close( after, before, "quadrature construction/destruction retains no PETSc bytes", 0.0 );
    }
    }

    // Its own section rather than a tail of --allocation: this checks that a
    // 1D element embeds in a 1D mesh, which has nothing to do with allocation
    // accounting, and running `ctest -R karma_fe_allocation` should not
    // silently also run something its name does not describe.
    if( selected( "--embedded-1d" ) )
    {
        KARMAMesh edge_mesh( 1, PETSC_COMM_WORLD );
        check( !throws( [&] { KARMAFEBase fe( edge_mesh, GeometryVariant::Edge2,
                                                p1_basis( GeometryVariant::Edge2 ), 2 ); } ),
                "embedded one-dimensional finite elements are supported" );
    }

    if( selected( "--degenerate" ) )
    {
    {
        KARMAMesh tiny( 2, PETSC_COMM_WORLD );
        const auto tiny_element =
            add_element( tiny, moab::MBTRI, { Point( 0, 0, 0 ), Point( 1.e-10, 0, 0 ), Point( 0, 1.e-10, 0 ) } );
        KARMAFEBase tiny_fe( tiny, GeometryVariant::Tri3, p1_basis( GeometryVariant::Tri3 ), 2 );
        check( !throws( [&] { tiny_fe.reinit( tiny_element ); } ), "tiny well-conditioned triangle remains valid" );
        Real tiny_area = 0.0;
        for( const Real weight : tiny_fe.get_JxW() )
            tiny_area += weight;
        close( tiny_area, 0.5e-20, "tiny triangle area", 1.e-30 );

        KARMAMesh degenerate( 2, PETSC_COMM_WORLD );
        const auto degenerate_element =
            add_element( degenerate, moab::MBTRI, { Point( 0, 0, 0 ), Point( 1, 0, 0 ), Point( 2, 0, 0 ) } );
        KARMAFEBase degenerate_fe( degenerate, GeometryVariant::Tri3, p1_basis( GeometryVariant::Tri3 ), 2 );
        check( throws( [&] { degenerate_fe.reinit( degenerate_element ); } ), "degenerate triangle is rejected" );

        KARMAMesh ill_conditioned( 2, PETSC_COMM_WORLD );
        const auto ill_element =
            add_element( ill_conditioned, moab::MBTRI, { Point( 0, 0, 0 ), Point( 1, 0, 0 ), Point( 1, 1.e-16, 0 ) } );
        KARMAFEBase ill_fe( ill_conditioned, GeometryVariant::Tri3, p1_basis( GeometryVariant::Tri3 ), 2 );
        check( throws( [&] { ill_fe.reinit( ill_element ); } ), "ill-conditioned triangle is rejected" );

        KARMAMesh inverted( 2, PETSC_COMM_WORLD );
        const auto inverted_element =
            add_element( inverted, moab::MBTRI, { Point( 0, 0, 0 ), Point( 0, 1, 0 ), Point( 1, 0, 0 ) } );
        KARMAFEBase inverted_fe( inverted, GeometryVariant::Tri3, p1_basis( GeometryVariant::Tri3 ), 2 );
        check( !throws( [&] { inverted_fe.reinit( inverted_element ); } ),
               "embedded surface orientation is independent of global XY orientation" );

        KARMAMesh folded_quad( 2, PETSC_COMM_WORLD );
        const auto folded_quad_element =
            add_element( folded_quad, moab::MBQUAD,
                         { Point( 0, 0, 0 ), Point( 1, 0, 0 ), Point( 0.45, 0.45, 0 ), Point( 0, 1, 0 ) } );
        KARMAFEBase folded_quad_fe( folded_quad, GeometryVariant::Quad4, p1_basis( GeometryVariant::Quad4 ), 2 );
        check( throws( [&] { folded_quad_fe.reinit( folded_quad_element ); } ),
               "folded quad is rejected from corner Jacobians" );

        KARMAMesh folded_hex( 3, PETSC_COMM_WORLD );
        const auto folded_hex_element =
            add_element( folded_hex, moab::MBHEX,
                         { Point( 0, 0, 0 ), Point( 1, 0, 0 ), Point( 1, 1, 0 ), Point( 0, 1, 0 ), Point( 0, 0, 1 ),
                           Point( 1, 0, 1 ), Point( 0.4, 0.4, 0.4 ), Point( 0, 1, 1 ) } );
        KARMAFEBase folded_hex_fe( folded_hex, GeometryVariant::Hex8, p1_basis( GeometryVariant::Hex8 ), 2 );
        check( throws( [&] { folded_hex_fe.reinit( folded_hex_element ); } ),
               "folded hex is rejected from corner Jacobians" );

        KARMAMesh warped_hex( 3, PETSC_COMM_WORLD );
        const auto warped_hex_element =
            add_element( warped_hex, moab::MBHEX,
                         { Point( 0, 0, 0 ), Point( 1, 0, 0 ), Point( 1, 1, 0 ), Point( 0, 1, 0 ), Point( 0, 0, 1 ),
                           Point( 1, 0, 1 ), Point( 1.1, 1, 1 ), Point( 0, 1, 1 ) } );
        KARMAFEBase warped_hex_fe( warped_hex, GeometryVariant::Hex8, p1_basis( GeometryVariant::Hex8 ), 2 );
        check( !throws( [&] { warped_hex_fe.reinit( warped_hex_element ); } ),
               "valid warped trilinear hex is supported" );


        KARMAMesh nonplanar( 2, PETSC_COMM_WORLD );
        const auto nonplanar_element =
            add_element( nonplanar, moab::MBTRI, { Point( 0, 0, 0 ), Point( 1, 0, 0 ), Point( 0, 1, 1.e-4 ) } );
        KARMAFEBase nonplanar_fe( nonplanar, GeometryVariant::Tri3, p1_basis( GeometryVariant::Tri3 ), 2 );
        check( !throws( [&] { nonplanar_fe.reinit( nonplanar_element ); } ),
               "embedded non-axis-aligned surface is supported" );


        KARMAMesh translated_plane( 2, PETSC_COMM_WORLD );
        const auto translated_element =
            add_element( translated_plane, moab::MBTRI, { Point( 0, 0, 2 ), Point( 1, 0, 2 ), Point( 0, 1, 2 ) } );
        KARMAFEBase translated_fe( translated_plane, GeometryVariant::Tri3, p1_basis( GeometryVariant::Tri3 ), 2 );
        check( !throws( [&] { translated_fe.reinit( translated_element ); } ),
               "embedded translated surface is supported" );

    }
    }

    if( selected( "--basis" ) )
    {
    {
        KARMAMesh mesh( 2, PETSC_COMM_WORLD );
        const auto element = add_element( mesh, moab::MBTRI,
                                          { Point( 0, 0, 0 ), Point( 1, 0, 0 ), Point( 0, 1, 0 ) } );
        const BasisKey key{ ReferenceCell::Triangle, FieldBasisFamily::LagrangeNodal, 0,
                            BasisContinuity::dG };
        const KARMABasis basis = KARMABasisRegistry().create( key );
        KARMAFEBase fe( mesh, GeometryVariant::Tri3, basis, 2 );
        fe.reinit( element );
        check( fe.get_phi().size() == 1, "P0 field size is independent of Tri3 geometry nodes" );
        for( std::size_t q = 0; q < fe.get_JxW().size(); ++q )
        {
            close( fe.get_phi()[0][q], 1.0, "P0 value on linear geometry" );
            close( fe.get_dphi()[0][q].norm(), 0.0, "P0 gradient on linear geometry" );
        }

        const BasisKey p2_key{ ReferenceCell::Triangle, FieldBasisFamily::LagrangeNodal, 2,
                               BasisContinuity::dG };
        KARMAFEBase p2( mesh, GeometryVariant::Tri3, KARMABasisRegistry().create( p2_key ), 4 );
        p2.reinit( element );
        check( p2.get_phi().size() == 6, "P2 field has six rows on Tri3 geometry" );
        for( std::size_t q = 0; q < p2.get_JxW().size(); ++q )
        {
            Real sum = 0.0;
            Gradient gradient_sum = Gradient::Zero();
            for( std::size_t i = 0; i < p2.get_phi().size(); ++i )
            {
                sum += p2.get_phi()[i][q];
                gradient_sum += p2.get_dphi()[i][q];
            }
            close( sum, 1.0, "P2 field partitions unity on linear geometry" );
            close( gradient_sum.norm(), 0.0, "P2 physical gradients sum to zero" );
        }

        const auto tri6_nodes = KARMAReferenceTopology( GeometryVariant::Tri6 ).reference_nodes();
        const auto tri6 = add_element( mesh, moab::MBTRI, tri6_nodes );
        KARMAFEBase curved_p0( mesh, GeometryVariant::Tri6, basis, 4 );
        curved_p0.reinit( tri6 );
        check( curved_p0.get_phi().size() == 1, "P0 field size is independent of Tri6 geometry nodes" );
        check( throws( [&] { curved_p0.reinit( element ); } ), "bound geometry variant rejects a node-count change" );
        check( !curved_p0.is_valid(), "geometry variant mismatch poisons FE state" );

        KARMAMesh surface_mesh( 3, PETSC_COMM_WORLD );
        const std::vector< Point > surface_nodes{ Point( 0, 0, 0 ), Point( 2, 0, 1 ), Point( 0, 3, 1 ) };
        const auto surface = add_element( surface_mesh, moab::MBTRI, surface_nodes );
        const BasisKey surface_key{ ReferenceCell::Triangle, FieldBasisFamily::LagrangeNodal, 1,
                                    BasisContinuity::cG };
        KARMAFEBase embedded_surface( surface_mesh, GeometryVariant::Tri3,
                                     KARMABasisRegistry().create( surface_key ), 2 );
        embedded_surface.reinit( surface );
        for( std::size_t q = 0; q < embedded_surface.get_JxW().size(); ++q )
        {
            for( unsigned coordinate = 0; coordinate < 3; ++coordinate )
            {
                Gradient reconstructed = Gradient::Zero();
                for( std::size_t i = 0; i < surface_nodes.size(); ++i )
                    reconstructed += surface_nodes[i]( coordinate ) * embedded_surface.get_dphi()[i][q];
                const Vector3 tangent0 = surface_nodes[1] - surface_nodes[0];
                const Vector3 tangent1 = surface_nodes[2] - surface_nodes[0];
                Matrix3 projector = Matrix3::Zero();
                const Matrix3 jacobian = ( Matrix3() << tangent0, tangent1, Vector3::Zero() ).finished();
                const auto metric = jacobian.topLeftCorner< 3, 2 >().transpose() * jacobian.topLeftCorner< 3, 2 >();
                projector = jacobian * ( Matrix3() << metric.inverse()( 0, 0 ), metric.inverse()( 0, 1 ), 0,
                                                     metric.inverse()( 1, 0 ), metric.inverse()( 1, 1 ), 0,
                                                     0, 0, 0 ).finished() * jacobian.transpose();
                close( ( reconstructed - projector.col( coordinate ) ).norm(), 0.0,
                       "embedded surface coordinate gradient is the tangential projector", 2.e-10 );
            }
        }

        const BasisKey edge_key{ ReferenceCell::Edge, FieldBasisFamily::LagrangeNodal, 1,
                                 BasisContinuity::cG };
        KARMAMesh edge_mesh( 3, PETSC_COMM_WORLD );
        const auto edge = add_element( edge_mesh, moab::MBEDGE,
                                       { Point( 0, 0, 0 ), Point( 1, 1, 1 ) } );
        KARMAFEBase embedded_edge( edge_mesh, GeometryVariant::Edge2,
                                  KARMABasisRegistry().create( edge_key ), 2 );
        embedded_edge.reinit( edge );
        allocation_count.store( 0, std::memory_order_relaxed );
        count_allocations.store( true, std::memory_order_relaxed );
        embedded_edge.reinit( edge );
        count_allocations.store( false, std::memory_order_relaxed );
        const auto measured_allocations = allocation_count.load( std::memory_order_relaxed );
        if( measured_allocations != 0 ) std::cerr << "volume allocations: " << measured_allocations << '\n';
        check( measured_allocations == 0, "warmed volume reinit performs zero instrumented ordinary/aligned new calls" );
        const Gradient tangent_gradient = Vector3::Ones() / 3.0;
        for( std::size_t q = 0; q < embedded_edge.get_JxW().size(); ++q )
        {
            close( ( embedded_edge.get_dphi()[1][q] - tangent_gradient ).norm(), 0.0,
                   "embedded edge uses minimum-norm tangential gradient" );
            close( ( embedded_edge.get_dphi()[0][q] + tangent_gradient ).norm(), 0.0,
                   "embedded edge gradient orientation" );
        }
    }
    }

    {
        // One section per geometry group rather than all 17 variants in one
        // test: this loop was ~104s under MPI and set the floor on the
        // suite's wall time. The groups partition the original list exactly
        // (7 + 4 + 6 = 17) and each variant is independent of the others -
        // every iteration builds its own mesh and element - so the split
        // changes scheduling, not coverage. The per-group count assertion
        // below is what catches a variant accidentally dropped from a group.
        struct VariantGroup
        {
            const char* section;
            std::vector< GeometryVariant > variants;
        };
        const std::vector< VariantGroup > variant_groups{
            { "--variants-line-surface",
              { GeometryVariant::Edge2, GeometryVariant::Edge3, GeometryVariant::Tri3, GeometryVariant::Tri6,
                GeometryVariant::Quad4, GeometryVariant::Quad8, GeometryVariant::Quad9 } },
            { "--variants-tet-hex",
              { GeometryVariant::Tet4, GeometryVariant::Tet10, GeometryVariant::Hex8, GeometryVariant::Hex27 } },
            { "--variants-prism-pyramid",
              { GeometryVariant::Prism6, GeometryVariant::Prism15, GeometryVariant::Prism18,
                GeometryVariant::Pyramid5, GeometryVariant::Pyramid13, GeometryVariant::Pyramid14 } } };

        for( const auto& group : variant_groups )
        {
        if( !selected( group.section ) ) continue;
        const auto& variants = group.variants;
        KARMABasisRegistry registry;
        std::size_t combinations = 0;
        for( const GeometryVariant variant : variants )
        {
            const KARMAReferenceTopology reference( variant );
            KARMAMesh mesh( reference.dimension(), PETSC_COMM_WORLD );
            const auto element = add_element( mesh, reference.topology(), reference.reference_nodes() );
            for( unsigned order = 0; order <= 2; ++order )
                for( const BasisContinuity continuity : { BasisContinuity::cG, BasisContinuity::dG,
                                                          BasisContinuity::SE } )
                {
                    if( order == 0 && continuity == BasisContinuity::cG ) continue;
                    const BasisKey key{ reference_cell( variant ), FieldBasisFamily::LagrangeNodal, order,
                                        continuity };
                    KARMAFEBase fe( mesh, variant, registry.create( key ), 4 );
                    fe.reinit( element );
                    if( order == 2 && continuity == BasisContinuity::dG )
                    {
                        allocation_count.store( 0, std::memory_order_relaxed );
                        count_allocations.store( true, std::memory_order_relaxed );
                        fe.reinit( element );
                        count_allocations.store( false, std::memory_order_relaxed );
                        check( allocation_count.load( std::memory_order_relaxed ) == 0,
                               "each geometry variant volume reinit makes zero instrumented ordinary/aligned new calls" );
                        if( reference.order() == 2 )
                        {
                            KARMAFEBase face( mesh, variant, registry.create( key ), 4, FEIntegrationDomain::Face );
                            for( unsigned side = 0; side < reference.side_count(); ++side )
                            {
                                face.reinit( element, side );
                                const auto points = face.get_xyz();
                                allocation_count.store( 0, std::memory_order_relaxed );
                                count_allocations.store( true, std::memory_order_relaxed );
                                face.reinit( element, side );
                                face.reinit_at_physical_points( element, points );
                                count_allocations.store( false, std::memory_order_relaxed );
                                check( allocation_count.load( std::memory_order_relaxed ) == 0,
                                       "each quadratic geometry face/point reinit makes zero instrumented ordinary/aligned new calls" );
                            }
                        }
                    }
                    ++combinations;
                    for( std::size_t q = 0; q < fe.get_JxW().size(); ++q )
                    {
                        Real sum = 0.0;
                        Gradient gradient_sum = Gradient::Zero();
                        check( std::isfinite( fe.get_JxW()[q] ) && fe.get_JxW()[q] > 0.0,
                               "matrix JxW is finite and positive" );
                        check( fe.get_xyz()[q].allFinite(), "matrix physical point is finite" );
                        for( std::size_t i = 0; i < fe.get_phi().size(); ++i )
                        {
                            sum += fe.get_phi()[i][q];
                            gradient_sum += fe.get_dphi()[i][q];
                            check( std::isfinite( fe.get_phi()[i][q] ) && fe.get_dphi()[i][q].allFinite(),
                                   "matrix basis values and gradients are finite" );
                        }
                        close( sum, 1.0, "matrix partition of unity", 2.e-10 );
                        close( gradient_sum.norm(), 0.0, "matrix gradient sum", 2.e-9 );
                    }
                }
        }
        check( combinations == variants.size() * 8, "all geometry and built-in basis combinations instantiated" );
        }
    }

    if( selected( "--hex27" ) )
    {
    {
        const KARMAReferenceTopology reference( GeometryVariant::Hex27 );
        KARMAMesh mesh( 3, PETSC_COMM_WORLD );
        const auto element = add_element( mesh, moab::MBHEX, reference.reference_nodes() );
        const BasisKey key{ ReferenceCell::Hexahedron, FieldBasisFamily::LagrangeNodal, 2,
                            BasisContinuity::dG };
        const KARMABasis basis = KARMABasisRegistry().create( key );
        KARMAFEBase volume( mesh, GeometryVariant::Hex27, basis, 4 );
        KARMAFEBase face( mesh, GeometryVariant::Hex27, basis, 4, FEIntegrationDomain::Face );
        volume.reinit( element );
        face.reinit( element, 0 );
        const auto physical_points = face.get_xyz();
        const auto* jxw_data = face.get_JxW().data();
        const auto* xyz_data = face.get_xyz().data();
        const auto* normals_data = face.get_normals().data();
        std::vector< const Real* > phi_data;
        std::vector< const Gradient* > dphi_data;
        for( std::size_t i = 0; i < face.get_phi().size(); ++i )
        {
            phi_data.push_back( face.get_phi()[i].data() );
            dphi_data.push_back( face.get_dphi()[i].data() );
        }

        allocation_count.store( 0, std::memory_order_relaxed );
        count_allocations.store( true, std::memory_order_relaxed );
        volume.reinit( element );
        count_allocations.store( false, std::memory_order_relaxed );
        check( allocation_count.load( std::memory_order_relaxed ) == 0,
               "Hex27 volume reinit performs zero instrumented ordinary/aligned new calls" );

        allocation_count.store( 0, std::memory_order_relaxed );
        count_allocations.store( true, std::memory_order_relaxed );
        face.reinit( element, 0 );
        count_allocations.store( false, std::memory_order_relaxed );
        check( allocation_count.load( std::memory_order_relaxed ) == 0,
               "Hex27 face reinit performs zero instrumented ordinary/aligned new calls" );

        allocation_count.store( 0, std::memory_order_relaxed );
        count_allocations.store( true, std::memory_order_relaxed );
        face.reinit_at_physical_points( element, physical_points );
        count_allocations.store( false, std::memory_order_relaxed );
        check( allocation_count.load( std::memory_order_relaxed ) == 0,
               "Hex27 physical-point reinit performs zero instrumented ordinary/aligned new calls" );
        const Point one_point = physical_points.front();
        allocation_count.store( 0, std::memory_order_relaxed );
        count_allocations.store( true, std::memory_order_relaxed );
        face.reinit_at_physical_points( element, std::span< const Point >( &one_point, 1 ) );
        face.reinit( element, 0 );
        count_allocations.store( false, std::memory_order_relaxed );
        check( allocation_count.load( std::memory_order_relaxed ) == 0,
               "Hex27 active-size changes make zero instrumented ordinary/aligned new calls" );
        check( face.get_JxW().data() == jxw_data && face.get_xyz().data() == xyz_data &&
                   face.get_normals().data() == normals_data,
               "active-size changes preserve outer data pointers" );
        for( std::size_t i = 0; i < face.get_phi().size(); ++i )
        {
            check( face.get_phi()[i].data() == phi_data[i], "active-size changes preserve phi row data" );
            check( face.get_dphi()[i].data() == dphi_data[i], "active-size changes preserve dphi row data" );
        }
        face.reinit( element, 0 );
        const auto valid_size = face.n_quadrature_points();
        const auto valid_xyz = face.get_xyz();
        std::vector< Point > too_many( face.get_xyz().capacity() + 1, Point::Zero() );
        check( throws( [&] { face.reinit_at_physical_points( element, too_many ); } ),
                "physical-point count above capacity is rejected" );
        check( face.is_valid() && face.n_quadrature_points() == valid_size,
               "oversize physical-point rejection preserves prior state" );
        check( face.get_xyz() == valid_xyz, "oversize physical-point rejection preserves prior values" );
        check( throws( [&] { face.reinit_at_physical_points( element, {} ); } ),
               "zero physical-point count is rejected" );
        check( face.is_valid() && face.n_quadrature_points() == valid_size && face.get_xyz() == valid_xyz,
               "zero physical-point rejection preserves prior state" );

        const KARMAReferenceTopology pyramid_reference( GeometryVariant::Pyramid14 );
        KARMAMesh pyramid_mesh( 3, PETSC_COMM_WORLD );
        const auto pyramid_element = add_element( pyramid_mesh, moab::MBPYRAMID,
                                                  pyramid_reference.reference_nodes() );
        const BasisKey pyramid_key{ ReferenceCell::Pyramid, FieldBasisFamily::LagrangeNodal, 2,
                                    BasisContinuity::dG };
        const KARMABasis pyramid_basis = KARMABasisRegistry().create( pyramid_key );
        KARMAFEBase pyramid_volume( pyramid_mesh, GeometryVariant::Pyramid14, pyramid_basis, 4 );
        KARMAFEBase pyramid_face( pyramid_mesh, GeometryVariant::Pyramid14, pyramid_basis, 4,
                                 FEIntegrationDomain::Face );
        pyramid_volume.reinit( pyramid_element );
        pyramid_face.reinit( pyramid_element, 4 );
        const auto pyramid_points = pyramid_face.get_xyz();
        const auto allocations_for = [&]( auto&& operation ) {
            allocation_count.store( 0, std::memory_order_relaxed );
            count_allocations.store( true, std::memory_order_relaxed );
            operation();
            count_allocations.store( false, std::memory_order_relaxed );
            return allocation_count.load( std::memory_order_relaxed );
        };
        const auto pyramid_volume_allocations = allocations_for( [&] { pyramid_volume.reinit( pyramid_element ); } );
        const auto pyramid_face_allocations = allocations_for( [&] { pyramid_face.reinit( pyramid_element, 4 ); } );
        const auto pyramid_point_allocations = allocations_for(
            [&] { pyramid_face.reinit_at_physical_points( pyramid_element, pyramid_points ); } );
        if( pyramid_volume_allocations + pyramid_face_allocations + pyramid_point_allocations != 0 )
            std::cerr << "pyramid allocations: " << pyramid_volume_allocations << ' ' << pyramid_face_allocations
                      << ' ' << pyramid_point_allocations << '\n';
        check( pyramid_volume_allocations == 0 && pyramid_face_allocations == 0 && pyramid_point_allocations == 0,
               "Pyramid14 volume, face, and physical-point reinit perform zero instrumented ordinary/aligned new calls" );
    }
    }

    if( selected( "--volumes" ) )
    {
    {
        test_volume( moab::MBTRI, 2, { Point( 0, 0, 0 ), Point( 1, 0, 0 ), Point( 0, 1, 0 ) }, 0.5, 1.0 / 12.0,
                     1.0 / 24.0 );
        test_volume( moab::MBTRI, 2, { Point( 1, 2, 0 ), Point( 3, 2, 0 ), Point( 2, 5, 0 ) }, 3.0, 12.5, 18.0 );
        test_volume( moab::MBQUAD, 2, { Point( 1, 2, 0 ), Point( 3, 3, 0 ), Point( 4, 6, 0 ), Point( 2, 5, 0 ) }, 5.0,
                     100.0 / 3.0, 625.0 / 12.0, true );
        test_volume( moab::MBQUAD, 2, { Point( 0, 0, 0 ), Point( 2, 0, 0 ), Point( 2.5, 1, 0 ), Point( 0, 1, 0 ) },
                     2.25, 123.0 / 32.0, 131.0 / 96.0 );

        test_volume( moab::MBTET, 3, { Point( 0, 0, 0 ), Point( 2, 0, 0 ), Point( 0, 1, 0 ), Point( 0, 0, 3 ) }, 1.0,
                     0.4, 0.1 );
        test_volume( moab::MBHEX, 3,
                     { Point( 1, 2, 3 ), Point( 3, 3, 3 ), Point( 4, 6, 3 ), Point( 2, 5, 3 ), Point( 1.5, 2, 5 ),
                       Point( 3.5, 3, 5 ), Point( 4.5, 6, 5 ), Point( 2.5, 5, 5 ) },
                     10.0, 80.0, 685.0 / 6.0, true );
    }
    }

    // An unrecognized selector must fail rather than pass by running nothing:
    // a green test that executed no assertion is worse than a red one.
    if( !karma_section_recognized )
    {
        std::cerr << "unknown section '" << karma_section << "'\n";
        PetscFinalize();
        return 2;
    }

    ierr = PetscFinalize();
    if( ierr ) return ierr;
    return failures == 0 ? 0 : 1;
}
