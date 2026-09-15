// Tests for mesh-side topology and face finite-element evaluation.

#include "karma_exceptions.h"
#include "karma_fe_base.h"
#include "karma_mesh.h"

#include <petsc.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
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
    if( type == moab::MBTRI ) return GeometryVariant::Tri3;
    if( type == moab::MBQUAD ) return GeometryVariant::Quad4;
    if( type == moab::MBTET ) return GeometryVariant::Tet4;
    if( type == moab::MBHEX ) return GeometryVariant::Hex8;
    if( type == moab::MBPRISM ) return GeometryVariant::Prism6;
    return GeometryVariant::Pyramid5;
}

KARMA::KARMABasis p1_basis( KARMA::GeometryVariant variant )
{
    using namespace KARMA;
    return KARMABasisRegistry().create( { reference_cell( variant ), FieldBasisFamily::LagrangeNodal, 1,
                                          BasisContinuity::cG } );
}

moab::EntityHandle add_element( KARMA::KARMAMesh& mesh, moab::EntityType type,
                                const std::vector< KARMA::Point >& points,
                                const std::vector< unsigned >& permutation = {} )
{
    std::vector< double > coordinates;
    for( const auto& point : points )
        for( int d = 0; d < 3; ++d )
            coordinates.push_back( point( d ) );
    moab::Range vertices;
    check( mesh.moab()->create_vertices( coordinates.data(), static_cast< int >( points.size() ), vertices ) ==
               moab::MB_SUCCESS,
           "create face-test vertices" );
    std::vector< moab::EntityHandle > handles( vertices.begin(), vertices.end() );
    std::vector< moab::EntityHandle > connectivity( handles.size() );
    for( std::size_t i = 0; i < handles.size(); ++i )
        connectivity[i] = handles[permutation.empty() ? i : permutation[i]];
    moab::EntityHandle element = 0;
    check( mesh.moab()->create_element( type, connectivity.data(), static_cast< int >( connectivity.size() ),
                                        element ) == moab::MB_SUCCESS,
           "create face-test element" );
    mesh.assign_ids();
    mesh.update();
    return element;
}

void check_poisoned( const KARMA::KARMAFEBase& fe, const std::string& path )
{
    check( !fe.is_valid(), path + " nonfinite nodal coordinate leaves FE invalid" );
    for( const KARMA::Real value : fe.get_JxW() )
        check( !std::isfinite( value ), path + " failure poisons JxW" );
    for( const auto& row : fe.get_phi() )
        for( const KARMA::Real value : row )
            check( !std::isfinite( value ), path + " failure poisons phi" );
    for( const auto& row : fe.get_dphi() )
        for( const KARMA::Gradient& value : row )
            check( !value.allFinite(), path + " failure poisons dphi" );
    for( const KARMA::Point& point : fe.get_xyz() )
        check( !point.allFinite(), path + " failure poisons xyz" );
    for( const KARMA::Vector3& normal : fe.get_normals() )
        check( !normal.allFinite(), path + " failure poisons normals" );
}

enum class ReinitPath
{
    Volume,
    Face,
    PhysicalPoints
};

void test_nonfinite_nodal_coordinate( ReinitPath path, int component, KARMA::Real invalid )
{
    using namespace KARMA;
    std::vector< Point > points{ Point( 0, 0, 0 ), Point( 1, 0, 0 ), Point( 0, 1, 0 ) };
    points.front()( component ) = invalid;
    KARMAMesh mesh( 2, PETSC_COMM_WORLD );
    const auto element = add_element( mesh, moab::MBTRI, points );
    const auto domain = path == ReinitPath::Volume ? FEIntegrationDomain::Volume : FEIntegrationDomain::Face;
    KARMAFEBase fe( mesh, GeometryVariant::Tri3, p1_basis( GeometryVariant::Tri3 ), 2, domain );
    const std::string name = path == ReinitPath::Volume ? "volume" : path == ReinitPath::Face ? "face" : "physical-point";

    if( path == ReinitPath::Volume )
        check( throws( [&] { fe.reinit( element ); } ), name + " rejects nonfinite nodal coordinate" );
    else if( path == ReinitPath::Face )
        check( throws( [&] { fe.reinit( element, 0 ); } ), name + " rejects nonfinite nodal coordinate" );
    else
    {
        std::vector< Point > physical_points( fe.get_xyz().size(), Point( 0.25, 0.25, 0.0 ) );
        check( throws( [&] { fe.reinit_at_physical_points( element, physical_points ); } ),
               name + " rejects nonfinite nodal coordinate" );
    }
    check_poisoned( fe, name );

    const moab::EntityHandle* connectivity = nullptr;
    int connectivity_size                  = 0;
    check( mesh.moab()->get_connectivity( element, connectivity, connectivity_size ) == moab::MB_SUCCESS,
           name + " gets connectivity to restore coordinate" );
    const double restored[3] = { 0.0, 0.0, 0.0 };
    check( mesh.moab()->set_coords( connectivity, 1, restored ) == moab::MB_SUCCESS,
           name + " restores finite nodal coordinate" );
    if( path == ReinitPath::Volume )
        fe.reinit( element );
    else if( path == ReinitPath::Face )
        fe.reinit( element, 0 );
    else
    {
        std::vector< Point > physical_points( fe.get_xyz().size(), Point( 0.25, 0.25, 0.0 ) );
        fe.reinit_at_physical_points( element, physical_points );
    }
    check( fe.is_valid(), name + " accepts restored finite nodal coordinate" );
}

KARMA::Real side_measure( const KARMA::KARMAMesh& mesh, moab::EntityHandle element, unsigned side )
{
    const auto handles = mesh.side_vertices( element, side );
    std::vector< KARMA::Real > coordinates( 3 * handles.size() );
    check( mesh.moab()->get_coords( handles.data(), static_cast< int >( handles.size() ), coordinates.data() ) ==
               moab::MB_SUCCESS,
           "get side coordinates for analytic measure" );
    if( handles.size() == 2 )
        return ( KARMA::point_at( coordinates.data(), 1 ) - KARMA::point_at( coordinates.data(), 0 ) ).norm();
    if( handles.size() == 3 )
        return 0.5 * KARMA::cross( KARMA::point_at( coordinates.data(), 1 ) - KARMA::point_at( coordinates.data(), 0 ),
                                   KARMA::point_at( coordinates.data(), 2 ) - KARMA::point_at( coordinates.data(), 0 ) )
                         .norm();
    return 0.5 * ( KARMA::cross( KARMA::point_at( coordinates.data(), 1 ) - KARMA::point_at( coordinates.data(), 0 ),
                                 KARMA::point_at( coordinates.data(), 2 ) - KARMA::point_at( coordinates.data(), 0 ) )
                       .norm() +
                   KARMA::cross( KARMA::point_at( coordinates.data(), 2 ) - KARMA::point_at( coordinates.data(), 0 ),
                                 KARMA::point_at( coordinates.data(), 3 ) - KARMA::point_at( coordinates.data(), 0 ) )
                       .norm() );
}

void test_all_faces( moab::EntityType type, unsigned dimension, const std::vector< KARMA::Point >& points,
                     KARMA::Real volume )
{
    using namespace KARMA;
    KARMAMesh mesh( dimension, PETSC_COMM_WORLD );
    const auto element    = add_element( mesh, type, points );
    int communicator_size = 1;
    check( MPI_Comm_size( PETSC_COMM_WORLD, &communicator_size ) == MPI_SUCCESS, "query MPI size for face test" );
    if( communicator_size > 1 ) mesh.resolve_shared_entities();
    const auto variant = linear_variant( type );
    KARMAFEBase face( mesh, variant, p1_basis( variant ), 2, FEIntegrationDomain::Face );
    const auto& jxw            = face.get_JxW();
    const auto& phi            = face.get_phi();
    const auto& dphi           = face.get_dphi();
    const auto& xyz            = face.get_xyz();
        const auto& normals        = face.get_normals();
        check( face.n_quadrature_points() == jxw.size(), "active face count matches exposed vector size" );
        const Real* jxw_data       = jxw.data();

    const Point* xyz_data      = xyz.data();
    const Vector3* normal_data = normals.data();
    std::vector< const Real* > phi_rows;
    std::vector< const Gradient* > dphi_rows;
    for( std::size_t i = 0; i < phi.size(); ++i )
    {
        phi_rows.push_back( phi[i].data() );
        dphi_rows.push_back( dphi[i].data() );
    }

    const unsigned expected_sides = type == moab::MBTRI ? 3 : type == moab::MBQUAD || type == moab::MBTET ? 4
                                                           : type == moab::MBPRISM || type == moab::MBPYRAMID ? 5 : 6;
    check( mesh.n_sides( element ) == expected_sides, "CN side count for topology" );
    const moab::EntityHandle* parent_vertices = nullptr;
    int parent_size                           = 0;
    check( mesh.moab()->get_connectivity( element, parent_vertices, parent_size ) == moab::MB_SUCCESS,
           "get parent connectivity" );
    Real flux = 0.0;
    std::vector< Real > first_phi;
    Vector3 first_normal = Vector3::Zero();
    for( unsigned side = 0; side < mesh.n_sides( element ); ++side )
    {
        face.reinit( element, side );
        const auto expected_side_nodes = mesh.side_vertices( element, side ).size();
        check( expected_side_nodes == ( dimension == 2 ? 2u : type == moab::MBTET ? 3u
                                                  : type == moab::MBPRISM ? ( side < 3 ? 4u : 3u )
                                                  : type == moab::MBPYRAMID ? ( side < 4 ? 3u : 4u ) : 4u ),
               "CN side connectivity size for topology" );

        if( communicator_size == 1 )
            check( mesh.is_boundary_side( element, side ), "single-element side is boundary" );
        if( side == 0 )
        {
            for( const auto& row : phi )
                first_phi.push_back( row.front() );
            first_normal = normals.front();
        }
        Real area             = 0.0;
        const auto side_nodes = mesh.side_vertices( element, side );
        for( std::size_t q = 0; q < jxw.size(); ++q )
        {
            area += jxw[q];
            close( normals[q].norm(), 1.0, "unit face normal" );
            check( dot( normals[q], xyz[q] - mesh.centroid( element ) ) > 0.0, "outward face normal" );
            Point interpolated = Point::Zero();
            for( std::size_t i = 0; i < phi.size(); ++i )
            {
                interpolated += phi[i][q] * points[i];
                if( std::find( side_nodes.begin(), side_nodes.end(), parent_vertices[i] ) == side_nodes.end() )
                    close( phi[i][q], 0.0, "off-face parent basis is zero" );
            }
            close( ( interpolated - xyz[q] ).norm(), 0.0, "face parent basis reproduces physical point" );
            flux += jxw[q] * dot( xyz[q], normals[q] );
        }
        close( area, side_measure( mesh, element, side ), "analytic side measure" );

        if( type != moab::MBPYRAMID )
        {
            std::vector< Point > side_points;
            for( std::size_t q = 0; q < jxw.size(); ++q )
            {
                double coordinates[3]{};
                const moab::EntityHandle node = side_nodes[q % side_nodes.size()];
                check( mesh.moab()->get_coords( &node, 1, coordinates ) == moab::MB_SUCCESS,
                       "get side node for direct basis check" );
                side_points.emplace_back( coordinates[0], coordinates[1], coordinates[2] );
            }
            face.reinit_at_physical_points( element, side_points );
            for( std::size_t q = 0; q < side_points.size(); ++q )
                for( int i = 0; i < parent_size; ++i )
                    close( phi[static_cast< std::size_t >( i )][q],
                           parent_vertices[i] == side_nodes[q % side_nodes.size()] ? 1.0 : 0.0,
                           "face basis has expected parent-local nodal value" );
        }

    }

    close( flux, static_cast< Real >( dimension ) * volume, "divergence theorem for identity field" );
    check( jxw.data() == jxw_data && xyz.data() == xyz_data && normals.data() == normal_data,
           "all face outer storage remains stable" );
    for( std::size_t i = 0; i < phi.size(); ++i )
    {
        check( phi[i].data() == phi_rows[i], "every face phi row remains stable" );
        check( dphi[i].data() == dphi_rows[i], "every face dphi row remains stable" );
    }
    bool changed = false;
    for( std::size_t i = 0; i < phi.size(); ++i )
        changed = changed || std::abs( phi[i].front() - first_phi[i] ) > 1.e-12;
    check( changed, "face basis values change across distinct sides" );
    bool normal_changed = false;
    for( unsigned side = 0; side < mesh.n_sides( element ); ++side )
    {
        face.reinit( element, side );
        normal_changed = normal_changed || ( normals.front() - first_normal ).norm() > 1.e-12;
    }
    check( normal_changed, "normal values change across distinct sides" );

    check( throws( [&] { face.reinit( element, mesh.n_sides( element ) ); } ),
           "out-of-range FE side is rejected" );
    check_poisoned( face, "out-of-range FE side" );
    face.reinit( element, 0 );
    check( face.is_valid(), "face FE recovers after out-of-range side" );
}
}  // namespace

int main( int argc, char** argv )
{
    PetscErrorCode ierr = PetscInitialize( &argc, &argv, nullptr, nullptr );
    if( ierr ) return ierr;

    using namespace KARMA;
    if( argc > 1 && std::string_view( argv[1] ) == "--shared-hex" )
    {
        PetscOptionsClearValue( nullptr, "--shared-hex" );
        {
            KARMAMesh mesh( 3, PETSC_COMM_WORLD );
            mesh.generate_box( 2, 1, 1, Point( 0, 0, 0 ), Point( 2, 1, 1 ) );
            const auto first = mesh.elements().front();
            const auto second = mesh.elements().back();
            unsigned first_side = mesh.n_sides( first );
            unsigned second_side = mesh.n_sides( second );
            unsigned first_matches = 0;
            unsigned second_matches = 0;
            for( unsigned side = 0; side < mesh.n_sides( first ); ++side )
                if( mesh.neighbor( first, side ) == second )
                {
                    first_side = side;
                    ++first_matches;
                }
            for( unsigned side = 0; side < mesh.n_sides( second ); ++side )
                if( mesh.neighbor( second, side ) == first )
                {
                    second_side = side;
                    ++second_matches;
                }
            check( first_matches == 1 && second_matches == 1,
                   "focused hex pair has exactly one reciprocal shared face" );

            KARMAFEBase face( mesh, GeometryVariant::Hex8, p1_basis( GeometryVariant::Hex8 ), 2,
                              FEIntegrationDomain::Face );
            face.reinit( first, first_side );
            const auto points = face.get_xyz();
            const auto normals = face.get_normals();
            face.reinit( second, second_side );
            for( std::size_t q = 0; q < points.size(); ++q )
                close( ( normals[q] + face.get_normals()[q] ).norm(), 0.0,
                       "focused shared-hex normals are opposite" );
            face.reinit_at_physical_points( second, points );
            for( std::size_t q = 0; q < points.size(); ++q )
                close( ( points[q] - face.get_xyz()[q] ).norm(), 0.0,
                       "focused shared-hex physical transfer roundtrips" );
        }
        ierr = PetscFinalize();
        if( ierr ) return ierr;
        return failures == 0 ? 0 : 1;
    }
    int size = 1;
    check( MPI_Comm_size( PETSC_COMM_WORLD, &size ) == MPI_SUCCESS, "query replicated-SPMD MPI communicator size" );
    int rank = 0;
    check( MPI_Comm_rank( PETSC_COMM_WORLD, &rank ) == MPI_SUCCESS, "query replicated-SPMD face rank" );
    check( rank >= 0 && rank < size, "replicated-SPMD face numerical checks execute on this rank" );

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

    if( selected( "--reinit-paths" ) )
    {
    for( const ReinitPath path : { ReinitPath::Volume, ReinitPath::Face, ReinitPath::PhysicalPoints } )
        for( int component = 0; component < 3; ++component )
            for( const Real invalid : { std::numeric_limits< Real >::quiet_NaN(),
                                        std::numeric_limits< Real >::infinity(),
                                        -std::numeric_limits< Real >::infinity() } )
                test_nonfinite_nodal_coordinate( path, component, invalid );
    {
        test_all_faces( moab::MBTRI, 2, { Point( 0, 0, 0 ), Point( 2, 0, 0 ), Point( 0, 1, 0 ) }, 1.0 );
        test_all_faces( moab::MBQUAD, 2, { Point( 0, 0, 0 ), Point( 2, 0, 0 ), Point( 2, 1, 0 ), Point( 0, 1, 0 ) },
                        2.0 );
        test_all_faces( moab::MBTET, 3, { Point( 0, 0, 0 ), Point( 2, 0, 0 ), Point( 0, 1, 0 ), Point( 0, 0, 3 ) },
                        1.0 );
        test_all_faces( moab::MBHEX, 3,
                         { Point( 0, 0, 0 ), Point( 2, 0, 0 ), Point( 2, 1, 0 ), Point( 0, 1, 0 ), Point( 0, 0, 3 ),
                           Point( 2, 0, 3 ), Point( 2, 1, 3 ), Point( 0, 1, 3 ) },
                         6.0 );
        test_all_faces( moab::MBPRISM, 3,
                        { Point( 0, 0, 0 ), Point( 2, 0, 0 ), Point( 0, 1, 0 ), Point( 0, 0, 3 ),
                          Point( 2, 0, 3 ), Point( 0, 1, 3 ) },
                        3.0 );
        test_all_faces( moab::MBPYRAMID, 3,
                        { Point( -1, -1, 0 ), Point( 1, -1, 0 ), Point( 1, 1, 0 ), Point( -1, 1, 0 ),
                          Point( 0, 0, 3 ) },
                        4.0 );

    }
    }

    if( size == 1 )
    {
        // Grouped exactly as karma_fe's --variants sections are, and for the
        // same reason: one test over all 17 parent variants ran ~166s and
        // bounded the suite. Each iteration builds its own mesh and element
        // and `boundary_measure` is reset per variant, so the variants are
        // independent and the split changes scheduling, not coverage. The
        // groups partition the original list exactly (7 + 4 + 6 = 17).
        struct ParentVariantGroup
        {
            const char* section;
            std::vector< GeometryVariant > variants;
        };
        const std::vector< ParentVariantGroup > parent_variant_groups{
            { "--parent-variants-line-surface",
              { GeometryVariant::Edge2, GeometryVariant::Edge3, GeometryVariant::Tri3, GeometryVariant::Tri6,
                GeometryVariant::Quad4, GeometryVariant::Quad8, GeometryVariant::Quad9 } },
            { "--parent-variants-tet-hex",
              { GeometryVariant::Tet4, GeometryVariant::Tet10, GeometryVariant::Hex8, GeometryVariant::Hex27 } },
            { "--parent-variants-prism-pyramid",
              { GeometryVariant::Prism6, GeometryVariant::Prism15, GeometryVariant::Prism18,
                GeometryVariant::Pyramid5, GeometryVariant::Pyramid13, GeometryVariant::Pyramid14 } } };

        for( const auto& parent_group : parent_variant_groups )
        {
        if( !selected( parent_group.section ) ) continue;
        for( const GeometryVariant variant : parent_group.variants )
        {
            const KARMAReferenceTopology reference( variant );
            auto nodes = reference.reference_nodes();
            if( variant == GeometryVariant::Quad8 ) nodes[4]( 2 ) = 0.2;
            KARMAMesh mesh( 3, PETSC_COMM_WORLD );
            const auto element = add_element( mesh, reference.topology(), nodes );
            Real boundary_measure = 0.0;
            for( unsigned order = 0; order <= 2; ++order )
                for( const BasisContinuity continuity : { BasisContinuity::cG, BasisContinuity::dG,
                                                          BasisContinuity::SE } )
                {
                    if( order == 0 && continuity == BasisContinuity::cG ) continue;
                    const BasisKey key{ reference_cell( variant ), FieldBasisFamily::LagrangeNodal, order,
                                        continuity };
                    KARMAFEBase face( mesh, variant, KARMABasisRegistry().create( key ), 4,
                                      FEIntegrationDomain::Face );
                    const auto* phi_rows = face.get_phi().data();
                    const auto* dphi_rows = face.get_dphi().data();
                    std::vector< const Real* > phi_data;
                    std::vector< const Gradient* > dphi_data;
                    for( std::size_t i = 0; i < face.get_phi().size(); ++i )
                    {
                        phi_data.push_back( face.get_phi()[i].data() );
                        dphi_data.push_back( face.get_dphi()[i].data() );
                    }
                    Real combination_boundary_measure = 0.0;
                    for( unsigned side = 0; side < reference.side_count(); ++side )
                    {
                        face.reinit( element, side );
                        check( face.get_phi().data() == phi_rows && face.get_dphi().data() == dphi_rows,
                               "face matrix nested vector objects remain stable" );
                        for( std::size_t i = 0; i < face.get_phi().size(); ++i )
                        {
                            check( face.get_phi()[i].data() == phi_data[i], "face matrix phi data remains stable" );
                            check( face.get_dphi()[i].data() == dphi_data[i], "face matrix dphi data remains stable" );
                        }
                        Real measure = 0.0;
                        for( std::size_t q = 0; q < face.get_JxW().size(); ++q )
                        {
                            measure += face.get_JxW()[q];
                            close( face.get_normals()[q].norm(), 1.0, "face matrix normal is unit" );
                            check( dot( face.get_normals()[q], face.get_xyz()[q] - mesh.centroid( element ) ) > 0.0,
                                   "face matrix normal points outward" );
                            Real sum = 0.0;
                            Gradient gradient_sum = Gradient::Zero();
                            for( std::size_t i = 0; i < face.get_phi().size(); ++i )
                            {
                                sum += face.get_phi()[i][q];
                                gradient_sum += face.get_dphi()[i][q];
                            }
                            close( sum, 1.0, "face matrix parent trace partitions unity", 2.e-10 );
                            close( gradient_sum.norm(), 0.0, "face matrix parent gradients sum to zero", 2.e-9 );
                        }
                        check( std::isfinite( measure ) && measure > 0.0,
                               "face matrix side measure is finite and positive" );
                        combination_boundary_measure += measure;
                    }
                    if( order == 2 && continuity == BasisContinuity::dG )
                        boundary_measure = combination_boundary_measure;
                }
            const Real expected_boundary = reference.topology() == moab::MBEDGE ? 2.0
                                           : reference.topology() == moab::MBTRI ? 2.0 + std::sqrt( 2.0 )
                                           : reference.topology() == moab::MBQUAD ? 4.0
                                           : reference.topology() == moab::MBTET ? 1.5 + std::sqrt( 3.0 ) / 2.0
                                           : reference.topology() == moab::MBHEX ? 6.0
                                           : reference.topology() == moab::MBPRISM ? 3.0 + std::sqrt( 2.0 )
                                                                                   : 4.0 + 4.0 * std::sqrt( 2.0 );
            if( variant != GeometryVariant::Quad8 )
                close( boundary_measure, expected_boundary, "quadratic reference-cell boundary measure", 2.e-10 );
            else
                check( boundary_measure > 4.0, "curved Quad8 edge increases boundary measure" );
        }
        }
    }

    if( selected( "--box-mesh" ) )
    {
    {
        KARMAMesh mesh( 3, PETSC_COMM_WORLD );
        mesh.generate_box( 2, 1, 1, Point( 0, 0, 0 ), Point( 2, 1, 1 ) );
        if( size > 1 )
        {
            check( !throws( [&] { (void)mesh.neighbor( mesh.elements().front(), 0 ); } ),
                   "replicated MPI permits local neighbor queries" );
            check( !throws( [&] { (void)mesh.is_boundary_side( mesh.elements().front(), 0 ); } ),
                   "replicated MPI permits local boundary queries" );
            mesh.resolve_shared_entities();
            check( !throws( [&] { (void)mesh.neighbor( mesh.elements().front(), 0 ); } ),
                   "distributed neighbor query succeeds after sharing resolution" );
            check( !throws( [&] { (void)mesh.is_boundary_side( mesh.elements().front(), 0 ); } ),
                   "distributed boundary query succeeds after sharing resolution" );
        }
        if( size == 1 )
        {
        const auto first  = mesh.elements().front();
        const auto second = mesh.elements().back();

        check( mesh.n_sides( first ) == 6, "hex has six sides" );
        std::array< moab::EntityHandle, 4 > side_buffer{};
        for( unsigned side = 0; side < mesh.n_sides( first ); ++side )
        {
            const auto vertices = mesh.side_vertices( first, side );
            check( vertices.size() == 4, "hex side has four vertices" );
            check( mesh.side_vertices( first, side, side_buffer ) == 4,
                   "caller-buffer side connectivity has four vertices" );
            check( std::equal( vertices.begin(), vertices.end(), side_buffer.begin() ),
                   "caller-buffer side connectivity preserves CN order" );
            const auto neighbor = mesh.neighbor( first, side );
            check( mesh.is_boundary_side( first, side ) == !neighbor.has_value(),
                   "boundary classification agrees with neighbor lookup" );
            const auto id = mesh.side_id( first, side );
            check( id.element == mesh.global_id( first ) && id.side == side,
                   "persistent side identity uses global element id and CN side index" );
        }

        unsigned shared_side = mesh.n_sides( first );
        for( unsigned side = 0; side < mesh.n_sides( first ); ++side )
            if( mesh.neighbor( first, side ) == second ) shared_side = side;
        check( shared_side < mesh.n_sides( first ), "first element discovers second as neighbor" );
        check( !mesh.is_boundary_side( first, shared_side ), "shared side is interior" );

        unsigned reverse_side = mesh.n_sides( second );
        for( unsigned side = 0; side < mesh.n_sides( second ); ++side )
            if( mesh.neighbor( second, side ) == first ) reverse_side = side;
        check( reverse_side < mesh.n_sides( second ), "neighbor discovery is reciprocal" );
        check( throws( [&] { (void)mesh.side_vertices( first, 6 ); } ), "out-of-range side is rejected" );
        GlobalId invalid_id = -1;
        check( mesh.moab()->tag_set_data( mesh.global_id_tag(), &first, 1, &invalid_id ) == moab::MB_SUCCESS,
               "invalidate element global id" );
        check( throws( [&] { (void)mesh.side_id( first, 0 ); } ), "side identity rejects an unassigned global id" );
        mesh.assign_ids();

        KARMAFEBase face( mesh, GeometryVariant::Hex8, p1_basis( GeometryVariant::Hex8 ), 2, FEIntegrationDomain::Face );
        const auto& jxw     = face.get_JxW();
        const auto& phi     = face.get_phi();
        const auto& dphi    = face.get_dphi();
        const auto& xyz     = face.get_xyz();
        const auto& normals = face.get_normals();
        check( !face.is_valid(), "face starts invalid before first reinit" );
        for( const Real weight : jxw )
            check( !std::isfinite( weight ), "initial JxW is poisoned" );
        for( const auto& row : phi )
            for( const Real value : row )
                check( !std::isfinite( value ), "initial phi is poisoned" );
        const auto* phi_row  = phi.front().data();
        const auto* dphi_row = dphi.front().data();

        Real total_flux = 0.0;
        for( unsigned side = 0; side < mesh.n_sides( first ); ++side )
        {
            face.reinit( first, side );
            check( face.is_valid(), "successful face reinit marks state valid" );
            Real area = 0.0;
            for( std::size_t q = 0; q < jxw.size(); ++q )
            {
                area += jxw[q];
                close( normals[q].norm(), 1.0, "face normal has unit length" );
                check( dot( normals[q], xyz[q] - mesh.centroid( first ) ) > 0.0, "face normal points out of parent" );
                Real phi_sum = 0.0;
                for( const auto& row : phi )
                    phi_sum += row[q];
                close( phi_sum, 1.0, "parent basis partitions unity on face" );
                total_flux += jxw[q] * dot( xyz[q], normals[q] );
            }
            close( area, 1.0, "unit-cube face area" );
        }
        close( total_flux, 3.0, "linear-field divergence theorem on unit cube" );
        check( &jxw == &face.get_JxW() && &phi == &face.get_phi() && &dphi == &face.get_dphi() &&
                   &xyz == &face.get_xyz() && &normals == &face.get_normals(),
               "face outer references remain stable" );
        check( phi_row == face.get_phi().front().data() && dphi_row == face.get_dphi().front().data(),
               "face nested references remain stable" );

        face.reinit( first, shared_side );
        const auto shared_points   = face.get_xyz();
        const Vector3 first_normal = face.get_normals().front();
        face.reinit( second, reverse_side );
        close( ( first_normal + face.get_normals().front() ).norm(), 0.0,
               "shared-face normals have opposite orientation" );
        face.reinit_at_physical_points( second, shared_points );
        face.reinit( second, reverse_side );
        const auto self_points = face.get_xyz();
        face.reinit_at_physical_points( second, face.get_xyz() );
        check( face.is_valid(), "physical-point reinit accepts exact self alias" );
        check( face.n_quadrature_points() == self_points.size(), "self-alias preserves active point count" );
        for( std::size_t q = 0; q < self_points.size(); ++q )
            close( ( face.get_xyz()[q] - self_points[q] ).norm(), 0.0,
                   "self-aliased physical-point reinit preserves coordinates" );
        face.reinit_at_physical_points( second, std::span< const Point >( shared_points.data(), 1 ) );
        check( face.n_quadrature_points() == 1 && face.get_xyz().size() == 1,
               "physical-point reinit accepts an arbitrary count within capacity" );
        face.reinit( second, reverse_side );
        check( face.get_xyz().size() == shared_points.size(), "native face reinit restores its active size" );
        face.reinit_at_physical_points( second, shared_points );
        for( const Real weight : jxw )
            check( std::isnan( weight ), "neighbor trace invalidates JxW" );
        for( const Vector3& normal : normals )
            check( !normal.allFinite(), "neighbor trace invalidates normals" );
        for( std::size_t q = 0; q < phi.front().size(); ++q )
        {
            Real sum = 0.0;
            for( const auto& row : phi )
                sum += row[q];
            close( sum, 1.0, "neighbor parent basis partitions unity" );
            close( ( xyz[q] - shared_points[q] ).norm(), 0.0, "neighbor trace keeps shared physical point" );
        }

        const auto shared_nodes = mesh.side_vertices( first, shared_side );
        std::vector< Point > nodal_points;
        for( auto it = shared_nodes.rbegin(); it != shared_nodes.rend(); ++it )
        {
            double coordinate[3]{};
            check( mesh.moab()->get_coords( &*it, 1, coordinate ) == moab::MB_SUCCESS,
                   "get permuted shared-face node" );
            nodal_points.emplace_back( coordinate[0], coordinate[1], coordinate[2] );
        }
        face.reinit_at_physical_points( second, nodal_points );
        const moab::EntityHandle* second_conn = nullptr;
        int second_size                       = 0;
        check( mesh.moab()->get_connectivity( second, second_conn, second_size ) == moab::MB_SUCCESS,
               "get neighbor local connectivity" );
        for( std::size_t q = 0; q < nodal_points.size(); ++q )
            for( int i = 0; i < second_size; ++i )
            {
                const auto expected = second_conn[i] == shared_nodes[shared_nodes.size() - 1 - q] ? 1.0 : 0.0;
                close( phi[static_cast< std::size_t >( i )][q], expected, "neighbor basis follows local node mapping" );
            }

        face.reinit( first, shared_side );
        PetscLogDouble allocation_before = 0.0;
        PetscLogDouble allocation_after  = 0.0;
        check( PetscMallocGetCurrentUsage( &allocation_before ) == PETSC_SUCCESS,
               "query PETSc allocation usage before reinit" );
        for( int repetition = 0; repetition < 32; ++repetition )
        {
            face.reinit( first, shared_side );
            face.reinit_at_physical_points( second, shared_points );
        }
        check( PetscMallocGetCurrentUsage( &allocation_after ) == PETSC_SUCCESS,
               "query PETSc allocation usage after reinit" );
        close( allocation_after, allocation_before, "steady-state face and neighbor reinit perform no PETSc allocation",
               0.0 );
        std::vector< Point > outside( shared_points.size(), Point( 4.0, 4.0, 4.0 ) );
        check( throws( [&] { face.reinit_at_physical_points( second, outside ); } ),
               "reverse mapping rejects points outside the parent" );
        check( !face.is_valid(), "failed neighbor trace marks FE state invalid" );
        for( const Real weight : jxw )
            check( !std::isfinite( weight ), "failed trace poisons JxW" );
        for( const auto& row : phi )
            for( const Real value : row )
                check( !std::isfinite( value ), "failed trace poisons phi" );
        for( const auto& row : dphi )
            for( const Gradient& value : row )
                check( !value.allFinite(), "failed trace poisons dphi" );
        for( const Point& point : xyz )
            check( !point.allFinite(), "failed trace poisons xyz" );
        for( const Vector3& normal : normals )
            check( !normal.allFinite(), "failed trace poisons normals" );
        face.reinit( first, shared_side );
        check( face.is_valid(), "successful reinit restores validity after failure" );
        for( const Real weight : jxw )
            check( std::isfinite( weight ), "successful reinit restores JxW" );
        }
    }
    }

    if( selected( "--face-integration" ) )
    {
    {
        KARMAMesh mesh( 2, PETSC_COMM_WORLD );
        const auto element =
            add_element( mesh, moab::MBTRI, { Point( 0, 0, 0 ), Point( 1, 0, 0 ), Point( 0, 1, 0 ) } );
        KARMAFEBase face( mesh, GeometryVariant::Tri3, p1_basis( GeometryVariant::Tri3 ), 2, FEIntegrationDomain::Face );
        face.reinit( element, 0 );
        const auto valid_points = face.get_xyz();
        auto points             = valid_points;
        for( Point& point : points )
            point( 2 ) += 1.e-4;
        check( throws( [&] { face.reinit_at_physical_points( element, points ); } ),
               "two-dimensional trace rejects points away from its global XY plane" );
        check( !face.is_valid(), "off-plane trace leaves FE state invalid" );

        const Real nan = std::numeric_limits< Real >::quiet_NaN();
        for( int component = 0; component < 3; ++component )
        {
            points = valid_points;
            points.front()( component ) = nan;
            check( throws( [&] { face.reinit_at_physical_points( element, points ); } ),
                   "reverse mapping rejects a NaN physical coordinate" );
            check( !face.is_valid(), "NaN trace point leaves FE state invalid" );
        }
        for( const Real infinity : { std::numeric_limits< Real >::infinity(),
                                     -std::numeric_limits< Real >::infinity() } )
        {
            points              = valid_points;
            points.front()( 0 ) = infinity;
            check( throws( [&] { face.reinit_at_physical_points( element, points ); } ),
                   "reverse mapping rejects an infinite physical coordinate" );
            check( !face.is_valid(), "infinite trace point leaves FE state invalid" );
        }
    }
    }

    if( selected( "--nonconforming" ) )
    {
    if( size == 1 )
    {
        KARMAMesh mesh( 2, PETSC_COMM_WORLD );
        const double coords[] = { 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, -1, 0 };
        moab::Range vertices;
        check( mesh.moab()->create_vertices( coords, 4, vertices ) == moab::MB_SUCCESS,
               "create permuted-neighbor vertices" );
        std::vector< moab::EntityHandle > v( vertices.begin(), vertices.end() );
        const moab::EntityHandle first_conn[]  = { v[0], v[1], v[2] };
        const moab::EntityHandle second_conn[] = { v[1], v[0], v[3] };
        moab::EntityHandle first               = 0;
        moab::EntityHandle second              = 0;
        check( mesh.moab()->create_element( moab::MBTRI, first_conn, 3, first ) == moab::MB_SUCCESS,
               "create first permuted triangle" );
        check( mesh.moab()->create_element( moab::MBTRI, second_conn, 3, second ) == moab::MB_SUCCESS,
               "create second permuted triangle" );
        mesh.assign_ids();
        mesh.update();
        unsigned first_side  = mesh.n_sides( first );
        unsigned second_side = mesh.n_sides( second );
        for( unsigned side = 0; side < mesh.n_sides( first ); ++side )
            if( mesh.neighbor( first, side ) == second ) first_side = side;
        for( unsigned side = 0; side < mesh.n_sides( second ); ++side )
            if( mesh.neighbor( second, side ) == first ) second_side = side;
        check( first_side < mesh.n_sides( first ) && second_side < mesh.n_sides( second ),
               "permuted triangles discover shared edge" );

        KARMAFEBase face( mesh, GeometryVariant::Tri3, p1_basis( GeometryVariant::Tri3 ), 2, FEIntegrationDomain::Face );
        face.reinit( first, first_side );
        const auto first_points  = face.get_xyz();
        const auto first_normals = face.get_normals();
        face.reinit( second, second_side );
        for( std::size_t q = 0; q < first_points.size(); ++q )
        {
            close( ( first_points[q] - face.get_xyz()[first_points.size() - 1 - q] ).norm(), 0.0,
                   "reversed local edge reverses quadrature-point ordering" );
            close( ( first_normals[q] + face.get_normals()[first_points.size() - 1 - q] ).norm(), 0.0,
                   "permuted shared-edge normals are opposite" );
        }
        face.reinit_at_physical_points( second, first_points );
        for( std::size_t q = 0; q < first_points.size(); ++q )
        {
            close( face.get_phi()[0][q], first_points[q]( 0 ), "permuted neighbor basis maps local node zero" );
            close( face.get_phi()[1][q], 1.0 - first_points[q]( 0 ), "permuted neighbor basis maps local node one" );
            close( face.get_phi()[2][q], 0.0, "permuted neighbor off-face basis is zero" );
        }
    }
    }

    if( selected( "--tet-prism" ) )
    {
    if( size == 1 )
    {
        for( const auto [tet_variant, prism_variant] : {
                 std::pair{ GeometryVariant::Tet4, GeometryVariant::Prism6 },
                 std::pair{ GeometryVariant::Tet10, GeometryVariant::Prism18 } } )
        {
            const KARMAReferenceTopology tet_reference( tet_variant );
            const KARMAReferenceTopology prism_reference( prism_variant );
            KARMAMesh tet_mesh( 3, PETSC_COMM_WORLD );
            KARMAMesh prism_mesh( 3, PETSC_COMM_WORLD );
            const auto tet = add_element( tet_mesh, moab::MBTET, tet_reference.reference_nodes() );
            const auto prism = add_element( prism_mesh, moab::MBPRISM, prism_reference.reference_nodes() );
            const unsigned order = tet_reference.order();
            const BasisKey tet_key{ ReferenceCell::Tetrahedron, FieldBasisFamily::LagrangeNodal, order,
                                    BasisContinuity::dG };
            const BasisKey prism_key{ ReferenceCell::Prism, FieldBasisFamily::LagrangeNodal, order,
                                      BasisContinuity::dG };
            const auto tet_basis = KARMABasisRegistry().create( tet_key );
            const auto prism_basis = KARMABasisRegistry().create( prism_key );
            const auto tet_nodes = tet_reference.reference_nodes();
            const auto prism_nodes = prism_reference.reference_nodes();

            KARMAFEBase tet_source( tet_mesh, tet_variant, tet_basis, 2, FEIntegrationDomain::Face );
            KARMAFEBase prism_target( prism_mesh, prism_variant, prism_basis, 6, FEIntegrationDomain::Face );
            prism_target.reinit( prism, 3 );
            const auto prism_native_count = prism_target.n_quadrature_points();
            const auto* prism_jxw_data = prism_target.get_JxW().data();
            const auto* prism_phi_rows = prism_target.get_phi().data();
            const auto* prism_dphi_rows = prism_target.get_dphi().data();
            const auto* prism_xyz_data = prism_target.get_xyz().data();
            const auto* prism_normal_data = prism_target.get_normals().data();
            std::vector< const Real* > prism_phi_data;
            std::vector< const Gradient* > prism_dphi_data;
            for( std::size_t i = 0; i < prism_target.get_phi().size(); ++i )
            {
                prism_phi_data.push_back( prism_target.get_phi()[i].data() );
                prism_dphi_data.push_back( prism_target.get_dphi()[i].data() );
            }
            tet_source.reinit( tet, 3 );
            const auto& tet_source_points = tet_source.get_xyz();
            const auto tet_source_count = tet_source_points.size();
            check( tet_source_count == 3 && prism_native_count == 12,
                   "tet source and prism target have expected degree-2 and degree-6 counts" );
            check( tet_source_count != prism_native_count,
                   "tet source count differs from prism target native count" );
            PetscLogDouble allocation_before = 0.0;
            PetscLogDouble allocation_after = 0.0;
            check( PetscMallocGetCurrentUsage( &allocation_before ) == PETSC_SUCCESS,
                   "query PETSc allocation usage before tetrahedron-to-prism transfer" );
            prism_target.reinit_at_physical_points( prism, tet_source.get_xyz() );
            check( PetscMallocGetCurrentUsage( &allocation_after ) == PETSC_SUCCESS,
                   "query PETSc allocation usage after tetrahedron-to-prism transfer" );
            close( allocation_after, allocation_before,
                   "tetrahedron-to-prism transfer performs no PETSc allocation", 0.0 );
            check( prism_target.n_quadrature_points() == tet_source_count &&
                       prism_target.get_xyz().size() == tet_source_count,
                   "prism target adopts tetrahedron source count" );
            check( prism_target.get_JxW().data() == prism_jxw_data &&
                       prism_target.get_phi().data() == prism_phi_rows &&
                       prism_target.get_dphi().data() == prism_dphi_rows &&
                       prism_target.get_xyz().data() == prism_xyz_data &&
                       prism_target.get_normals().data() == prism_normal_data,
                   "prism target output storage remains stable" );
            for( std::size_t i = 0; i < prism_target.get_phi().size(); ++i )
                check( prism_target.get_phi()[i].data() == prism_phi_data[i] &&
                           prism_target.get_dphi()[i].data() == prism_dphi_data[i],
                       "every prism target matrix row remains stable" );
            const auto tet_field = [order]( const Point& point ) {
                return 1.0 + 2.0 * point( 0 ) - 3.0 * point( 1 ) +
                       ( order == 2 ? point( 0 ) * point( 1 ) + 0.5 * point( 0 ) * point( 0 ) : 0.0 );
            };
            for( std::size_t q = 0; q < tet_source_count; ++q )
            {
                close( ( prism_target.get_xyz()[q] - tet_source_points[q] ).norm(), 0.0,
                       "prism target accepts tetrahedron source points", 2.e-10 );
                Real source_value = 0.0;
                Real target_value = 0.0;
                for( std::size_t i = 0; i < tet_source.get_phi().size(); ++i )
                    source_value += tet_field( tet_nodes[i] ) * tet_source.get_phi()[i][q];
                for( std::size_t i = 0; i < prism_target.get_phi().size(); ++i )
                    target_value += tet_field( prism_nodes[i] ) * prism_target.get_phi()[i][q];
                close( source_value, tet_field( tet_source_points[q] ),
                       "tet source reconstructs physical polynomial", 2.e-10 );
                close( target_value, tet_field( tet_source_points[q] ),
                       "prism target reconstructs physical polynomial", 2.e-10 );
            }
            prism_target.reinit( prism, 3 );
            check( prism_target.n_quadrature_points() == prism_native_count &&
                       prism_target.get_xyz().size() == prism_native_count,
                   "prism target native side reinit restores native count" );

            KARMAFEBase prism_source( prism_mesh, prism_variant, prism_basis, 2, FEIntegrationDomain::Face );
            KARMAFEBase tet_target( tet_mesh, tet_variant, tet_basis, 6, FEIntegrationDomain::Face );
            tet_target.reinit( tet, 3 );
            const auto tet_native_count = tet_target.n_quadrature_points();
            const auto* tet_jxw_data = tet_target.get_JxW().data();
            const auto* tet_phi_rows = tet_target.get_phi().data();
            const auto* tet_dphi_rows = tet_target.get_dphi().data();
            const auto* tet_xyz_data = tet_target.get_xyz().data();
            const auto* tet_normal_data = tet_target.get_normals().data();
            std::vector< const Real* > tet_phi_data;
            std::vector< const Gradient* > tet_dphi_data;
            for( std::size_t i = 0; i < tet_target.get_phi().size(); ++i )
            {
                tet_phi_data.push_back( tet_target.get_phi()[i].data() );
                tet_dphi_data.push_back( tet_target.get_dphi()[i].data() );
            }
            prism_source.reinit( prism, 3 );
            const auto& prism_source_points = prism_source.get_xyz();
            const auto prism_source_count = prism_source_points.size();
            check( prism_source_count == 3 && tet_native_count == 12,
                   "prism source and tetrahedron target have expected degree-2 and degree-6 counts" );
            check( prism_source_count != tet_native_count,
                   "prism source count differs from tetrahedron target native count" );
            allocation_before = 0.0;
            allocation_after = 0.0;
            check( PetscMallocGetCurrentUsage( &allocation_before ) == PETSC_SUCCESS,
                   "query PETSc allocation usage before prism-to-tetrahedron transfer" );
            tet_target.reinit_at_physical_points( tet, prism_source.get_xyz() );
            check( PetscMallocGetCurrentUsage( &allocation_after ) == PETSC_SUCCESS,
                   "query PETSc allocation usage after prism-to-tetrahedron transfer" );
            close( allocation_after, allocation_before,
                   "prism-to-tetrahedron transfer performs no PETSc allocation", 0.0 );
            check( tet_target.n_quadrature_points() == prism_source_count &&
                       tet_target.get_xyz().size() == prism_source_count,
                   "tetrahedron target adopts prism source count" );
            check( tet_target.get_JxW().data() == tet_jxw_data && tet_target.get_phi().data() == tet_phi_rows &&
                       tet_target.get_dphi().data() == tet_dphi_rows && tet_target.get_xyz().data() == tet_xyz_data &&
                       tet_target.get_normals().data() == tet_normal_data,
                   "tetrahedron target output storage remains stable" );
            for( std::size_t i = 0; i < tet_target.get_phi().size(); ++i )
                check( tet_target.get_phi()[i].data() == tet_phi_data[i] &&
                           tet_target.get_dphi()[i].data() == tet_dphi_data[i],
                       "every tetrahedron target matrix row remains stable" );
            const auto prism_field = [order]( const Point& point ) {
                return -0.5 + point( 0 ) + 4.0 * point( 1 ) +
                       ( order == 2 ? point( 1 ) * point( 1 ) - 0.25 * point( 0 ) * point( 1 ) : 0.0 );
            };
            for( std::size_t q = 0; q < prism_source_count; ++q )
            {
                close( ( tet_target.get_xyz()[q] - prism_source_points[q] ).norm(), 0.0,
                       "tetrahedron target accepts prism source points", 2.e-10 );
                Real source_value = 0.0;
                Real target_value = 0.0;
                for( std::size_t i = 0; i < prism_source.get_phi().size(); ++i )
                    source_value += prism_field( prism_nodes[i] ) * prism_source.get_phi()[i][q];
                for( std::size_t i = 0; i < tet_target.get_phi().size(); ++i )
                    target_value += prism_field( tet_nodes[i] ) * tet_target.get_phi()[i][q];
                close( source_value, prism_field( prism_source_points[q] ),
                       "prism source reconstructs physical polynomial", 2.e-10 );
                close( target_value, prism_field( prism_source_points[q] ),
                       "tetrahedron target reconstructs physical polynomial", 2.e-10 );
            }
            tet_target.reinit( tet, 3 );
            check( tet_target.n_quadrature_points() == tet_native_count &&
                       tet_target.get_xyz().size() == tet_native_count,
                   "tetrahedron target native side reinit restores native count" );
        }
    }
    }

    if( selected( "--edge-cases" ) )
    {
    if( size == 1 )
    {
        KARMAMesh mesh( 3, PETSC_COMM_WORLD );
        const double coords[] = { 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 3 };
        moab::Range vertices;
        check( mesh.moab()->create_vertices( coords, 4, vertices ) == moab::MB_SUCCESS, "create tetrahedron vertices" );
        std::vector< moab::EntityHandle > v( vertices.begin(), vertices.end() );
        moab::EntityHandle tetrahedron = 0;
        check( mesh.moab()->create_element( moab::MBTET, v.data(), 4, tetrahedron ) == moab::MB_SUCCESS,
               "create tetrahedron" );
        mesh.assign_ids();
        mesh.update();

        KARMAFEBase face( mesh, GeometryVariant::Tet4, p1_basis( GeometryVariant::Tet4 ), 2, FEIntegrationDomain::Face );
        std::vector< Real > areas;
        for( unsigned side = 0; side < mesh.n_sides( tetrahedron ); ++side )
        {
            face.reinit( tetrahedron, side );
            Real area = 0.0;
            for( const Real weight : face.get_JxW() )
                area += weight;
            areas.push_back( area );
        }
        std::sort( areas.begin(), areas.end() );
        const std::vector< Real > expected{ 1.0, 1.5, 3.0, 3.5 };
        for( std::size_t i = 0; i < areas.size(); ++i )
            close( areas[i], expected[i], "tetrahedron face area" );
    }

    if( size == 1 )
    {
        KARMAMesh tiny( 2, PETSC_COMM_WORLD );
        const auto tiny_element =
            add_element( tiny, moab::MBTRI, { Point( 0, 0, 0 ), Point( 1.e-10, 0, 0 ), Point( 0, 1.e-10, 0 ) } );
        KARMAFEBase tiny_face( tiny, GeometryVariant::Tri3, p1_basis( GeometryVariant::Tri3 ), 2, FEIntegrationDomain::Face );
        check( !throws( [&] { tiny_face.reinit( tiny_element, 0 ); } ), "tiny well-conditioned face remains valid" );
        Real tiny_length = 0.0;
        for( const Real weight : tiny_face.get_JxW() )
            tiny_length += weight;
        close( tiny_length, 1.e-10, "tiny face length", 1.e-20 );

        KARMAMesh collapsed( 2, PETSC_COMM_WORLD );
        const auto collapsed_element =
            add_element( collapsed, moab::MBTRI, { Point( 0, 0, 0 ), Point( 1, 0, 0 ), Point( 0, 0, 0 ) } );
        KARMAFEBase collapsed_face( collapsed, GeometryVariant::Tri3, p1_basis( GeometryVariant::Tri3 ), 2, FEIntegrationDomain::Face );
        unsigned collapsed_side = collapsed.n_sides( collapsed_element );
        for( unsigned side = 0; side < collapsed.n_sides( collapsed_element ); ++side )
        {
            const auto nodes = collapsed.side_vertices( collapsed_element, side );
            double first_coordinate[3]{};
            double second_coordinate[3]{};
            check( collapsed.moab()->get_coords( &nodes[0], 1, first_coordinate ) == moab::MB_SUCCESS,
                   "query first collapsed-side coordinate" );
            check( collapsed.moab()->get_coords( &nodes[1], 1, second_coordinate ) == moab::MB_SUCCESS,
                   "query second collapsed-side coordinate" );
            if( ( Point( first_coordinate[0], first_coordinate[1], first_coordinate[2] ) -
                  Point( second_coordinate[0], second_coordinate[1], second_coordinate[2] ) )
                    .norm() == 0.0 )
                collapsed_side = side;
        }
        check( collapsed_side < collapsed.n_sides( collapsed_element ), "locate collapsed face" );
        check( throws( [&] { collapsed_face.reinit( collapsed_element, collapsed_side ); } ),
               "collapsed face is rejected" );
    }

    if( size == 1 )
    {
        KARMAMesh mesh( 2, PETSC_COMM_WORLD );
        const double coords[] = { 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 2, 0 };
        moab::Range vertices;
        check( mesh.moab()->create_vertices( coords, 5, vertices ) == moab::MB_SUCCESS,
               "create non-manifold vertices" );
        std::vector< moab::EntityHandle > v( vertices.begin(), vertices.end() );
        const int indices[][3] = { { 0, 1, 2 }, { 1, 0, 3 }, { 0, 1, 4 } };
        for( const auto& triangle : indices )
        {
            const moab::EntityHandle conn[] = { v[triangle[0]], v[triangle[1]], v[triangle[2]] };
            moab::EntityHandle element      = 0;
            check( mesh.moab()->create_element( moab::MBTRI, conn, 3, element ) == moab::MB_SUCCESS,
                   "create non-manifold triangle" );
        }
        mesh.assign_ids();
        mesh.update();
        const auto element   = mesh.elements().front();
        unsigned shared_side = mesh.n_sides( element );
        for( unsigned side = 0; side < mesh.n_sides( element ); ++side )
            if( mesh.side_vertices( element, side )[0] == v[0] || mesh.side_vertices( element, side )[0] == v[1] )
            {
                const auto side_vertices = mesh.side_vertices( element, side );
                if( std::find( side_vertices.begin(), side_vertices.end(), v[0] ) != side_vertices.end() &&
                    std::find( side_vertices.begin(), side_vertices.end(), v[1] ) != side_vertices.end() )
                    shared_side = side;
            }
        check( shared_side < mesh.n_sides( element ), "locate non-manifold side" );
        check( throws( [&] { (void)mesh.neighbor( element, shared_side ); } ), "non-manifold side is rejected" );
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
