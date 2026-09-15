// KARMA: K(C)ode for Analysis of Reactor and other Multiphysics Applications
// Copyright (C) 2007-2011  Vijay S Mahadevan
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published
// by the Free Software Foundation; either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program; if not, write to the Free Software Foundation,
// Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
//
// Tests for the MOAB-backed mesh that replaced the libMesh-derived one.

#include "karma_exceptions.h"
#include "karma_mesh.h"
#include "karma_moab_io.h"

#include "MBTagConventions.hpp"

#include <petsc.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>

static_assert(!std::is_move_constructible_v<KARMA::KARMAMesh>);
static_assert(!std::is_move_assignable_v<KARMA::KARMAMesh>);

namespace
{
	int failures = 0;

	void check (bool ok, const std::string& what)
	{
		if (!ok)
		{
			std::cerr << "FAIL: " << what << std::endl;
			++failures;
		}
	}

	void equal (std::size_t got, std::size_t expected, const std::string& what)
	{
		if (got != expected)
		{
			std::cerr << "FAIL: " << what
			          << " (got " << got << ", expected " << expected << ")" << std::endl;
			++failures;
		}
	}

	void close (KARMA::Real a, KARMA::Real b, const std::string& what,
	            KARMA::Real tol = 1.e-12)
	{
		if (std::abs(a - b) > tol)
		{
			std::cerr << "FAIL: " << what
			          << " (got " << a << ", expected " << b << ")" << std::endl;
			++failures;
		}
	}

    std::vector< int > conventional_side_ids( const KARMA::KARMAMesh& mesh, moab::EntityHandle element,
                                              unsigned side )
    {
        moab::Tag global_id = nullptr;
        check( mesh.moab()->tag_get_handle( "GLOBAL_ID", global_id ) == moab::MB_SUCCESS,
               "partition fixture has GLOBAL_ID" );
        const auto vertices = mesh.side_vertices( element, side );
        std::vector< int > ids( vertices.size() );
        check( mesh.moab()->tag_get_data( global_id, vertices.data(), static_cast< int >( vertices.size() ), ids.data() ) ==
                   moab::MB_SUCCESS,
               "read GLOBAL_ID side identity" );
        return ids;
    }

    int partitioned_neighbor_test( const std::string& filename, bool without_ghosts )
    {
        using namespace KARMA;
        failures = 0;
        KARMAMesh mesh( 2, PETSC_COMM_WORLD );
        KARMAMoabIO( mesh ).read_mesh(
            filename, "PARALLEL=READ_PART;PARTITION=PARALLEL_PARTITION;PARALLEL_RESOLVE_SHARED_ENTS" );
        if( !without_ghosts )
            check( mesh.parallel_comm()->exchange_ghost_cells( 2, 0, 1, 1, true ) == moab::MB_SUCCESS,
                   "exchange one layer of neighboring elements" );
        mesh.update();

        bool rejected = false;
        try
        {
            mesh.validate_owned_side_topology();
        }
        catch( const InvalidArgument& error )
        {
            rejected = std::string_view( error.what() ).find( "ghost" ) != std::string_view::npos;
        }
        if( without_ghosts )
        {
            check( rejected, "missing partition-interface ghost is rejected collectively" );
        }
        else
        {
            check( !rejected, "partition fixture with ghosts has valid owned-side topology" );
            std::size_t local_physical = 0;
            bool saw_owned_ghost_neighbor = false;
            for( const auto element : mesh.owned_elements() )
                for( unsigned side = 0; side < mesh.n_sides( element ); ++side )
                {
                    const auto neighbor = mesh.neighbor( element, side );
                    if( !neighbor )
                    {
                        ++local_physical;
                        check( mesh.is_boundary_side( element, side ), "unmatched physical side is a boundary" );
                        continue;
                    }
                    const auto local = conventional_side_ids( mesh, element, side );
                    bool reciprocal = false;
                    for( unsigned neighbor_side = 0; neighbor_side < mesh.n_sides( *neighbor ); ++neighbor_side )
                    {
                        const auto remote = conventional_side_ids( mesh, *neighbor, neighbor_side );
                        auto local_identity = local;
                        auto remote_identity = remote;
                        std::sort( local_identity.begin(), local_identity.end() );
                        std::sort( remote_identity.begin(), remote_identity.end() );
                        if( local_identity == remote_identity )
                        {
                            reciprocal = true;
                            check( mesh.neighbor( *neighbor, neighbor_side ) == element,
                                   "owned-to-ghost neighbor relation is reciprocal" );
                            break;
                        }
                    }
                    check( reciprocal, "shared side has matching canonical identity and reciprocal neighbor" );
                    if( mesh.owned_elements().find( *neighbor ) == mesh.owned_elements().end() )
                        saw_owned_ghost_neighbor = true;
                }
            std::size_t global_physical = 0;
            MPI_Allreduce( &local_physical, &global_physical, 1, MPI_UNSIGNED_LONG, MPI_SUM, PETSC_COMM_WORLD );
            check( global_physical == 16, "only the 4x4 fixture exterior sides are physical boundaries" );
            int local_saw_ghost = saw_owned_ghost_neighbor ? 1 : 0;
            int global_saw_ghost = 0;
            MPI_Allreduce( &local_saw_ghost, &global_saw_ghost, 1, MPI_INT, MPI_MAX, PETSC_COMM_WORLD );
            check( global_saw_ghost == 1, "fixture exercises an owned-to-ghost neighbor" );

            moab::Tag global_id = nullptr;
            check( mesh.moab()->tag_get_handle( "GLOBAL_ID", global_id ) == moab::MB_SUCCESS,
                   "partition fixture has authoritative GLOBAL_ID for repair" );
            std::vector< int > invalid_ids( mesh.owned_elements().size(), 0 );
            if( !mesh.owned_elements().empty() )
                check( mesh.moab()->tag_set_data( global_id, mesh.owned_elements(), invalid_ids.data() ) == moab::MB_SUCCESS,
                       "test can invalidate owned element GLOBAL_ID values" );
            bool repair_rejected = false;
            try { mesh.validate_owned_side_topology(); }
            catch( const InvalidArgument& ) { repair_rejected = true; }
            check( !repair_rejected, "side topology validation repairs invalid partitioned GLOBAL_ID values collectively" );
            invalid_ids.resize( mesh.owned_elements().size() );
            if( !mesh.owned_elements().empty() )
                check( mesh.moab()->tag_get_data( global_id, mesh.owned_elements(), invalid_ids.data() ) == moab::MB_SUCCESS,
                       "read repaired owned element GLOBAL_ID values" );
            check( std::all_of( invalid_ids.begin(), invalid_ids.end(), []( int id ) { return id > 0; } ),
                   "partitioned GLOBAL_ID repair restores positive owned element values" );
        }

        int global_failures = 0;
        MPI_Allreduce( &failures, &global_failures, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD );
        return global_failures == 0 ? 0 : 1;
    }

    int empty_partition_global_id_repair_test()
    {
        using namespace KARMA;
        failures = 0;
        int rank = 0;
        int ranks = 0;
        MPI_Comm_rank( PETSC_COMM_WORLD, &rank );
        MPI_Comm_size( PETSC_COMM_WORLD, &ranks );
        check( ranks == 2, "empty GLOBAL_ID repair fixture requires two ranks" );

        KARMAMesh mesh( 1, PETSC_COMM_WORLD );
        moab::Tag global_id = nullptr;
        check( mesh.moab()->tag_get_handle( "GLOBAL_ID", 1, moab::MB_TYPE_INTEGER, global_id,
                                             moab::MB_TAG_DENSE | moab::MB_TAG_CREAT ) == moab::MB_SUCCESS,
               "create invalid existing GLOBAL_ID tag on every rank" );
        if( rank == 0 )
        {
            const double coordinates[] = { 0, 0, 0, 1, 0, 0 };
            moab::Range vertices;
            check( mesh.moab()->create_vertices( coordinates, 2, vertices ) == moab::MB_SUCCESS,
                   "create nonempty GLOBAL_ID repair vertices" );
            const std::array< moab::EntityHandle, 2 > connectivity{ vertices[0], vertices[1] };
            moab::EntityHandle edge = 0;
            check( mesh.moab()->create_element( moab::MBEDGE, connectivity.data(), 2, edge ) == moab::MB_SUCCESS,
                   "create nonempty GLOBAL_ID repair element" );
        }

        mesh.update();
        mesh.resolve_shared_entities( 0 );
        mesh.ensure_global_ids();
        std::vector< int > ids( mesh.vertices().size() );
        if( !mesh.vertices().empty() )
            check( mesh.moab()->tag_get_data( global_id, mesh.vertices(), ids.data() ) == moab::MB_SUCCESS,
                   "read repaired empty-partition GLOBAL_ID values" );
        check( std::all_of( ids.begin(), ids.end(), []( int id ) { return id > 0; } ),
               "empty-partition repair produces positive GLOBAL_ID values" );
        int global_failures = 0;
        MPI_Allreduce( &failures, &global_failures, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD );
        return global_failures == 0 ? 0 : 1;
    }

    int reserved_validation_tag_test()
    {
        using namespace KARMA;
        failures = 0;
        int rank = 0;
        int ranks = 0;
        MPI_Comm_rank( PETSC_COMM_WORLD, &rank );
        MPI_Comm_size( PETSC_COMM_WORLD, &ranks );
        check( ranks == 2, "reserved validation tag fixture requires two ranks" );

        KARMAMesh mesh( 1, PETSC_COMM_WORLD );
        moab::Tag global_id = nullptr;
        check( mesh.moab()->tag_get_handle( "GLOBAL_ID", 1, moab::MB_TYPE_INTEGER, global_id,
                                             moab::MB_TAG_DENSE | moab::MB_TAG_CREAT ) == moab::MB_SUCCESS,
               "create invalid GLOBAL_ID for reserved tag test" );
        constexpr int validation_sentinel = 31337;
        moab::Tag user_validation_tag = nullptr;
        check( mesh.moab()->tag_get_handle( "__karma_global_id_validation__", 1, moab::MB_TYPE_INTEGER,
                                             user_validation_tag, moab::MB_TAG_DENSE | moab::MB_TAG_CREAT,
                                             &validation_sentinel ) == moab::MB_SUCCESS,
               "create pre-existing validation-named user tag" );
        if( rank == 0 )
        {
            const double coordinates[] = { 0, 0, 0, 1, 0, 0 };
            moab::Range vertices;
            check( mesh.moab()->create_vertices( coordinates, 2, vertices ) == moab::MB_SUCCESS,
                   "create nonempty GLOBAL_ID repair vertices" );
            const std::array< moab::EntityHandle, 2 > connectivity{ vertices[0], vertices[1] };
            moab::EntityHandle edge = 0;
            check( mesh.moab()->create_element( moab::MBEDGE, connectivity.data(), 2, edge ) == moab::MB_SUCCESS,
                   "create nonempty GLOBAL_ID repair element" );
        }

        mesh.update();
        mesh.resolve_shared_entities( 0 );
        std::vector< moab::Tag > tags_before;
        check( mesh.moab()->tag_get_tags( tags_before ) == moab::MB_SUCCESS,
               "inventory tags before reserved validation-name rejection" );
        std::vector< int > ids_before( mesh.vertices().size() );
        if( !mesh.vertices().empty() )
            check( mesh.moab()->tag_get_data( global_id, mesh.vertices(), ids_before.data() ) == moab::MB_SUCCESS,
                   "read GLOBAL_ID values before reserved tag rejection" );
        bool rejected = false;
        try { mesh.ensure_global_ids(); }
        catch( const InvalidArgument& error )
        {
            rejected = std::string_view( error.what() ).find( "reserved" ) != std::string_view::npos;
        }
        check( rejected, "GLOBAL_ID repair rejects a pre-existing reserved validation tag collectively" );
        moab::Tag preserved_validation_tag = nullptr;
        check( mesh.moab()->tag_get_handle( "__karma_global_id_validation__", preserved_validation_tag ) ==
                   moab::MB_SUCCESS && preserved_validation_tag == user_validation_tag,
               "GLOBAL_ID repair rejection preserves the reserved user tag" );
        int preserved_sentinel = 0;
        if( !mesh.vertices().empty() )
            check( mesh.moab()->tag_get_data( preserved_validation_tag, &mesh.vertices().front(), 1,
                                                &preserved_sentinel ) == moab::MB_SUCCESS &&
                       preserved_sentinel == validation_sentinel,
                   "GLOBAL_ID repair rejection preserves reserved user values" );
        std::vector< moab::Tag > tags_after;
        check( mesh.moab()->tag_get_tags( tags_after ) == moab::MB_SUCCESS && tags_after == tags_before,
               "GLOBAL_ID repair rejection leaves the tag inventory unchanged" );
        std::vector< int > rejected_ids( mesh.vertices().size() );
        if( !mesh.vertices().empty() )
            check( mesh.moab()->tag_get_data( global_id, mesh.vertices(), rejected_ids.data() ) == moab::MB_SUCCESS &&
                       rejected_ids == ids_before,
                   "GLOBAL_ID repair rejection leaves invalid values unchanged" );

        if( rank == 0 )
        {
            std::vector< int > valid_vertex_ids( mesh.vertices().size() );
            std::iota( valid_vertex_ids.begin(), valid_vertex_ids.end(), 1 );
            check( mesh.moab()->tag_set_data( global_id, mesh.vertices(), valid_vertex_ids.data() ) == moab::MB_SUCCESS,
                   "install valid vertex GLOBAL_ID values" );
            const int valid_element_id = 1;
            check( mesh.moab()->tag_set_data( global_id, mesh.elements(), &valid_element_id ) == moab::MB_SUCCESS,
                   "install valid element GLOBAL_ID value" );
        }
        rejected = false;
        try { mesh.ensure_global_ids(); }
        catch( const InvalidArgument& error )
        {
            rejected = std::string_view( error.what() ).find( "reserved" ) != std::string_view::npos;
        }
        check( rejected, "valid GLOBAL_ID validation rejects a pre-existing reserved tag collectively" );
        tags_after.clear();
        check( mesh.moab()->tag_get_tags( tags_after ) == moab::MB_SUCCESS && tags_after == tags_before,
               "valid GLOBAL_ID rejection leaves the tag inventory unchanged" );
        int global_failures = 0;
        MPI_Allreduce( &failures, &global_failures, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD );
        return global_failures == 0 ? 0 : 1;
    }

    int collective_nonmanifold_test()
    {
        using namespace KARMA;
        failures = 0;
        int rank = 0;
        int ranks = 0;
        MPI_Comm_rank( PETSC_COMM_WORLD, &rank );
        MPI_Comm_size( PETSC_COMM_WORLD, &ranks );
        check( ranks == 2, "distributed non-manifold fixture requires two ranks" );
        KARMAMesh mesh( 2, PETSC_COMM_WORLD );
        const double rank_zero_coordinates[] = { 0, 0, 0, 1, 0, 0, 0, 1, 0 };
        const double rank_one_coordinates[] = { 0, 0, 0, 1, 0, 0, 0, -1, 0, 0.5, 1, 0 };
        moab::Range vertices;
        check( mesh.moab()->create_vertices( rank == 0 ? rank_zero_coordinates : rank_one_coordinates,
                                             rank == 0 ? 3 : 4, vertices ) == moab::MB_SUCCESS,
               "create distributed non-manifold vertices" );
        moab::Tag global_id = nullptr;
        check( mesh.moab()->tag_get_handle( "GLOBAL_ID", 1, moab::MB_TYPE_INTEGER, global_id,
                                             moab::MB_TAG_DENSE | moab::MB_TAG_CREAT ) == moab::MB_SUCCESS,
               "create distributed non-manifold GLOBAL_ID tag" );
        const std::vector< int > vertex_ids = rank == 0 ? std::vector< int >{ 1, 2, 3 }
                                                         : std::vector< int >{ 1, 2, 4, 5 };
        check( mesh.moab()->tag_set_data( global_id, vertices, vertex_ids.data() ) == moab::MB_SUCCESS,
               "assign matching distributed non-manifold GLOBAL_ID values" );
        moab::EntityHandle partition = 0;
        check( mesh.moab()->create_meshset( moab::MESHSET_SET, partition ) == moab::MB_SUCCESS,
               "create distributed non-manifold partition set" );
        moab::Range local_elements;
        moab::EntityHandle triangle = 0;
        if( rank == 0 )
        {
            const std::array< moab::EntityHandle, 3 > source{ vertices[0], vertices[1], vertices[2] };
            check( mesh.moab()->create_element( moab::MBTRI, source.data(), 3, triangle ) == moab::MB_SUCCESS,
                   "create rank-zero source triangle" );
            local_elements.insert( triangle );
        }
        else
        {
            const std::array< moab::EntityHandle, 3 > first{ vertices[1], vertices[0], vertices[2] };
            const std::array< moab::EntityHandle, 3 > second{ vertices[0], vertices[1], vertices[3] };
            check( mesh.moab()->create_element( moab::MBTRI, first.data(), 3, triangle ) == moab::MB_SUCCESS,
                   "create first rank-one non-manifold triangle" );
            local_elements.insert( triangle );
            check( mesh.moab()->create_element( moab::MBTRI, second.data(), 3, triangle ) == moab::MB_SUCCESS,
                   "create second rank-one non-manifold triangle" );
            local_elements.insert( triangle );
        }
        check( mesh.moab()->add_entities( partition, local_elements ) == moab::MB_SUCCESS,
               "add local triangles to distributed non-manifold partition set" );
        check( mesh.parallel_comm()->resolve_shared_ents( partition, 2, -1, &global_id ) == moab::MB_SUCCESS,
               "resolve shared vertices from distributed GLOBAL_ID values" );
        check( mesh.parallel_comm()->exchange_ghost_cells( 2, 0, 1, 0, true ) == moab::MB_SUCCESS,
               "exchange distributed non-manifold ghost triangles" );
        mesh.update();

        if( rank == 0 )
        {
            check( mesh.owned_elements().size() == 1, "rank zero owns the source triangle" );
            moab::Range ghosts;
            check( mesh.parallel_comm()->get_pstatus_entities( 2, PSTATUS_GHOST, ghosts ) == moab::MB_SUCCESS,
                   "query distributed non-manifold ghost triangles" );
            check( ghosts.size() == 2, "rank zero receives both non-manifold neighbor candidates as ghosts" );

            const moab::EntityHandle source = *mesh.owned_elements().begin();
            unsigned shared_side = mesh.n_sides( source );
            for( unsigned side = 0; side < mesh.n_sides( source ); ++side )
            {
                auto ids = conventional_side_ids( mesh, source, side );
                std::sort( ids.begin(), ids.end() );
                if( ids == std::vector< int >{ 1, 2 } )
                {
                    shared_side = side;
                    break;
                }
            }
            check( shared_side < mesh.n_sides( source ), "locate the distributed non-manifold source side" );
            std::size_t candidates = 0;
            if( shared_side < mesh.n_sides( source ) )
            {
                const auto side_ids = conventional_side_ids( mesh, source, shared_side );
                for( const auto candidate : mesh.elements() )
                {
                    if( candidate == source ) continue;
                    const moab::EntityHandle* connectivity = nullptr;
                    int count = 0;
                    check( mesh.moab()->get_connectivity( candidate, connectivity, count ) == moab::MB_SUCCESS,
                           "read distributed candidate connectivity" );
                    std::vector< int > candidate_ids( static_cast< std::size_t >( count ) );
                    check( mesh.moab()->tag_get_data( global_id, connectivity, count, candidate_ids.data() ) ==
                               moab::MB_SUCCESS,
                           "read distributed candidate GLOBAL_ID values" );
                    if( std::all_of( side_ids.begin(), side_ids.end(), [&]( int id ) {
                            return std::find( candidate_ids.begin(), candidate_ids.end(), id ) != candidate_ids.end();
                        } ) )
                        ++candidates;
                }
            }
            check( candidates == 2, "source side sees two locally ghost-visible neighbor candidates" );
        }

        bool rejected = false;
        try { mesh.validate_owned_side_topology(); }
        catch( const InvalidArgument& error )
        {
            const std::string_view message( error.what() );
            rejected = message.find( "rank 0:" ) != std::string_view::npos &&
                       message.find( "non-manifold side has multiple neighbors" ) != std::string_view::npos;
        }
        check( rejected, "distributed non-manifold topology is rejected collectively from rank zero" );
        int global_failures = 0;
        MPI_Allreduce( &failures, &global_failures, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD );
        return global_failures == 0 ? 0 : 1;
    }
}

int main (int argc, char** argv)
{
	PetscErrorCode ierr = PetscInitialize(&argc, &argv, NULL, NULL);
	if (ierr) return ierr;
    if( argc == 3 && std::string_view( argv[1] ) == "--partitioned-neighbors" )
    {
        const int result = partitioned_neighbor_test( argv[2], false );
        PetscFinalize();
        return result;
    }
    if( argc == 3 && std::string_view( argv[1] ) == "--partitioned-neighbors-without-ghosts" )
    {
        const int result = partitioned_neighbor_test( argv[2], true );
        PetscFinalize();
        return result;
    }
    if( argc == 2 && std::string_view( argv[1] ) == "--collective-nonmanifold" )
    {
        const int result = collective_nonmanifold_test();
        PetscFinalize();
        return result;
    }
    if( argc == 2 && std::string_view( argv[1] ) == "--empty-partition-global-id-repair" )
    {
        const int result = empty_partition_global_id_repair_test();
        PetscFinalize();
        return result;
    }
    if( argc == 2 && std::string_view( argv[1] ) == "--reserved-global-id-validation-tag" )
    {
        const int result = reserved_validation_tag_test();
        PetscFinalize();
        return result;
    }

	using namespace KARMA;

	int n_ranks = 1;
	MPI_Comm_size(PETSC_COMM_WORLD, &n_ranks);

	// ---- 3-D structured box --------------------------------------------
	{
		KARMAMesh mesh(3, PETSC_COMM_WORLD);
		mesh.generate_box(4, 4, 4);

		equal(mesh.n_elements(), 64,  "4x4x4 box element count");
		equal(mesh.n_vertices(), 125, "4x4x4 box vertex count");   // 5^3
		equal(mesh.dimension(),  3,   "box dimension");

		// The unit cube is the default extent.
		const BoundingBox box = mesh.bounding_box();
		close(box.min()(0), 0.0, "box min x");
		close(box.max()(2), 1.0, "box max z");
		close(box.extent()(1), 1.0, "box extent y");

		// Every element of a 4^3 unit box is a 0.25-cube.
		const moab::Range& elems = mesh.elements();
		check(!elems.empty(), "element range non-empty");

		Real buf[3 * 27];
		const std::size_t nconn = mesh.element_coordinates(*elems.begin(),
		                                                   std::span<Real>(buf, 3 * 27));
		equal(nconn, 8, "hex has 8 vertices");

		// Centroids must all fall inside the domain.
		for (const moab::EntityHandle e : elems)
			check(box.contains(mesh.centroid(e), 1.e-12), "centroid inside bounding box");
	}

	// ---- anisotropic box over a non-unit interval -----------------------
	{
		KARMAMesh mesh(3, PETSC_COMM_WORLD);
		mesh.generate_box(2, 3, 4, Point(-1.0, 0.0, 2.0), Point(1.0, 3.0, 6.0));

		equal(mesh.n_elements(), 24, "2x3x4 element count");
		equal(mesh.n_vertices(), 60, "2x3x4 vertex count");        // 3*4*5

		const BoundingBox box = mesh.bounding_box();
		close(box.min()(0), -1.0, "aniso min x");
		close(box.max()(0),  1.0, "aniso max x");
		close(box.max()(1),  3.0, "aniso max y");
		close(box.min()(2),  2.0, "aniso min z");
		close(box.max()(2),  6.0, "aniso max z");

		// Total volume must equal the analytic 2 * 3 * 4.
		close(box.extent()(0) * box.extent()(1) * box.extent()(2), 24.0,
		      "anisotropic box volume", 1.e-10);
	}

	// ---- 2-D ------------------------------------------------------------
	{
		KARMAMesh mesh(2, PETSC_COMM_WORLD);
		mesh.generate_box(3, 3, 0);

		equal(mesh.n_elements(), 9,  "3x3 quad count");
		equal(mesh.n_vertices(), 16, "3x3 vertex count");
		equal(mesh.dimension(),  2,  "2-D dimension");
	}

	// ---- HDF5 round-trip -------------------------------------------------
        // Only on one rank: a parallel write needs a partition with stable identities.
	if (n_ranks == 1)
	{
		const std::string file = "karma_mesh_test.h5m";

		std::size_t n_elem = 0, n_vert = 0;
		BoundingBox original;
		{
			KARMAMesh mesh(3, PETSC_COMM_WORLD);
			mesh.generate_box(3, 3, 3);
			n_elem   = mesh.n_elements();
			n_vert   = mesh.n_vertices();
			original = mesh.bounding_box();
			mesh.write(file);
		}

		{
			KARMAMesh reread(3, PETSC_COMM_WORLD);
			reread.read(file);

			equal(reread.n_elements(), n_elem, "h5m round-trip element count");
			equal(reread.n_vertices(), n_vert, "h5m round-trip vertex count");

			const BoundingBox box = reread.bounding_box();
			close(box.min()(0), original.min()(0), "h5m round-trip min x");
			close(box.max()(2), original.max()(2), "h5m round-trip max z");
		}

		std::remove(file.c_str());

		// VTK, the other format the physics output path will use.
		const std::string vtk = "karma_mesh_test.vtk";
		{
			KARMAMesh mesh(3, PETSC_COMM_WORLD);
			mesh.generate_box(2, 2, 2);
			mesh.write(vtk);
		}
		{
			KARMAMesh reread(3, PETSC_COMM_WORLD);
			reread.read(vtk);
			equal(reread.n_elements(), 8, "vtk round-trip element count");
		}
		std::remove(vtk.c_str());
	}

	// ---- parallel bookkeeping -------------------------------------------
	{
		KARMAMesh mesh(3, PETSC_COMM_WORLD);
		mesh.generate_box(4, 4, 4);

		check(mesh.n_ranks() == n_ranks, "mesh reports the communicator size");
		check(mesh.rank() >= 0 && mesh.rank() < n_ranks, "mesh reports a valid rank");

		// Without a partition every rank holds the whole mesh, so owned ==
                // all. This is the invariant used by the replicated-mesh path.
		equal(mesh.owned_elements().size(), mesh.elements().size(),
		      "unpartitioned mesh owns all its elements");
	}

	// ---- private id tags -------------------------------------------------
	{
		KARMAMesh mesh(3, PETSC_COMM_WORLD);
		mesh.generate_box(2, 2, 2);

		std::vector<GlobalId> gids;
		mesh.global_ids(mesh.vertices(), gids);
		equal(gids.size(), 27, "global id count matches vertex count");

		// Every vertex numbered, none left at the -1 sentinel.
		for (const GlobalId g : gids)
			check(g >= 0, "vertex carries a global id");

		if (n_ranks == 1)
		{
			// Serial: ids are a permutation of [0, n).
			std::vector<GlobalId> sorted = gids;
			std::sort(sorted.begin(), sorted.end());
			for (std::size_t i = 0; i < sorted.size(); ++i)
				check(sorted[i] == static_cast<GlobalId>(i),
				      "serial global ids are contiguous from zero");
		}

		std::vector<LocalId> lids;
		mesh.local_ids(mesh.vertices(), lids);
		equal(lids.size(), 27, "local id count matches vertex count");

		// Local ids must be distinct; a collision would alias two entities.
		std::vector<LocalId> lsorted = lids;
		std::sort(lsorted.begin(), lsorted.end());
		check(std::adjacent_find(lsorted.begin(), lsorted.end()) == lsorted.end(),
		      "local ids are distinct");

		// The per-entity accessors must agree with the bulk read.
		const moab::EntityHandle v0 = *mesh.vertices().begin();
		check(mesh.global_id(v0) == gids[0], "global_id() agrees with bulk read");
		check(mesh.local_id(v0)  == lids[0], "local_id() agrees with bulk read");
	}

    // ---- conventional GLOBAL_ID assignment -------------------------------
    if (n_ranks == 1)
    {
        {
            KARMAMesh mesh(2, PETSC_COMM_WORLD);
            mesh.generate_box(2, 1, 0);
            const moab::EntityHandle first = mesh.elements().front();
            const moab::EntityHandle second = mesh.elements().back();
            mesh.resolve_shared_entities();

            moab::Tag conventional = nullptr;
            check(mesh.moab()->tag_get_handle("GLOBAL_ID", conventional) == moab::MB_SUCCESS,
                  "find serial nonreplicated conventional GLOBAL_ID tag");
            std::vector<int> duplicate_ids(mesh.vertices().size(), 7);
            duplicate_ids.front() = 0;
            check(mesh.moab()->tag_set_data(conventional, mesh.vertices(), duplicate_ids.data()) == moab::MB_SUCCESS,
                  "plant zero and duplicate serial nonreplicated GLOBAL_ID values");

            const auto shared_side = [&](moab::EntityHandle element, moab::EntityHandle other) {
                const moab::EntityHandle* connectivity = nullptr;
                int connectivity_size = 0;
                check(mesh.moab()->get_connectivity(other, connectivity, connectivity_size) == moab::MB_SUCCESS,
                      "read serial nonreplicated neighbor connectivity");
                for (unsigned side = 0; side < mesh.n_sides(element); ++side)
                {
                    const auto vertices = mesh.side_vertices(element, side);
                    if (std::all_of(vertices.begin(), vertices.end(), [&](moab::EntityHandle vertex) {
                            return std::find(connectivity, connectivity + connectivity_size, vertex) !=
                                   connectivity + connectivity_size;
                        }))
                        return side;
                }
                return mesh.n_sides(element);
            };
            const unsigned first_side = shared_side(first, second);
            const unsigned second_side = shared_side(second, first);
            check(first_side < mesh.n_sides(first) && second_side < mesh.n_sides(second),
                  "locate exact serial nonreplicated shared sides by handle");
            for (unsigned side = 0; side < mesh.n_sides(first); ++side)
                check(mesh.neighbor(first, side) == (side == first_side ? std::optional(second) : std::nullopt),
                      "first serial nonreplicated element has neighbor only on exact shared side");
            for (unsigned side = 0; side < mesh.n_sides(second); ++side)
                check(mesh.neighbor(second, side) == (side == second_side ? std::optional(first) : std::nullopt),
                      "second serial nonreplicated element has neighbor only on exact shared side");
        }

        KARMAMesh mesh(2, PETSC_COMM_WORLD);
        mesh.generate_box(2, 2, 0);

        mesh.ensure_global_ids();
        moab::Tag conventional = nullptr;
        check(mesh.moab()->tag_get_handle("GLOBAL_ID", conventional) == moab::MB_SUCCESS,
              "ensure_global_ids creates the conventional tag");
        for (const moab::Range* entities : { &mesh.vertices(), &mesh.elements() })
        {
            std::vector<int> ids(entities->size());
            mesh.moab()->tag_get_data(conventional, *entities, ids.data());
            std::sort(ids.begin(), ids.end());
            check(!ids.empty() && ids.front() == 1, "conventional ids start at one");
            check(std::adjacent_find(ids.begin(), ids.end()) == ids.end(),
                  "serial conventional ids are unique per dimension");
        }

        std::vector<GlobalId> private_ids;
        mesh.global_ids(mesh.vertices(), private_ids);
        std::vector<int> invalid(mesh.vertices().size(), 0);
        mesh.moab()->tag_set_data(conventional, mesh.vertices(), invalid.data());
        mesh.ensure_global_ids();
        mesh.moab()->tag_get_data(conventional, mesh.vertices(), invalid.data());
        check(std::all_of(invalid.begin(), invalid.end(), [](int id) { return id > 0; }),
              "invalid conventional ids are repaired");
        std::vector<GlobalId> private_ids_after;
        mesh.global_ids(mesh.vertices(), private_ids_after);
        check(private_ids_after == private_ids,
              "conventional assignment leaves __karma_global_id__ unchanged");
    }

	// ---- MOAB's GLOBAL_ID is never overwritten ---------------------------
	// This is the property the private tags exist for: a mesh arriving with
	// its own numbering must keep it.
	{
		KARMAMesh mesh(3, PETSC_COMM_WORLD);
		mesh.generate_box(1, 1, 1);          // one hex, 8 vertices

		moab::Tag conventional;
		int unset = 0;
		mesh.moab()->tag_get_handle("GLOBAL_ID", 1, moab::MB_TYPE_INTEGER,
		                            conventional,
		                            moab::MB_TAG_DENSE | moab::MB_TAG_CREAT, &unset);

		// Stamp a distinctive numbering, as a file reader would.
		std::vector<int> planted(mesh.vertices().size());
		for (std::size_t i = 0; i < planted.size(); ++i)
			planted[i] = static_cast<int>(1000 + i);
		mesh.moab()->tag_set_data(conventional, mesh.vertices(), planted.data());

		// Re-number KARMA's private ids, forcing a rewrite.
		mesh.assign_ids(true);

		std::vector<int> after(mesh.vertices().size());
		mesh.moab()->tag_get_data(conventional, mesh.vertices(), after.data());

		for (std::size_t i = 0; i < after.size(); ++i)
			check(after[i] == planted[i],
			      "assign_ids leaves MOAB's GLOBAL_ID untouched");

		// And the private tag is genuinely a different tag, not an alias.
		std::vector<GlobalId> priv;
		mesh.global_ids(mesh.vertices(), priv);
		bool differs = false;
		for (std::size_t i = 0; i < priv.size(); ++i)
			if (priv[i] != static_cast<GlobalId>(planted[i]))
				differs = true;
		check(differs, "private global ids are independent of GLOBAL_ID");
	}

    // ---- material lookup --------------------------------------------------
    {
        KARMAMesh mesh(2, PETSC_COMM_WORLD);
        mesh.generate_box(1, 1, 0);
        const moab::EntityHandle element = *mesh.elements().begin();
        check(mesh.material_id(element) == 0, "missing material set intentionally maps to zero");

        moab::Tag tag = nullptr;
        mesh.moab()->tag_get_handle("MATERIAL_SET", 1, moab::MB_TYPE_INTEGER, tag,
                                    moab::MB_TAG_SPARSE | moab::MB_TAG_CREAT);
        moab::EntityHandle first = 0;
        mesh.moab()->create_meshset(moab::MESHSET_SET, first);
        mesh.moab()->add_entities(first, &element, 1);
        int id = 7;
        mesh.moab()->tag_set_data(tag, &first, 1, &id);
        check(mesh.material_id(element) == 7, "one containing material set returns its id");

        moab::EntityHandle second = 0;
        mesh.moab()->create_meshset(moab::MESHSET_SET, second);
        mesh.moab()->add_entities(second, &element, 1);
        id = 8;
        mesh.moab()->tag_set_data(tag, &second, 1, &id);
        bool multiple_rejected = false;
        try
        {
            (void)mesh.material_id(element);
        }
        catch (const InvalidArgument&)
        {
            multiple_rejected = true;
        }
        check(multiple_rejected, "multiple containing material sets are rejected");
    }

	// ---- assign_ids is idempotent ----------------------------------------
	{
		KARMAMesh mesh(3, PETSC_COMM_WORLD);
		mesh.generate_box(2, 2, 2);

		std::vector<GlobalId> first;
		mesh.global_ids(mesh.vertices(), first);

		// A second call must not renumber: callers may already hold ids.
		mesh.assign_ids();

		std::vector<GlobalId> second;
		mesh.global_ids(mesh.vertices(), second);

		check(first == second, "assign_ids() is idempotent without force");

		// With force, renumbering is allowed but must stay well-formed.
		mesh.assign_ids(true);
		std::vector<GlobalId> forced;
		mesh.global_ids(mesh.vertices(), forced);
		for (const GlobalId g : forced)
			check(g >= 0, "forced renumber leaves every vertex numbered");
	}

	int rank = 0;
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
	if (rank == 0)
	{
		if (failures == 0)
			std::cout << "mesh tests passed on " << n_ranks << " rank(s)." << std::endl;
		else
			std::cout << "mesh tests: " << failures << " failure(s)." << std::endl;
	}

	ierr = PetscFinalize();
	if (ierr) return ierr;

	return failures == 0 ? 0 : 1;
}
