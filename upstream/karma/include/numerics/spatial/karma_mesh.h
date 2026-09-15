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


#ifndef __karma_mesh_h__
#define __karma_mesh_h__

#include "domain_shape.h"
#include "karma_geometry.h"
#include "karma_types.h"
#include "primary_axis.h"

#include "moab/Core.hpp"
#include "moab/Range.hpp"

#ifdef MOAB_HAVE_MPI
#include "moab/ParallelComm.hpp"
#endif

#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace KARMA
{

    struct SideId
    {
        GlobalId element;
        unsigned side;

        bool operator== (const SideId&) const = default;
    };

	/**
	 * The computational mesh.
	 *
	 * This owns a MOAB database and presents the queries KARMA's numerics
	 * need.  It replaces a class that derived from \p libMesh::Mesh; the
	 * relationship to the mesh library is now composition, so the backend can
	 * be reasoned about (and, if ever necessary, replaced) without changing
	 * the physics layer, which holds \p KARMAMesh by reference.
	 *
	 * Entities are MOAB handles.  Fields are MOAB tags.  Both are opaque
	 * integers, so element loops carry no pointer chasing and no per-element
	 * allocation.
	 *
     * The class is neither copyable nor movable: moab::Core and its bound
     * ParallelComm cannot be transferred safely.  The former class had a copy
     * constructor that shallow-copied a libMesh mesh; construct and populate a
     * distinct mesh explicitly instead.
	 */
	class KARMAMesh
	{
	public:

		/** Construct an empty mesh of the given spatial dimension. */
		explicit KARMAMesh (unsigned dim, MPI_Comm comm = MPI_COMM_WORLD);

		/** Construct and immediately load from a file. */
		KARMAMesh (unsigned dim, std::string_view filename,
		           MPI_Comm comm = MPI_COMM_WORLD);

		~KARMAMesh ();

		KARMAMesh (const KARMAMesh&)            = delete;
		KARMAMesh& operator= (const KARMAMesh&) = delete;
        KARMAMesh (KARMAMesh&&)                 = delete;
        KARMAMesh& operator= (KARMAMesh&&)      = delete;

		// -- backend access ---------------------------------------------------
		//
		// Deliberately public.  MOAB is the mesh representation, not an
		// implementation detail to be hidden: the FE, IO, partitioning and
		// coupling layers all need the raw interface, and wrapping every MOAB
		// call would add surface area without adding safety.

		[[nodiscard]] moab::Interface*       moab ()       noexcept { return &_moab; }
		[[nodiscard]] const moab::Interface* moab () const noexcept { return &_moab; }

#ifdef MOAB_HAVE_MPI
		[[nodiscard]] moab::ParallelComm* parallel_comm () const noexcept { return _pcomm; }
#endif

		[[nodiscard]] moab::EntityHandle root_set () const noexcept { return _root; }

		// -- topology ---------------------------------------------------------

		[[nodiscard]] unsigned dimension () const noexcept { return _dim; }

		[[nodiscard]] std::size_t n_vertices () const;
		[[nodiscard]] std::size_t n_elements () const;

		/**
		 * All elements of the mesh dimension, owned and ghosted.
		 *
		 * Returned by reference to a cached \p Range: assembly loops call this
		 * once and iterate, and the range must not be rebuilt per call.
		 * Invalidated by any topology change; call \p update() after one.
		 */
		[[nodiscard]] const moab::Range& elements () const noexcept { return _elements; }

		/** Elements owned by this rank. Equal to \p elements() in serial. */
		[[nodiscard]] const moab::Range& owned_elements () const noexcept { return _owned_elements; }

		[[nodiscard]] const moab::Range& vertices () const noexcept { return _vertices; }

		/** Re-read topology from MOAB into the cached ranges. */
		void update ();

        [[nodiscard]] unsigned n_sides (moab::EntityHandle element) const;

        /** Fill caller-owned storage in CN order; the span must fit the complete side. */
        [[nodiscard]] std::size_t side_vertices (moab::EntityHandle element,
                                                 unsigned side,
                                                 std::span<moab::EntityHandle> vertices) const;

        /** Allocating convenience overload, dynamically sized from the CN side topology. */
        [[nodiscard]] std::vector<moab::EntityHandle>
        side_vertices (moab::EntityHandle element, unsigned side) const;

        /** Sorted authoritative GLOBAL_ID values for a side's vertices. */
        [[nodiscard]] std::vector<GlobalId>
        canonical_side_vertex_ids (moab::EntityHandle element, unsigned side) const;

        /** Return the visible neighbor, or null only for a physical boundary side. */
        [[nodiscard]] std::optional<moab::EntityHandle>
        neighbor (moab::EntityHandle element, unsigned side) const;

        /** True only when the side has no local or remote neighboring element. */
        [[nodiscard]] bool is_boundary_side (moab::EntityHandle element,
                                              unsigned side) const;

        /** Collectively validate owned-side topology before distributed traversal. */
        void validate_owned_side_topology () const;

        [[nodiscard]] SideId side_id (moab::EntityHandle element, unsigned side) const;

		// -- entity identifiers -----------------------------------------------
		//
		// KARMA numbers entities with its own pair of tags rather than
		// MOAB's conventional GLOBAL_ID.  Two reasons:
		//
		//   * GLOBAL_ID belongs to the file.  A mesh read from disk, or
		//     produced by a generator, arrives with a numbering that means
		//     something to its author; overwriting it loses information the
		//     caller may still need.  KARMA never writes GLOBAL_ID.
		//
		//   * The local numbering is KARMA's own, not a view onto PETSc's.
		//     DMMoab maintains an equivalent mapping internally, but reading
		//     it would make every consumer of a local index depend on PETSc.
		//     Owning the tag keeps that dependency in one place.
		//
		// Both are dense tags, so lookup is an array index.

		static constexpr const char* global_id_tag_name = "__karma_global_id__";
		static constexpr const char* local_id_tag_name  = "__karma_local_id__";

		/**
		 * Number the entities of the mesh in KARMA's private tags.
		 *
		 * Global ids are unique across ranks and drive DoF ordering; local
		 * ids are contiguous within a rank and cover ghosts.  MOAB's
		 * \p GLOBAL_ID is left exactly as found.
		 *
		 * Idempotent: entities that already carry a valid private id keep it
		 * unless \p force is set, so re-numbering after a ghost exchange does
		 * not invalidate ids the caller has already read.
		 *
		 * Called automatically by \p generate_box() and \p read().
		 */
		void assign_ids (bool force = false);

        /** Ensure MOAB's conventional GLOBAL_ID identifies partitioned entities. */
        void ensure_global_ids () const;

		[[nodiscard]] moab::Tag global_id_tag () const noexcept { return _global_id_tag; }
		[[nodiscard]] moab::Tag local_id_tag  () const noexcept { return _local_id_tag;  }

		[[nodiscard]] GlobalId global_id (moab::EntityHandle entity) const;
		[[nodiscard]] LocalId  local_id  (moab::EntityHandle entity) const;

		/** Bulk read, which is what loops should use. */
		void global_ids (const moab::Range& entities, std::vector<GlobalId>& ids) const;
		void local_ids  (const moab::Range& entities, std::vector<LocalId>&  ids) const;

		// -- geometry ---------------------------------------------------------

		/** Vertex coordinates, packed as consecutive (x,y,z) triples. */
		void vertex_coordinates (const moab::Range& verts,
		                         std::vector<Real>& coords) const;

		/**
		 * Coordinates of one element's vertices, written into \p coords.
		 *
		 * @returns the number of vertices written.  The caller supplies the
		 * buffer so that an assembly loop allocates nothing; 27 entries (81
		 * reals) covers every element type MOAB can evaluate.
		 */
		std::size_t element_coordinates (moab::EntityHandle elem,
		                                 std::span<Real> coords) const;

		[[nodiscard]] Point centroid (moab::EntityHandle elem) const;

		[[nodiscard]] BoundingBox bounding_box () const;

		// -- construction and I/O ---------------------------------------------

		/**
		 * Generate a structured box, \p nx by \p ny by \p nz elements.
		 *
		 * Uses MOAB's structured-mesh interface, which stores the block
		 * implicitly: neighbour queries are arithmetic rather than lookups,
		 * and the vertex and element handles are contiguous.
		 */
		void generate_box (unsigned nx, unsigned ny, unsigned nz,
		                   const Point& lo = Point::Zero(),
		                   const Point& hi = Point::Ones());

		/**
		 * Read a mesh.  Format is inferred from the extension: \p .h5m
		 * (MOAB HDF5, parallel-capable), \p .vtk, \p .msh (Gmsh), \p .cub.
		 */
		void read (std::string_view filename, std::string_view options = "");

		void write (std::string_view filename, std::string_view options = "") const;

		// -- parallel ---------------------------------------------------------

		/**
		 * Establish inter-rank sharing after a partition, then exchange
		 * \p n_ghost_layers of ghost elements.  A no-op on one rank.
		 */
		void resolve_shared_entities (unsigned n_ghost_layers = 1);

		[[nodiscard]] int rank () const noexcept { return _rank; }
		[[nodiscard]] int n_ranks () const noexcept { return _n_ranks; }
        [[nodiscard]] MPI_Comm communicator() const noexcept { return _communicator; }
        [[nodiscard]] bool is_replicated() const noexcept { return _replicated; }

		// -- material and boundary sets ---------------------------------------

		/**
		 * @returns the material (subdomain) id of an element, or 0 if it
		 * carries no material tag.  Replaces libMesh's \p subdomain_id().
		 */
		[[nodiscard]] int material_id (moab::EntityHandle elem) const;

		/** Entity sets tagged MATERIAL_SET. */
		void material_sets (moab::Range& sets) const;

		/** Entity sets tagged NEUMANN_SET or DIRICHLET_SET. */
		void boundary_sets (moab::Range& sets) const;

	private:

		void init_communicator (MPI_Comm comm);
		void create_id_tags ();
        [[nodiscard]] std::optional<moab::EntityHandle>
        neighbor_local (moab::EntityHandle element, unsigned side) const;
        void collective_topology_error (const std::string& local_error) const;

		mutable moab::Core  _moab;
		moab::EntityHandle  _root  = 0;
		unsigned            _dim   = 3;

		moab::Tag _global_id_tag = nullptr;
		moab::Tag _local_id_tag  = nullptr;

#ifdef MOAB_HAVE_MPI
		moab::ParallelComm* _pcomm = nullptr;
#endif
        MPI_Comm _communicator = MPI_COMM_NULL;
		int _rank    = 0;
		int _n_ranks = 1;
        bool _replicated = true;
		moab::Range _vertices;
		moab::Range _elements;
		moab::Range _owned_elements;
	};

}

#endif // __karma_mesh_h__
