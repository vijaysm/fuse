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


#include "karma_mesh.h"

#include "karma_dof_map.h"
#include "karma_error.h"
#include "karma_exceptions.h"
#include "../../physics/common/impls/collective_signature.h"

#include "MBTagConventions.hpp"
#include "moab/CN.hpp"
#include "moab/HomXform.hpp"
#include "moab/MeshTopoUtil.hpp"
#include "moab/ScdInterface.hpp"

#include <algorithm>
#include <array>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace
{
	/**
	 * Convert a MOAB return code into a KARMA exception.
	 *
	 * MOAB reports failures by return value.  Letting one pass unchecked
	 * would leave the database partially modified and surface as a confusing
	 * fault much later, so every call is checked where it is made.
	 */
	inline void check (moab::ErrorCode rval, const char* what)
	{
		if (rval != moab::MB_SUCCESS)
			karma_exception("MOAB error in " << what
			                << " (code " << static_cast<int>(rval) << ")");
	}

    class TemporaryTag
    {
      public:
        explicit TemporaryTag(moab::Interface& database) : _database(database) {}
        ~TemporaryTag() noexcept
        {
            if (_created)
                (void)_database.tag_delete(_tag);
        }
        TemporaryTag(const TemporaryTag&) = delete;
        TemporaryTag& operator=(const TemporaryTag&) = delete;

        moab::Tag& put() noexcept { return _tag; }
        moab::Tag get() const noexcept { return _tag; }
        void mark_created() noexcept { _created = true; }
        moab::ErrorCode release() noexcept
        {
            if (!_created)
                return moab::MB_SUCCESS;
            const moab::ErrorCode error = _database.tag_delete(_tag);
            if (error == moab::MB_SUCCESS)
                _created = false;
            return error;
        }

      private:
        moab::Interface& _database;
        moab::Tag _tag = nullptr;
        bool _created = false;
    };
}

namespace KARMA
{

KARMAMesh::KARMAMesh (unsigned dim, MPI_Comm comm)
	: _dim(dim)
{
	_root = 0;   // MOAB names its root set 0 by convention
	init_communicator(comm);
}


KARMAMesh::KARMAMesh (unsigned dim, std::string_view filename, MPI_Comm comm)
	: KARMAMesh(dim, comm)
{
	read(filename);
}


KARMAMesh::~KARMAMesh ()
{
#ifdef MOAB_HAVE_MPI
	delete _pcomm;
	_pcomm = nullptr;
#endif
}


void KARMAMesh::init_communicator (MPI_Comm comm)
{
    _communicator = comm;
#ifdef MOAB_HAVE_MPI
	_pcomm   = new moab::ParallelComm(&_moab, comm);
	_rank    = _pcomm->rank();
	_n_ranks = _pcomm->size();
#else
	(void) comm;
	_rank    = 0;
	_n_ranks = 1;
#endif
}


void KARMAMesh::create_id_tags ()
{
	if (_global_id_tag && _local_id_tag)
		return;

	// -1 marks "not yet numbered", which is what makes assign_ids()
	// idempotent: an entity holding the default is new, anything else has a
	// numbering already worth keeping.
	const GlobalId unset_global = -1;
	const LocalId  unset_local  = -1;

	check(_moab.tag_get_handle(global_id_tag_name,
	                           1, moab::MB_TYPE_INTEGER, _global_id_tag,
	                           moab::MB_TAG_DENSE | moab::MB_TAG_CREAT,
	                           &unset_global),
	      "tag_get_handle(__karma_global_id__)");

	check(_moab.tag_get_handle(local_id_tag_name,
	                           1, moab::MB_TYPE_INTEGER, _local_id_tag,
	                           moab::MB_TAG_DENSE | moab::MB_TAG_CREAT,
	                           &unset_local),
	      "tag_get_handle(__karma_local_id__)");

	static_assert(sizeof(GlobalId) == sizeof(int),
	              "MB_TYPE_INTEGER is native int, and DMMoab reads the id tag as int. "
	              "Widening GlobalId needs a matching MOAB tag type and a PETSc built "
	              "with --with-64-bit-indices.");
	static_assert(sizeof(LocalId) == sizeof(int),
	              "MB_TYPE_INTEGER is native int.");
}


void KARMAMesh::ensure_global_ids () const
{
#ifdef MOAB_HAVE_MPI
    const MPI_Comm comm = _pcomm->comm();
    const auto collective_error = [&](std::string_view local_error)
    {
        if (_n_ranks == 1)
        {
            if (!local_error.empty())
                KARMA_THROW(InvalidArgument(std::string(local_error)));
            return;
        }
        detail::collective_signature_error(comm, local_error);
    };
#else
    const auto collective_error = [](std::string_view local_error)
    {
        if (!local_error.empty())
            KARMA_THROW(InvalidArgument(local_error));
    };
#endif
    const auto collective_prepare = [&](auto&& preparation, const char* failure)
    {
#ifdef MOAB_HAVE_MPI
        if (_n_ranks > 1)
        {
            detail::prepare_collectively(comm, std::forward<decltype(preparation)>(preparation), failure);
            return;
        }
#endif
        try { std::forward<decltype(preparation)>(preparation)(); }
        catch (...) { KARMA_THROW(InvalidArgument(failure)); }
    };

    moab::Tag global_id_tag = nullptr;
    bool tag_present = false;
    bool schema_valid = false;
    std::string_view local_error;
    moab::Range relevant;
    const auto valid_ids = [&]
    {
        global_id_tag = nullptr;
        tag_present = false;
        schema_valid = false;
        local_error = {};
        moab::Tag reserved_validation_tag = nullptr;
        const moab::ErrorCode reserved_lookup =
            _moab.tag_get_handle("__karma_global_id_validation__", reserved_validation_tag);
        if (reserved_lookup == moab::MB_SUCCESS)
            local_error = "reserved MOAB tag __karma_global_id_validation__ already exists";
        else if (reserved_lookup != moab::MB_TAG_NOT_FOUND)
            local_error = "failed to query reserved MOAB GLOBAL_ID validation tag";
        collective_error(local_error);
        const moab::ErrorCode lookup = _moab.tag_get_handle(GLOBAL_ID_TAG_NAME, global_id_tag);
        if (lookup == moab::MB_SUCCESS)
        {
            tag_present = true;
            moab::DataType type = moab::MB_TYPE_OPAQUE;
            int length = 0;
            const moab::ErrorCode type_error = _moab.tag_get_data_type(global_id_tag, type);
            const moab::ErrorCode length_error = _moab.tag_get_length(global_id_tag, length);
            if (type_error != moab::MB_SUCCESS || length_error != moab::MB_SUCCESS)
                local_error = "failed to inspect MOAB GLOBAL_ID schema";
            else if (type != moab::MB_TYPE_INTEGER || length != 1)
                local_error = "MOAB GLOBAL_ID must be a scalar integer tag; its cached tag cannot be replaced safely";
            else
                schema_valid = true;
        }
        else if (lookup != moab::MB_TAG_NOT_FOUND)
            local_error = "failed to query MOAB GLOBAL_ID tag";
        collective_error(local_error);

    int all_present = tag_present ? 1 : 0;
#ifdef MOAB_HAVE_MPI
    if (_n_ranks > 1)
    {
        if (MPI_Allreduce(MPI_IN_PLACE, &all_present, 1, MPI_INT, MPI_MIN, comm) != MPI_SUCCESS)
            KARMA_THROW(InvalidArgument("MPI_Allreduce failed while checking GLOBAL_ID presence"));
    }
#endif
    bool valid = all_present != 0 && schema_valid;
    relevant.clear();
    collective_prepare([&] {
        for (const int dimension : { 0, static_cast<int>(_dim) })
        {
            moab::Range entities;
            const moab::ErrorCode error = _moab.get_entities_by_dimension(_root, dimension, entities);
            if (error != moab::MB_SUCCESS)
            {
                local_error = "failed to query entities for GLOBAL_ID validation";
                break;
            }
            relevant.merge(entities);
        }
    }, "GLOBAL_ID entity query preparation failed");
    collective_error(local_error);

    std::array<moab::Range, 2> owned_by_dimension;
    std::array<std::vector<int>, 2> owned_ids;
    for (std::size_t index = 0; index < owned_by_dimension.size(); ++index)
    {
        const int dimension = index == 0 ? 0 : static_cast<int>(_dim);
        moab::ErrorCode error = moab::MB_SUCCESS;
        collective_prepare([&] {
            error = _moab.get_entities_by_dimension(_root, dimension, owned_by_dimension[index]);
#ifdef MOAB_HAVE_MPI
            if (error == moab::MB_SUCCESS && _n_ranks > 1 && !_replicated)
                error = _pcomm->filter_pstatus(owned_by_dimension[index], PSTATUS_NOT_OWNED, PSTATUS_NOT);
#endif
        }, "owned GLOBAL_ID entity preparation failed");
        if (error == moab::MB_SUCCESS && valid)
        {
            collective_prepare([&] { owned_ids[index].resize(owned_by_dimension[index].size()); },
                               "owned GLOBAL_ID allocation failed");
            if (!owned_by_dimension[index].empty())
                error = _moab.tag_get_data(global_id_tag, owned_by_dimension[index], owned_ids[index].data());
        }
        if (local_error.empty() && error != moab::MB_SUCCESS)
            local_error = "failed to read owned MOAB GLOBAL_ID values";
    }
    collective_error(local_error);

    if (valid)
    {
        for (auto& ids : owned_ids)
        {
            valid = valid && std::all_of(ids.begin(), ids.end(), [](int id) { return id > 0; });
            std::sort(ids.begin(), ids.end());
            valid = valid && std::adjacent_find(ids.begin(), ids.end()) == ids.end();
        }
    }

#ifdef MOAB_HAVE_MPI
    if (_n_ranks > 1)
    {
        int local_valid = valid ? 1 : 0;
        int global_valid = 0;
        if (MPI_Allreduce(&local_valid, &global_valid, 1, MPI_INT, MPI_MIN, comm) != MPI_SUCCESS)
            KARMA_THROW(InvalidArgument("MPI_Allreduce failed while validating GLOBAL_ID values"));
        valid = global_valid != 0;

        if (valid && _replicated)
        {
            for (const auto& ids : owned_ids)
            {
                local_error = {};
                int count = 0;
                try
                {
                    count = detail::checked_mpi_count(ids.size());
                }
                catch (const std::exception&)
                {
                    local_error = "GLOBAL_ID count exceeds MPI range";
                }
                collective_error(local_error);
                std::vector<int> counts;
                detail::prepare_collectively(comm, [&] { counts.resize(static_cast<std::size_t>(_n_ranks)); },
                                             "GLOBAL_ID count allocation failed");
                if (MPI_Allgather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, comm) != MPI_SUCCESS)
                    KARMA_THROW(InvalidArgument("MPI_Allgather failed while comparing replicated GLOBAL_IDs"));
                detail::MpiLayout layout;
                try
                {
                    layout = detail::checked_mpi_displacements(counts, 1);
                }
                catch (const std::exception&)
                {
                    local_error = "GLOBAL_ID count exceeds MPI range";
                }
                collective_error(local_error);
                const bool same_count = std::all_of(counts.begin(), counts.end(), [count](int value) { return value == count; });
                std::vector<int> gathered;
                detail::prepare_collectively(comm, [&] { gathered.resize(layout.total_count); },
                                             "replicated GLOBAL_ID gather allocation failed");
                if (MPI_Allgatherv(ids.data(), count, MPI_INT, gathered.data(), counts.data(),
                                   layout.displacements.data(), MPI_INT, comm) != MPI_SUCCESS)
                    KARMA_THROW(InvalidArgument("MPI_Allgatherv failed while comparing replicated GLOBAL_ID values"));
                for (int source = 0; source < _n_ranks; ++source)
                {
                    const std::size_t offset = static_cast<std::size_t>(layout.displacements[static_cast<std::size_t>(source)]);
                    valid = valid && same_count && std::equal(ids.begin(), ids.end(), gathered.begin() + offset);
                }
            }
        }
        else if (valid)
        {
            for (const auto& ids : owned_ids)
            {
                local_error = {};
                int local_count = 0;
                try
                {
                    local_count = detail::checked_mpi_count(ids.size());
                }
                catch (const std::exception&)
                {
                    local_error = "GLOBAL_ID count exceeds MPI range";
                }
                collective_error(local_error);
                std::vector<int> counts;
                detail::prepare_collectively(comm, [&] { counts.resize(static_cast<std::size_t>(_n_ranks)); },
                                             "GLOBAL_ID uniqueness count allocation failed");
                if (MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, comm) != MPI_SUCCESS)
                    KARMA_THROW(InvalidArgument("MPI_Allgather failed while validating GLOBAL_ID uniqueness"));
                detail::MpiLayout layout;
                try
                {
                    layout = detail::checked_mpi_displacements(counts, 1);
                }
                catch (const std::exception&)
                {
                    local_error = "GLOBAL_ID count exceeds MPI range";
                }
                collective_error(local_error);
                std::vector<int> all_ids;
                detail::prepare_collectively(comm, [&] { all_ids.resize(layout.total_count); },
                                             "GLOBAL_ID uniqueness gather allocation failed");
                if (MPI_Allgatherv(ids.data(), local_count, MPI_INT, all_ids.data(), counts.data(),
                                   layout.displacements.data(), MPI_INT, comm) != MPI_SUCCESS)
                    KARMA_THROW(InvalidArgument("MPI_Allgatherv failed while validating GLOBAL_ID uniqueness"));
                std::sort(all_ids.begin(), all_ids.end());
                valid = valid && std::adjacent_find(all_ids.begin(), all_ids.end()) == all_ids.end();
            }

            std::vector<int> original;
            std::vector<int> owner_values;
            collective_prepare([&] {
                original.resize(relevant.size());
                owner_values.resize(relevant.size());
            }, "GLOBAL_ID exchange buffer allocation failed");
            const int zero = 0;
            TemporaryTag validation_tag(_moab);
            const moab::ErrorCode create_error = _moab.tag_get_handle("__karma_global_id_validation__", 1,
                moab::MB_TYPE_INTEGER, validation_tag.put(), moab::MB_TAG_DENSE | moab::MB_TAG_CREAT, &zero);
            if (create_error == moab::MB_SUCCESS)
                validation_tag.mark_created();
            moab::ErrorCode copy_error = create_error;
            if (copy_error == moab::MB_SUCCESS && !relevant.empty())
                copy_error = _moab.tag_get_data(global_id_tag, relevant, original.data());
            if (copy_error == moab::MB_SUCCESS && !relevant.empty())
                copy_error = _moab.tag_set_data(validation_tag.get(), relevant, original.data());
            if (copy_error != moab::MB_SUCCESS)
                local_error = "failed to prepare temporary GLOBAL_ID validation tag";
            collective_error(local_error);
            const moab::ErrorCode exchange_error = _pcomm->exchange_tags(validation_tag.get(), relevant);
            if (exchange_error != moab::MB_SUCCESS)
                local_error = "failed to exchange temporary GLOBAL_ID validation tag";
            collective_error(local_error);
            moab::ErrorCode read_error = moab::MB_SUCCESS;
            if (!relevant.empty())
                read_error = _moab.tag_get_data(validation_tag.get(), relevant, owner_values.data());
            if (read_error != moab::MB_SUCCESS)
                local_error = "failed to read exchanged GLOBAL_ID validation values";
            collective_error(local_error);
            valid = valid && original == owner_values;
            const moab::ErrorCode delete_error = validation_tag.release();
            if (delete_error != moab::MB_SUCCESS)
                local_error = "failed to delete temporary GLOBAL_ID validation tag";
            collective_error(local_error);
        }

        local_valid = valid ? 1 : 0;
        if (MPI_Allreduce(&local_valid, &global_valid, 1, MPI_INT, MPI_MIN, comm) != MPI_SUCCESS)
            KARMA_THROW(InvalidArgument("MPI_Allreduce failed after GLOBAL_ID validation"));
        valid = global_valid != 0;
    }
#endif
        return valid;
    };

    if (valid_ids())
        return;

    std::vector<int> unset;
    collective_prepare([&] {
        if (tag_present && !relevant.empty()) unset.assign(relevant.size(), 0);
    }, "GLOBAL_ID clear buffer allocation failed");
    moab::ErrorCode clear_error = moab::MB_SUCCESS;
    if (tag_present && !relevant.empty())
        clear_error = _moab.tag_set_data(global_id_tag, relevant, unset.data());
    if (clear_error != moab::MB_SUCCESS)
        local_error = "failed to clear invalid MOAB GLOBAL_ID values before reassignment";
    collective_error(local_error);
#ifdef MOAB_HAVE_MPI
    if (_n_ranks > 1 && !_replicated)
    {
        const moab::ErrorCode assign_error = _pcomm->assign_global_ids(
            _root, static_cast<int>(_dim), 1, true, true, false);
        if (assign_error != moab::MB_SUCCESS)
            local_error = "MOAB assign_global_ids failed";
        collective_error(local_error);
    }
    else
#endif
    {
        if (!tag_present)
        {
            const int default_id = 0;
            const moab::ErrorCode create_error = _moab.tag_get_handle(
                GLOBAL_ID_TAG_NAME, 1, moab::MB_TYPE_INTEGER, global_id_tag,
                moab::MB_TAG_DENSE | moab::MB_TAG_CREAT, &default_id);
            if (create_error != moab::MB_SUCCESS)
                local_error = "failed to create MOAB GLOBAL_ID tag";
        }
        collective_error(local_error);
        for (const int dimension : { 0, static_cast<int>(_dim) })
        {
            moab::Range entities;
            moab::ErrorCode error = moab::MB_SUCCESS;
            std::vector<int> ids;
            collective_prepare([&] {
                error = _moab.get_entities_by_dimension(_root, dimension, entities);
                if (error == moab::MB_SUCCESS)
                {
                    ids.resize(entities.size());
                    std::iota(ids.begin(), ids.end(), 1);
                }
            }, "deterministic GLOBAL_ID preparation failed");
            if (error == moab::MB_SUCCESS && !entities.empty())
                error = _moab.tag_set_data(global_id_tag, entities, ids.data());
            if (error != moab::MB_SUCCESS)
                local_error = "failed to assign deterministic MOAB GLOBAL_ID values";
            collective_error(local_error);
        }
    }

    if (!valid_ids())
        KARMA_THROW(InvalidArgument("MOAB assign_global_ids did not produce valid authoritative IDs"));
}


void KARMAMesh::assign_ids (bool force)
{
	create_id_tags();

	// Each dimension is numbered in its own independent sequence: vertices
	// get [0, n_vertices), elements get [0, n_elements), and so on.
	//
	// This matters. DoF numbering is derived from the vertex ids, and DMMoab
	// requires them to be dense over the vertex set. A single sequence
	// spanning all dimensions would interleave element ids among them and
	// push the vertex ids past the end of that range -- which surfaces as
	// "Scatter indices in iy are out of range" from deep inside a PETSc
	// VecScatter, a long way from the cause.
	for (int d = 0; d <= static_cast<int>(_dim); ++d)
	{
		moab::Range ents;
		check(_moab.get_entities_by_dimension(_root, d, ents),
		      "get_entities_by_dimension(assign_ids)");

		// -- global ids ---------------------------------------------------
		//
        // Replicated meshes number each identical copy locally. Partitioned
        // meshes number owned entities collectively, then exchange those ids
        // to shared nonowners.
        moab::Range numbered_ents = ents;
#ifdef MOAB_HAVE_MPI
        if (_n_ranks > 1 && !_replicated && !numbered_ents.empty())
            check(_pcomm->filter_pstatus(numbered_ents, PSTATUS_NOT_OWNED, PSTATUS_NOT),
                  "filter_pstatus(assign_ids owned)");
#endif
        std::vector<GlobalId> gids(numbered_ents.size());
        if (!numbered_ents.empty())
            check(_moab.tag_get_data(_global_id_tag, numbered_ents, gids.data()),
                  "tag_get_data(global ids)");

        std::size_t n_to_number = 0;
        GlobalId local_max = -1;
        for (const GlobalId id : gids)
        {
            if (force || id < 0)
                ++n_to_number;
            else
                local_max = std::max(local_max, id);
        }

        GlobalId start = force ? 0 : local_max + 1;
#ifdef MOAB_HAVE_MPI
        if (_n_ranks > 1)
        {
            GlobalId global_max = -1;
            if (MPI_Allreduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, _pcomm->comm()) != MPI_SUCCESS)
                KARMA_THROW(InvalidArgument("MPI_Allreduce failed while assigning mesh ids"));
            const auto local_count = static_cast<GlobalId>(n_to_number);
            GlobalId offset = 0;
            if (MPI_Exscan(&local_count, &offset, 1, MPI_INT, MPI_SUM, _pcomm->comm()) != MPI_SUCCESS)
                KARMA_THROW(InvalidArgument("MPI_Exscan failed while assigning mesh ids"));
            start = (force ? 0 : global_max + 1) + (_rank == 0 ? 0 : offset);
        }
#endif

        GlobalId next = start;
        for (auto& id : gids)
            if (force || id < 0)
                id = next++;
        if (!numbered_ents.empty())
            check(_moab.tag_set_data(_global_id_tag, numbered_ents, gids.data()),
                  "tag_set_data(global ids)");

#ifdef MOAB_HAVE_MPI
        if (_n_ranks > 1 && !_replicated)
            check(_pcomm->exchange_tags(_global_id_tag, ents),
                  "exchange_tags(global ids)");
#endif

        if (ents.empty())
            continue;

		// -- local ids ------------------------------------------------------
		//
		// Rank-local, contiguous within the dimension, covering ghosts.
		// Always rewritten: local numbering is only meaningful relative to
		// the current entity set, and a ghost exchange changes that set.
		std::vector<LocalId> lids(ents.size());
		for (std::size_t i = 0; i < lids.size(); ++i)
			lids[i] = static_cast<LocalId>(i);

		check(_moab.tag_set_data(_local_id_tag, ents, lids.data()),
		      "tag_set_data(local ids)");
	}

	// MOAB's own GLOBAL_ID is deliberately left untouched: it belongs to
	// whoever produced the mesh.
}


GlobalId KARMAMesh::global_id (moab::EntityHandle entity) const
{
	GlobalId id = -1;
	check(_moab.tag_get_data(_global_id_tag, &entity, 1, &id),
	      "tag_get_data(global_id)");
	return id;
}


LocalId KARMAMesh::local_id (moab::EntityHandle entity) const
{
	LocalId id = -1;
	check(_moab.tag_get_data(_local_id_tag, &entity, 1, &id),
	      "tag_get_data(local_id)");
	return id;
}


void KARMAMesh::global_ids (const moab::Range& entities,
                            std::vector<GlobalId>& ids) const
{
	ids.resize(entities.size());
	if (!entities.empty())
		check(_moab.tag_get_data(_global_id_tag, entities, ids.data()),
		      "tag_get_data(global ids)");
}


void KARMAMesh::local_ids (const moab::Range& entities,
                           std::vector<LocalId>& ids) const
{
	ids.resize(entities.size());
	if (!entities.empty())
		check(_moab.tag_get_data(_local_id_tag, entities, ids.data()),
		      "tag_get_data(local ids)");
}


void KARMAMesh::update ()
{
	_vertices.clear();
	_elements.clear();
	_owned_elements.clear();

	check(_moab.get_entities_by_dimension(_root, 0, _vertices),
	      "get_entities_by_dimension(vertices)");
	check(_moab.get_entities_by_dimension(_root, static_cast<int>(_dim), _elements),
	      "get_entities_by_dimension(elements)");

#ifdef MOAB_HAVE_MPI
	if (_n_ranks > 1)
	{
		_owned_elements = _elements;
		check(_pcomm->filter_pstatus(_owned_elements, PSTATUS_NOT_OWNED, PSTATUS_NOT),
		      "filter_pstatus(owned elements)");
	}
	else
#endif
	{
		_owned_elements = _elements;
	}
}


std::size_t KARMAMesh::n_vertices () const
{
	return _vertices.size();
}


std::size_t KARMAMesh::n_elements () const
{
	return _elements.size();
}

unsigned KARMAMesh::n_sides( moab::EntityHandle element ) const
{
    const moab::EntityType type = _moab.type_from_handle( element );
    if( moab::CN::Dimension( type ) != static_cast< int >( _dim ) )
        throw InvalidArgument( "side query requires a mesh element" );
    return static_cast< unsigned >( moab::CN::NumSubEntities( type, static_cast< int >( _dim ) - 1 ) );
}

std::size_t KARMAMesh::side_vertices( moab::EntityHandle element, unsigned side,
                                      std::span< moab::EntityHandle > vertices ) const
{
    const moab::EntityType type = _moab.type_from_handle( element );
    if( side >= n_sides( element ) ) throw InvalidArgument( "side index is out of range" );

    const moab::EntityHandle* connectivity = nullptr;
    int nconn                              = 0;
    check( _moab.get_connectivity( element, connectivity, nconn ), "get_connectivity(side)" );

    moab::EntityType side_type = moab::MBMAXTYPE;
    int side_size              = 0;
    const short* indices       = moab::CN::SubEntityVertexIndices( type, static_cast< int >( _dim ) - 1,
                                                                   static_cast< int >( side ), side_type, side_size );
    if( !indices || side_size <= 0 || vertices.size() < static_cast< std::size_t >( side_size ) )
        throw InvalidArgument( "invalid side connectivity buffer" );

    for( int i = 0; i < side_size; ++i )
    {
        if( indices[i] < 0 || indices[i] >= nconn ) throw LogicError();
        vertices[static_cast< std::size_t >( i )] = connectivity[indices[i]];
    }
    return static_cast< std::size_t >( side_size );
}

std::vector< moab::EntityHandle > KARMAMesh::side_vertices( moab::EntityHandle element, unsigned side ) const
{
    const moab::EntityType type = _moab.type_from_handle( element );
    if( side >= n_sides( element ) ) throw InvalidArgument( "side index is out of range" );
    const moab::EntityType side_type =
        moab::CN::SubEntityType( type, static_cast< int >( _dim ) - 1, static_cast< int >( side ) );
    std::vector< moab::EntityHandle > vertices(
        static_cast< std::size_t >( moab::CN::VerticesPerEntity( side_type ) ) );
    const std::size_t count = side_vertices( element, side, vertices );
    if( count != vertices.size() ) throw LogicError();
    return vertices;
}

std::vector< GlobalId > KARMAMesh::canonical_side_vertex_ids( moab::EntityHandle element, unsigned side ) const
{
    const auto vertices = side_vertices( element, side );
    moab::Tag global_id = nullptr;
    if( _moab.tag_get_handle( GLOBAL_ID_TAG_NAME, global_id ) != moab::MB_SUCCESS )
        throw InvalidArgument( "side identity requires authoritative MOAB GLOBAL_ID" );
    std::vector< GlobalId > ids( vertices.size() );
    check( _moab.tag_get_data( global_id, vertices.data(), static_cast< int >( vertices.size() ), ids.data() ),
           "tag_get_data(GLOBAL_ID canonical side)" );
    if( std::any_of( ids.begin(), ids.end(), []( GlobalId id ) { return id <= 0; } ) )
        throw InvalidArgument( "side identity requires positive authoritative MOAB GLOBAL_ID values" );
    std::sort( ids.begin(), ids.end() );
    if( std::adjacent_find( ids.begin(), ids.end() ) != ids.end() )
        throw InvalidArgument( "side identity requires unique authoritative MOAB GLOBAL_ID values" );
    return ids;
}

void KARMAMesh::collective_topology_error( const std::string& local_error ) const
{
#ifdef MOAB_HAVE_MPI
    if( _n_ranks > 1 )
    {
        const int candidate = local_error.empty() ? _n_ranks : _rank;
        int failing_rank = _n_ranks;
        if( MPI_Allreduce( &candidate, &failing_rank, 1, MPI_INT, MPI_MIN, _pcomm->comm() ) != MPI_SUCCESS )
            KARMA_THROW( InvalidArgument( "MPI_Allreduce failed during side-topology validation" ) );
        if( failing_rank == _n_ranks ) return;
        int length = _rank == failing_rank ? static_cast< int >( local_error.size() ) : 0;
        if( MPI_Bcast( &length, 1, MPI_INT, failing_rank, _pcomm->comm() ) != MPI_SUCCESS )
            KARMA_THROW( InvalidArgument( "MPI_Bcast failed during side-topology validation" ) );
        std::string message( static_cast< std::size_t >( length ), '\0' );
        if( _rank == failing_rank ) message = local_error;
        if( MPI_Bcast( message.data(), length, MPI_CHAR, failing_rank, _pcomm->comm() ) != MPI_SUCCESS )
            KARMA_THROW( InvalidArgument( "MPI_Bcast failed during side-topology validation" ) );
        KARMA_THROW( InvalidArgument( "rank " + std::to_string( failing_rank ) + ": " + message ) );
    }
#endif
    if( !local_error.empty() ) KARMA_THROW( InvalidArgument( local_error ) );
}

std::optional< moab::EntityHandle > KARMAMesh::neighbor_local( moab::EntityHandle element, unsigned side ) const
{

    std::array< moab::EntityHandle, moab::CN::MAX_NODES_PER_ELEMENT > side_conn{};
    const std::size_t side_size = side_vertices( element, side, side_conn );
    moab::Tag conventional_ids = nullptr;
    const bool use_global_ids = _n_ranks > 1 && !_replicated &&
                                _moab.tag_get_handle( GLOBAL_ID_TAG_NAME, conventional_ids ) == moab::MB_SUCCESS;
    if( _n_ranks > 1 && !_replicated && !use_global_ids )
        KARMA_THROW( InvalidArgument( "partitioned side matching requires authoritative MOAB GLOBAL_ID" ) );
    std::vector< int > side_ids;
    if( use_global_ids )
    {
        side_ids.resize( side_size );
        check( _moab.tag_get_data( conventional_ids, side_conn.data(), static_cast< int >( side_size ), side_ids.data() ),
               "tag_get_data(GLOBAL_ID neighbor side)" );
    }

    moab::EntityHandle match = 0;
    unsigned match_count     = 0;
    // Search every visible owned or ghost element by authoritative identity.
    // Ghost vertices need not reuse the owner's local handles, so bridge
    // adjacency alone cannot establish cross-partition side incidence.
    for( const moab::EntityHandle candidate : _elements )
    {
        if( candidate == element ) continue;
        const moab::EntityHandle* conn = nullptr;
        int nconn                      = 0;
        check( _moab.get_connectivity( candidate, conn, nconn ), "get_connectivity(neighbor)" );
        bool contains_side = false;
        if( use_global_ids )
        {
            std::vector< int > candidate_ids( static_cast< std::size_t >( nconn ) );
            check( _moab.tag_get_data( conventional_ids, conn, nconn, candidate_ids.data() ),
                   "tag_get_data(GLOBAL_ID neighbor candidate)" );
            const moab::EntityType type = _moab.type_from_handle( candidate );
            for( int candidate_side = 0; candidate_side < moab::CN::NumSubEntities( type, static_cast< int >( _dim ) - 1 );
                 ++candidate_side )
            {
                moab::EntityType side_type = moab::MBMAXTYPE;
                int candidate_side_size = 0;
                const short* indices = moab::CN::SubEntityVertexIndices( type, static_cast< int >( _dim ) - 1,
                                                                            candidate_side, side_type, candidate_side_size );
                if( candidate_side_size != static_cast< int >( side_size ) || !indices ) continue;
                std::vector< int > candidate_side_ids;
                candidate_side_ids.reserve( side_size );
                for( int index = 0; index < candidate_side_size; ++index ) candidate_side_ids.push_back( candidate_ids[indices[index]] );
                auto expected = side_ids;
                std::sort( expected.begin(), expected.end() );
                std::sort( candidate_side_ids.begin(), candidate_side_ids.end() );
                if( candidate_side_ids == expected ) { contains_side = true; break; }
            }
        }
        else
            contains_side =
                std::all_of( side_conn.begin(), side_conn.begin() + static_cast< std::ptrdiff_t >( side_size ),
                             [conn, nconn]( moab::EntityHandle vertex ) {
                                 return std::find( conn, conn + nconn, vertex ) != conn + nconn;
                             } );
        if( contains_side )
        {
            match = candidate;
            if( ++match_count > 1 ) throw InvalidArgument( "non-manifold side has multiple neighbors" );
        }
    }

    if( match_count != 0 ) return match;

#ifdef MOAB_HAVE_MPI
    if( _n_ranks > 1 && !_replicated )
    {
        moab::EntityHandle side_entity = 0;
        check( _moab.side_element( element, static_cast< int >( _dim ) - 1, static_cast< int >( side ), side_entity ),
               "side_element(neighbor)" );
        unsigned char pstatus = 0;
        check( _pcomm->get_pstatus( side_entity, pstatus ), "get_pstatus(side neighbor)" );
        if( ( pstatus & PSTATUS_INTERFACE ) != 0 )
        {
            std::ostringstream description;
            if( _moab.tag_get_handle( GLOBAL_ID_TAG_NAME, conventional_ids ) == moab::MB_SUCCESS )
            {
                std::vector< int > ids( side_size );
                check( _moab.tag_get_data( conventional_ids, side_conn.data(), static_cast< int >( side_size ), ids.data() ),
                       "tag_get_data(GLOBAL_ID side neighbor)" );
                std::sort( ids.begin(), ids.end() );
                description << " [GLOBAL_ID vertices";
                for( const int id : ids ) description << ' ' << id;
                description << ']';
            }
            KARMA_THROW( InvalidArgument( "required ghost neighbor is absent for partition-interface side" +
                                          description.str() ) );
        }
    }
#endif
    return std::nullopt;
}

std::optional< moab::EntityHandle > KARMAMesh::neighbor( moab::EntityHandle element, unsigned side ) const
{ return neighbor_local( element, side ); }

bool KARMAMesh::is_boundary_side( moab::EntityHandle element, unsigned side ) const
{ return !neighbor( element, side ).has_value(); }

void KARMAMesh::validate_owned_side_topology() const
{
    std::string local_error;
    try
    {
        if( _n_ranks > 1 && !_replicated ) ensure_global_ids();
        for( const moab::EntityHandle element : _owned_elements )
            for( unsigned side = 0; side < n_sides( element ); ++side )
                (void)neighbor_local( element, side );
    }
    catch( const std::exception& error )
    {
        local_error = error.what();
    }
    collective_topology_error( local_error );
}

SideId KARMAMesh::side_id( moab::EntityHandle element, unsigned side ) const
{
    if( side >= n_sides( element ) ) throw InvalidArgument( "side index is out of range" );
    const GlobalId id = global_id( element );
    if( id < 0 ) throw InvalidArgument( "side identity requires an assigned global element id" );
    return { id, side };
}

void KARMAMesh::vertex_coordinates (const moab::Range& verts,
                                    std::vector<Real>& coords) const
{
	coords.resize(3 * verts.size());
	if (!verts.empty())
		check(_moab.get_coords(verts, coords.data()), "get_coords");
}


std::size_t KARMAMesh::element_coordinates (moab::EntityHandle elem,
                                            std::span<Real> coords) const
{
	const moab::EntityHandle* conn = nullptr;
	int nconn = 0;
	check(_moab.get_connectivity(elem, conn, nconn), "get_connectivity");

	if (coords.size() < static_cast<std::size_t>(3 * nconn))
		karma_exception("element_coordinates: buffer holds " << coords.size()
		                << " reals, need " << 3 * nconn);

	check(_moab.get_coords(conn, nconn, coords.data()), "get_coords(element)");

	return static_cast<std::size_t>(nconn);
}


Point KARMAMesh::centroid (moab::EntityHandle elem) const
{
	// 27 vertices is the largest element MOAB can evaluate (QuadraticHex).
	Real buf[3 * 27];
	const std::size_t nconn = element_coordinates(elem, std::span<Real>(buf, 3 * 27));

	Point c = Point::Zero();
	for (std::size_t i = 0; i < nconn; ++i)
		c += point_at(buf, i);

	return c / static_cast<Real>(nconn);
}


BoundingBox KARMAMesh::bounding_box () const
{
	std::vector<Real> coords;
	vertex_coordinates(_vertices, coords);

	BoundingBox box;
	for (std::size_t i = 0; i < _vertices.size(); ++i)
		box.extend(point_at(coords.data(), i));

	return box;
}


void KARMAMesh::generate_box (unsigned nx, unsigned ny, unsigned nz,
                              const Point& lo, const Point& hi)
{
	// Elements are created explicitly rather than through MOAB's structured
	// interface (ScdInterface).  A structured box stores connectivity
	// implicitly and answers get_connectivity() with MB_NOT_IMPLEMENTED,
	// but explicit connectivity is exactly what the FE layer needs:
	// DMMoabFEMComputeBasis is driven by an element's vertex list.  The
	// memory saved by an implicit representation is not worth a mesh the
	// assembly kernels cannot traverse.
	const unsigned ex = nx;
	const unsigned ey = (_dim >= 2 ? ny : 1);
	const unsigned ez = (_dim >= 3 ? nz : 1);

	const unsigned vx = ex + 1;
	const unsigned vy = (_dim >= 2 ? ey + 1 : 1);
	const unsigned vz = (_dim >= 3 ? ez + 1 : 1);

	const Point d((hi(0) - lo(0)) / static_cast<Real>(ex),
	              (_dim >= 2 ? (hi(1) - lo(1)) / static_cast<Real>(ey) : 0.0),
	              (_dim >= 3 ? (hi(2) - lo(2)) / static_cast<Real>(ez) : 0.0));

	// Vertices, i fastest.
	std::vector<double> coords;
	coords.reserve(3 * vx * vy * vz);
	for (unsigned k = 0; k < vz; ++k)
		for (unsigned j = 0; j < vy; ++j)
			for (unsigned i = 0; i < vx; ++i)
			{
				coords.push_back(lo(0) + static_cast<Real>(i) * d(0));
				coords.push_back(lo(1) + static_cast<Real>(j) * d(1));
				coords.push_back(lo(2) + static_cast<Real>(k) * d(2));
			}

	moab::Range verts;
	check(_moab.create_vertices(coords.data(),
	                            static_cast<int>(coords.size() / 3), verts),
	      "create_vertices");

	std::vector<moab::EntityHandle> vh(verts.begin(), verts.end());
	const auto vid = [vx, vy](unsigned i, unsigned j, unsigned k)
	{
		return static_cast<std::size_t>((k * vy + j) * vx + i);
	};

	// Elements, in MOAB's canonical vertex order for each type.
	moab::EntityHandle conn[8];
	moab::EntityHandle elem = 0;

	if (_dim == 1)
	{
		for (unsigned i = 0; i < ex; ++i)
		{
			conn[0] = vh[vid(i,     0, 0)];
			conn[1] = vh[vid(i + 1, 0, 0)];
			check(_moab.create_element(moab::MBEDGE, conn, 2, elem), "create_element(edge)");
		}
	}
	else if (_dim == 2)
	{
		for (unsigned j = 0; j < ey; ++j)
			for (unsigned i = 0; i < ex; ++i)
			{
				conn[0] = vh[vid(i,     j,     0)];
				conn[1] = vh[vid(i + 1, j,     0)];
				conn[2] = vh[vid(i + 1, j + 1, 0)];
				conn[3] = vh[vid(i,     j + 1, 0)];
				check(_moab.create_element(moab::MBQUAD, conn, 4, elem), "create_element(quad)");
			}
	}
	else
	{
		for (unsigned k = 0; k < ez; ++k)
			for (unsigned j = 0; j < ey; ++j)
				for (unsigned i = 0; i < ex; ++i)
				{
					conn[0] = vh[vid(i,     j,     k)];
					conn[1] = vh[vid(i + 1, j,     k)];
					conn[2] = vh[vid(i + 1, j + 1, k)];
					conn[3] = vh[vid(i,     j + 1, k)];
					conn[4] = vh[vid(i,     j,     k + 1)];
					conn[5] = vh[vid(i + 1, j,     k + 1)];
					conn[6] = vh[vid(i + 1, j + 1, k + 1)];
					conn[7] = vh[vid(i,     j + 1, k + 1)];
					check(_moab.create_element(moab::MBHEX, conn, 8, elem), "create_element(hex)");
				}
	}

	assign_ids();
	update();
}


void KARMAMesh::read (std::string_view filename, std::string_view options)
{
	const std::string file(filename);
	const std::string opts(options);
    _replicated = opts.find("PARALLEL=") == std::string::npos;

	// A null file set means "load into the root set". Passing &_root would
	// hand MOAB the handle 0, which it asserts against.
    const moab::ErrorCode load_error = _moab.load_file(file.c_str(), nullptr,
                                                        opts.empty() ? nullptr : opts.c_str());
#ifdef MOAB_HAVE_MPI
    if (_n_ranks > 1)
    {
        int local_failure = load_error == moab::MB_SUCCESS ? 0 : 1;
        int any_failure = 0;
        if (MPI_Allreduce(&local_failure, &any_failure, 1, MPI_INT, MPI_MAX, _pcomm->comm()) != MPI_SUCCESS)
            KARMA_THROW(InvalidArgument("MPI_Allreduce failed after mesh load"));
        if (any_failure)
            KARMA_THROW(InvalidArgument("MOAB mesh load failed on at least one rank"));
    }
    else
#endif
    {
        check(load_error, "load_file");
    }

	assign_ids();
	update();
}


void KARMAMesh::write (std::string_view filename, std::string_view options) const
{
	const std::string file(filename);
	const std::string opts(options);

	// Likewise: null output set writes everything in the root set.
	check(_moab.write_file(file.c_str(), nullptr,
	                       opts.empty() ? nullptr : opts.c_str(),
	                       nullptr, 0),
	      "write_file");
}


void KARMAMesh::resolve_shared_entities (unsigned n_ghost_layers)
{
    _replicated = false;
#ifdef MOAB_HAVE_MPI
	if (_n_ranks <= 1)
		return;

	// Match entities across ranks by KARMA's own global id rather than
	// MOAB's GLOBAL_ID, so sharing agrees with the DoF numbering.
	create_id_tags();
	check(_pcomm->resolve_shared_ents(_root, static_cast<int>(_dim), -1,
	                                  &_global_id_tag),
	      "resolve_shared_ents");

	if (n_ghost_layers > 0)
		check(_pcomm->exchange_ghost_cells(static_cast<int>(_dim),
		                                   0,                                 // bridge over vertices
		                                   static_cast<int>(n_ghost_layers),
		                                   0,
		                                   true),                             // store remote handles
		      "exchange_ghost_cells");

	update();

	// Ghosts arrived, so the local numbering has to cover them. Global ids
	// are preserved: assign_ids() only fills entities that lack one.
	assign_ids();

#else
	(void) n_ghost_layers;
#endif
}


int KARMAMesh::material_id (moab::EntityHandle elem) const
{
    moab::Tag tag;
    const moab::ErrorCode tag_error = _moab.tag_get_handle(MATERIAL_SET_TAG_NAME, tag);
    if (tag_error == moab::MB_TAG_NOT_FOUND)
        return 0;
    check(tag_error, "tag_get_handle(material)");

    moab::DataType data_type = moab::MB_TYPE_OPAQUE;
    int length = 0;
    check(_moab.tag_get_data_type(tag, data_type), "tag_get_data_type(material)");
    check(_moab.tag_get_length(tag, length), "tag_get_length(material)");
    if (data_type != moab::MB_TYPE_INTEGER || length != 1)
        KARMA_THROW(InvalidArgument("MATERIAL_SET tag must contain one integer"));

    // The id lives on the unique tagged set that contains the element.
    moab::Range sets;
    check(_moab.get_entities_by_type_and_tag(_root, moab::MBENTITYSET, &tag, nullptr, 1, sets),
          "get_entities_by_type_and_tag(material)");
    int material = 0;
    bool found = false;
    for (const moab::EntityHandle set : sets)
    {
        if (!_moab.contains_entities(set, &elem, 1))
            continue;
        if (found)
            KARMA_THROW(InvalidArgument("element belongs to multiple material sets"));
        check(_moab.tag_get_data(tag, &set, 1, &material), "tag_get_data(material)");
        found = true;
    }

    return found ? material : 0;
}


void KARMAMesh::material_sets (moab::Range& sets) const
{
	sets.clear();

	moab::Tag tag;
	if (_moab.tag_get_handle(MATERIAL_SET_TAG_NAME, tag) != moab::MB_SUCCESS)
		return;

	check(_moab.get_entities_by_type_and_tag(_root, moab::MBENTITYSET, &tag,
	                                         nullptr, 1, sets),
	      "get_entities_by_type_and_tag(material)");
}


void KARMAMesh::boundary_sets (moab::Range& sets) const
{
	sets.clear();

	for (const char* name : { NEUMANN_SET_TAG_NAME, DIRICHLET_SET_TAG_NAME })
	{
		moab::Tag tag;
		if (_moab.tag_get_handle(name, tag) != moab::MB_SUCCESS)
			continue;

		moab::Range found;
		check(_moab.get_entities_by_type_and_tag(_root, moab::MBENTITYSET, &tag,
		                                         nullptr, 1, found),
		      "get_entities_by_type_and_tag(boundary)");
		sets.merge(found);
	}
}

}
