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

#include "karma_error.h"
#include "karma_mesh.h"

#ifdef KARMA_HAVE_LIBMESH
#include "mesh_base.h"
#include "parmetis_partitioner.h"
#include "metis_partitioner.h"
#endif

#include "karma_partitioner.h"


KARMAPartitioner::KARMAPartitioner(KARMA::PartitionerType part_type)
{
	switch (part_type)
	{
		case 0:
			m_partitioner = new ParmetisPartitioner() ;
			break ;
		case 1:
			m_partitioner = new MetisPartitioner() ;
			break ;
		default:
			std::cerr << "Currently only ParMetis and Metis partitioners are supported by KARMA." ;
			break ;
	}
}

KARMAPartitioner::~KARMAPartitioner()
{
	delete m_partitioner ;
}

// ------------------------------------------------------------
// KARMAPartitioner implementation
void KARMAPartitioner::Partition (KARMAMesh& mesh,
					 const unsigned int n_sbdmns = KARMA::num_processors )
{
	m_partitioner->partition ( dynamic_cast<MeshBase&>(mesh), n_sbdmns ) ;
}

// ------------------------------------------------------------
// KARMAPartitioner implementation
void KARMAPartitioner::RePartition (KARMAMesh& mesh,
					 const unsigned int n_sbdmns = KARMA::num_processors )
{
	m_partitioner->repartition ( dynamic_cast<MeshBase&>(mesh), n_sbdmns ) ;
}
