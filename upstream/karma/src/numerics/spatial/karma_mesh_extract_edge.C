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


#include "karma.h"
#include "petsc.h"

#include <vector>
#include <cassert>

#include "mesh.h"
#include "mesh_data.h"
#include "elem.h"
#include "node.h"
#include "boundary_info.h"

#include "karma_error.h"
#include "karma_mesh_utils.h"


#if   defined(LIBMESH_HAVE_TR1_UNORDERED_MAP)
# include <tr1/unordered_map>
#elif defined(LIBMESH_HAVE_UNORDERED_MAP)
# include <unordered_map>
#elif defined(LIBMESH_HAVE_HASH_MAP)

# include <hash_map>
#elif defined(LIBMESH_HAVE_EXT_HASH_MAP)
# include <ext/hash_map>
#else
# include <map>
#endif
#include "remote_elem.h"

namespace KARMA
{

	// Mesh Parent - 2-D parent mesh
	// unsigned int direction - Direction of 1-D mesh to be extracted
	// double ps - Location of the edge perpendicular to "direction"
	void karma_mesh_extract_edge(const Mesh& parent, Mesh* boundary_mesh, MeshData* boundary_mesh_data, unsigned int direction, double ps)
	{
		assert(parent.mesh_dimension() == 2) ;
		assert(direction > 0 && direction < 4) ;

		assert(boundary_mesh != NULL) ;
		assert(boundary_mesh->mesh_dimension() == 1) ;
		
		//parent.print_info() ;
	
		//Mesh *boundary_mesh = new Mesh(1);
		
		bool use_mesh_data = false ;
		if (boundary_mesh_data != NULL)
			use_mesh_data = true ;
		
		boundary_mesh->clear() ;
		
		std::vector<unsigned int> elem_num, node_ids_local, node_ids_global ;
		std::vector<unsigned int>::iterator node_loc ;
		std::map<unsigned int,unsigned int> node_map ;
	
		MeshBase::const_element_iterator       el     = parent.elements_begin();
		const MeshBase::const_element_iterator end_el = parent.elements_end();
	
		for ( ; el != end_el; ++el)
		{
			Elem* elem = *el;
			int numnodes = 0;
	
			for (unsigned int inode=0; inode<elem->n_nodes(); inode++)
			{
				Node* node = elem->get_node(inode) ;
	
				node_loc = std::find( node_ids_global.begin(), node_ids_global.end(), node->id());
	
				// continue search if we have already passed through this node before..
				if( node_loc != node_ids_global.end() && node_ids_global.size() != 0)
				{
					numnodes++ ;
					node_ids_local.push_back(inode) ;
					continue ;
				}
	
				bool killnode = true ;
				if (fabs((*node)(direction-1)-ps) < 1e-12)	killnode = false;
	
				if(!killnode)
				{
					//std::cout << "\nAdding node " << node->id() << " and " << node_ids_global.size() << " = [" << (*node)(0) << ", " << (*node)(1) << ", " << (*node)(2) << "] ." ;
	
					boundary_mesh->add_point(Point((*node)(0), (*node)(1), (*node)(2)), node_ids_global.size(), libMesh::processor_id()) ;
	
					node_map.insert(std::pair<unsigned int, unsigned int>(node->id(),node_ids_global.size())) ;
					
					numnodes++ ;
					node_ids_local.push_back(inode) ;
					node_ids_global.push_back(node->id()) ;
				}
			}
		//	std::cout << "\nElement {" << elem->id() << "} : Contains total nodes = " << numnodes  << "] ." ;
			
			elem_num.push_back(numnodes) ;
		}
		
		//std::cin.get() ;
		
		MeshBase::const_element_iterator       nel     = parent.elements_begin();
		const MeshBase::const_element_iterator nend_el = parent.elements_end();
		
		std::map<unsigned int,unsigned int>::iterator nditer ;
		unsigned int inode = 0 ;	int iel = 0, elem_id_counter = 0 ;
		for ( ; nel != nend_el; ++nel)
		{
			Elem* elem = *nel;
	
			if (elem_num[iel] >= 2)
			for (unsigned int side=0; side<elem->n_sides(); side++)
			{
				if (elem->neighbor(side) == NULL)	// boundary side
				{
					int isuccess = 0 ;
					for (unsigned int jnode=0; jnode<elem_num[iel]; jnode++)
					{
						if (elem->is_node_on_side(node_ids_local[inode+jnode], side))
						{
	//						std::cout << "\n[" << inode << "," << jnode << "] = Element {" << elem->side(side)->id() << "} : Contains node " << elem->node(node_ids_global[inode+jnode])  << "] ." ;
							++isuccess ;
						}
					}
					
					if(isuccess >=2)	// since this is 1-D, we expect atleast an edge2 element.
					{
						//std::cout << "\nContains node " << node_ids_global[inode+jnode]  << "] ." ;
						Elem* edge = elem->build_edge(side).release();
						//std::cout << "\nAdding element of type: " << edge->type() ;
						for (unsigned int jnode=0; jnode<edge->n_nodes(); jnode++)
						{
							Node* nd = edge->get_node(jnode) ;
							nditer = node_map.find(nd->id()) ;
							if (nditer != node_map.end())
								nd->set_id(nditer->second) ;
						}
						edge->set_id(elem_id_counter++) ;
						boundary_mesh->add_elem(edge) ;

						// Maybe add boundary conditions for this element
						//if (parent.boundary_info->boundary_id (elem, side) != parent.boundary_info->invalid_id)
						//	boundary_mesh->boundary_info->add_side (edge, side,  parent.boundary_info->boundary_id (elem, side));
					}
				}
			}
			
			inode += elem_num[iel] ;
			++iel;
		}
		
		// Find and set neighbors
		int totelem = boundary_mesh->n_elem() ;
		for (int iel=0; iel < totelem; iel++)
		{
			Elem* elem = boundary_mesh->elem(iel) ;
	
			// Just go through the numbering and set neighboring elements based on it. 
			// This works since we are only working on 1-D edges now.
			if(iel > 0)
			{
				// Edge left
				elem->set_neighbor(0, boundary_mesh->elem(iel-1)) ;
			}
			
			if(iel < totelem-1)
			{
				// Edge right
				elem->set_neighbor(1, boundary_mesh->elem(iel+1)) ;
			}
		}
		
		//boundary_mesh->prepare_for_use() ;

//std::cin.get() ;

//PetscPrintf(KARMA_COMM_WORLD,"\nboundary_mesh->write();");
boundary_mesh->write("test_1d_mesh.msh") ;
	//PetscPrintf(KARMA_COMM_WORLD,"\tdone ..") ;

	boundary_mesh->clear() ;
	if(use_mesh_data)
		boundary_mesh_data->clear() ;
	
	boundary_mesh->read("test_1d_mesh.msh", boundary_mesh_data) ;
	
	return ;

	karma_error() ;
	
	PetscPrintf(KARMA_COMM_WORLD,"\nboundary_mesh->renumber_nodes_and_elements();");
	boundary_mesh->renumber_nodes_and_elements();
	PetscPrintf(KARMA_COMM_WORLD,"\tdone ..") ;
	
	PetscPrintf(KARMA_COMM_WORLD,"\nboundary_mesh->find_neighbors();");
	// Let all the elements find their neighbors
	boundary_mesh->find_neighbors(false, false);
	/*
	bool reset_remote_elements = false ;
	int _dim = 1 ;
	const bool tempflg = false ;
	
	{
		PetscPrintf(KARMA_COMM_WORLD,"\nelem->set_neighbor();");
		MeshBase::const_element_iterator       el     = boundary_mesh->elements_begin();
		const MeshBase::const_element_iterator el_end = boundary_mesh->elements_end();		
		
		
		//TODO:[BSK] This should be removed later?!		
		for (; el != el_end; ++el)
		{
			Elem* elem = *el;
			PetscPrintf(KARMA_COMM_WORLD,"\nelem->num_neighbor() = ");
			PetscPrintf(KARMA_COMM_WORLD,"\n %D.", elem->n_sides());
		    for (unsigned int s=0; s<elem->n_neighbors(); s++)
			{
				PetscPrintf(KARMA_COMM_WORLD,"\nSide=%D: ",s) ;
				PetscPrintf(KARMA_COMM_WORLD,"elem->neighbor(s) != remote_elem; %D",elem->neighbor(s) != remote_elem) ;
				if (elem->neighbor(s) != remote_elem || reset_remote_elements)
				{
					PetscPrintf(KARMA_COMM_WORLD,"\n\tSetting null neighbor for side s=%D.", s) ;
					elem->set_neighbor(s,NULL);
				}
			}
		}
		PetscPrintf(KARMA_COMM_WORLD,"\ndone ..") ;
		
		PetscPrintf(KARMA_COMM_WORLD,"\nFind neighboring elements by first finding elements with identical side keys");
		// Find neighboring elements by first finding elements
		// with identical side keys and then check to see if they
		// are neighbors
		{
			// data structures -- Use the hash_multimap if available
			typedef unsigned int                    key_type;
			typedef std::pair<Elem*, unsigned char> val_type;
			typedef std::pair<key_type, val_type>   key_val_pair;

#if   defined(LIBMESH_HAVE_UNORDERED_MAP)
			typedef std::unordered_multimap<key_type, val_type> map_type;    
#elif defined(LIBMESH_HAVE_TR1_UNORDERED_MAP)
			typedef std::tr1::unordered_multimap<key_type, val_type> map_type;    
#elif defined(LIBMESH_HAVE_HASH_MAP)    
			typedef std::hash_multimap<key_type, val_type> map_type;    
#elif defined(LIBMESH_HAVE_EXT_HASH_MAP)
# if    (__GNUC__ == 3) && (__GNUC_MINOR__ == 0) // gcc 3.0   
			typedef std::hash_multimap<key_type, val_type> map_type;
# elif (__GNUC__ >= 3)                          // gcc 3.1 & newer
			typedef __gnu_cxx::hash_multimap<key_type, val_type> map_type;
# else
// XLC and who knows what other compilers get here.
// Try the most standard thing we can:
			typedef std::multimap<key_type, val_type>  map_type;
# endif
#else
			typedef std::multimap<key_type, val_type>  map_type;
#endif
    
		    // A map from side keys to corresponding elements & side numbers  
		    map_type side_to_elem_map;
  
			el = boundary_mesh->elements_begin() ;
		    for (; el != el_end; ++el)
		    {
				Elem* element = *el;

				for (unsigned int ms=0; ms<element->n_neighbors(); ms++)
				{
				  next_side:
				    // If we haven't yet found a neighbor on this side, try.
				    // Even if we think our neighbor is remote, that
				    // information may be out of date.
				    if (element->neighbor(ms) == NULL ||
					element->neighbor(ms) == remote_elem)
				    {
						// Get the key for the side of this element
						const unsigned int key = element->key(ms);
						
						// Look for elements that have an identical side key
						std::pair <map_type::iterator, map_type::iterator>
						  bounds = side_to_elem_map.equal_range(key);
						
						// May be multiple keys, check all the possible
						// elements which _might_ be neighbors.
						if (bounds.first != bounds.second)
						{
						    // Get the side for this element
						    const AutoPtr<DofObject> my_side(element->side(ms));

						    // Look at all the entries with an equivalent key
						    while (bounds.first != bounds.second)
							{
								// Get the potential element
								Elem* neighbor = bounds.first->second.first;
								
								// Get the side for the neighboring element
								const unsigned int ns = bounds.first->second.second;
								const AutoPtr<DofObject> their_side(neighbor->side(ns));
								//libmesh_assert (my_side.get() != NULL);
								//libmesh_assert (their_side.get() != NULL);			

								// If found a match with my side
					                        //
								// We need special tests here for 1D:
								// since parents and children have an equal
								// side (i.e. a node), we need to check 
								// ns != ms, and we also check level() to
								// avoid setting our neighbor pointer to
								// any of our neighbor's descendants
								if( (*my_side == *their_side) && 
					                            (element->level() == neighbor->level()) &&
					                            ((_dim != 1) || (ns != ms)) )
								{
								    // So share a side.  Is this a mixed pair
								    // of subactive and active/ancestor
								    // elements?
									// If not, then we're neighbors.
								    // If so, then the subactive's neighbor is 

									if (element->subactive() == neighbor->subactive())
									{
										// an element is only subactive if it has
										// been coarsened but not deleted
										element->set_neighbor (ms,neighbor);
										neighbor->set_neighbor(ns,element);
									}
									else if (element->subactive())
									{
										element->set_neighbor(ms,neighbor);
									}
									else if (neighbor->subactive())
									{
										neighbor->set_neighbor(ns,element);
									}
									side_to_elem_map.erase (bounds.first);

									// get out of this nested crap
									goto next_side; 
							    }

							++bounds.first;
						    }
						}
					    
						// didn't find a match...
						// Build the map entry for this element
						key_val_pair kvp;
						
						kvp.first         = key;
						kvp.second.first  = element;
						kvp.second.second = ms;
						
						// use the lower bound as a hint for
						// where to put it.
#if defined(LIBMESH_HAVE_UNORDERED_MAP) || defined(LIBMESH_HAVE_TR1_UNORDERED_MAP) || defined(LIBMESH_HAVE_HASH_MAP) || defined(LIBMESH_HAVE_EXT_HASH_MAP)
						side_to_elem_map.insert (kvp);
#else
						side_to_elem_map.insert (bounds.first,kvp);
#endif
				    }
				}
		    }
		}
	}
	*/
	PetscPrintf(KARMA_COMM_WORLD,"\tdone ..") ;

	// Partition the mesh.
	PetscPrintf(KARMA_COMM_WORLD,"\nboundary_mesh->partition();");
	boundary_mesh->partition();
	PetscPrintf(KARMA_COMM_WORLD,"\tdone ..") ;

	// If we're using ParallelMesh, we'll want it parallelized.
	PetscPrintf(KARMA_COMM_WORLD,"\nboundary_mesh->delete_remote_elements();");
	boundary_mesh->delete_remote_elements();
	PetscPrintf(KARMA_COMM_WORLD,"\tdone ..") ;

	PetscPrintf(KARMA_COMM_WORLD,"\nboundary_mesh->renumber_nodes_and_elements();");
	boundary_mesh->renumber_nodes_and_elements();
	PetscPrintf(KARMA_COMM_WORLD,"\tdone ..") ;

	// Reset our PointLocator.  This needs to happen any time the elements
	// in the underlying elements in the mesh have changed, so we do it here.
	PetscPrintf(KARMA_COMM_WORLD,"\nboundary_mesh->clear_point_locator();");
	boundary_mesh->clear_point_locator();
	PetscPrintf(KARMA_COMM_WORLD,"\tdone ..") ;
	

		int in = 0 ;
		MeshBase::const_node_iterator       n     = boundary_mesh->nodes_begin();
		const MeshBase::const_node_iterator nend = boundary_mesh->nodes_end();
		for ( ; n != nend; ++n)
		{
			Node* nd = *n;
			std::cout << "\nNode " << in++ << " and ID = " << nd->id() << "." ;
		}
	
	
		//		std::cin.get() ;
	
		std::cout << "\nNumber of nodes:= " << boundary_mesh->n_local_nodes() << " and total elems:= " << boundary_mesh->n_active_elem() << ".\n" ;
		std::cin.get() ;
		
		//std::cout << "\nNumber of nodes:= " << boundary_mesh->n_local_nodes() << " and total elems:= " << boundary_mesh->n_active_elem() << ".\n" ;
	
		boundary_mesh->print_info() ;
	
		//userMesh.print_info() ;
		std::cin.get() ;
	
		//boundary_mesh->write(outfile) ;

		//return *boundary_mesh ;
	
	}

}
