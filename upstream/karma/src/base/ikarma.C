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


#include "ikarma.h"
#include <sstream>

// ------------------------------------------------------------
// GlobalReferenceCounter class static member initializations
bool GlobalReferenceCounter::enable_counter = true;		// make this a configurable option using #ifdef's
GlobalReferenceCounter::Counts GlobalReferenceCounter::counts;
unsigned int GlobalReferenceCounter::n_objects = 0;


// ------------------------------------------------------------
// GlobalReferenceCounter class members
std::string GlobalReferenceCounter::get_info ()
{
	if (enable_counter)
	{
		std::ostringstream out;
		
		out << '\n'
			<< " ---------------------------------------------------------------------------- \n"
			<< "| Reference count information                                                |\n"
			<< " ---------------------------------------------------------------------------- \n";
		
		for (Counts::iterator it = counts.begin();
			it != counts.end(); ++it)
		{
			const std::string name(it->first);
			const unsigned int creations    = it->second.first;
			const unsigned int destructions = it->second.second;
		
			out << "| " << name << " reference count information:\n"
			<< "|  Creations:    " << creations    << '\n'
			<< "|  Destructions: " << destructions << '\n';
		}
		
		out << " ---------------------------------------------------------------------------- \n";
		
		return out.str();
	}
	else
	{
		return "";
	}
}


