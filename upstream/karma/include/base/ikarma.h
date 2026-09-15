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



#ifndef __ikarma_h__
#define __ikarma_h__

#include <typeinfo>
#include <string>
#include <map>

/**
 * This class will serve as the base class for all complex interfaces
 * in KARMA. It maintains reference counting information, if requested.
 * Any class that is properly derived from this class will get reference
 * counted by default.
 */

// ------------------------------------------------------------
// GlobalReferenceCounter class definition
class GlobalReferenceCounter
{
public:

	/**
	* Destructor.
	*/
	virtual ~GlobalReferenceCounter ()
	{
		n_objects--;
	}

	/**
	* Prints the number of outstanding (created, but not yet
	* destroyed) objects.
	*/
	static unsigned int objects_in_memory ()
	{ return n_objects; }

	/**
	* Gets a string containing the reference information.
	*/
	static std::string get_info ();

protected:

	GlobalReferenceCounter()
	{
		n_objects++;
	}
	
	void increment_type_count(const std::string& type_name)
	{
		if (enable_counter)
		{
			std::pair<unsigned int, unsigned int>& p = counts[type_name];
			p.second++;
		}
	}
	
	void decrement_type_count(const std::string& type_name)
	{
		if (enable_counter)
		{
			std::pair<unsigned int, unsigned int>& p = counts[type_name];
			p.second--;
		}
	}
	
	/**
	* The number of objects.  Print the reference count
	* information when the number returns to 0.
	*/
	static unsigned int n_objects;

	/**
	* Data structure to log the information.  The log is
	* identified by the class name.
	*/
	typedef std::map<std::string, std::pair<unsigned int,
						unsigned int> > Counts;
	
	/**
	* Actually holds the data.
	*/
	static Counts counts;
	
	/**
	 * Flag to control whether reference count information
	 * is printed when print_info is called.
	 */
	static bool enable_counter;


};

template <typename T>
class IKARMA : public GlobalReferenceCounter
{
public:

	/**
	* Destructor.
	*/
	virtual ~IKARMA ()
	{
		decrement_type_count(typeid(T).name()) ;
	}

protected:

	/**
	 * Constructor. Protected so that you cannont
	 * instantiate a \p ReferenceCountedObject, only derive
	 * from it.
	 */
	IKARMA ()	: GlobalReferenceCounter ()
	{
		increment_type_count(typeid(T).name()) ;
	}

private:

};

#endif	// __ikarma_h__
