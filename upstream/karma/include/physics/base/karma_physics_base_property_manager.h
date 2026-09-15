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


#ifndef __karma_physics_base_property_manager_h__
#define __karma_physics_base_property_manager_h__

// C++ includes
#include <map>

// LibMesh includes
#include "vector_value.h"
#include "tensor_value.h"

#include "karma.h"

// Forward declarations
class KARMAParameters ;

/**
 * This class implements the KARMAPhysicsBasePropertyManager derived from KARMAOperator.
 * Essentially it is a base class for all custom physics based property evaluations
 * routines provided by user; These can vary according to test problem and hence 
 * provide a decoupled way to use same residual/jacobian operators with different
 * user provided property evaluation routines.
 *
 * @author Vijay M, 2009
 */

// template< typename T >
// struct karma_property_function
// {
// public:
	// typedef T (*function_pointer_signature)(const KARMAParameters& );
	
	// function_pointer_signature function_pointer ;
	
	// T operator () (const KARMAParameters& varparams) const 
	// {
		// return function_pointer(varparams);
	// }
	
	// karma_property_function ()
	// {
		// function_pointer = NULL ;
	// }
	
	// ~karma_property_function ()
	// {
		// function_pointer = NULL ;
	// }
	
// }

 
// template< typename T >
// typedef T (*karma_property_function)(const KARMAParameters& varparams);
// typedef Gradient (*parameterized_property_function)(const KARMAParameters& varparams);
// typedef Tensor (*parameterized_property_function)(const KARMAParameters& varparams);

// create explicit instantiations
// typedef parameterized_property_function karma_property_function<Real> ;

// generic property function definition
typedef Real (*parameterized_property_function)(const KARMAParameters& varparams);

// function definitions for exact solution, its gradient and the hessian.
typedef Real (*karma_property_function)(const KARMAParameters& varparams);
typedef Gradient (*karma_property_function_vector)(const KARMAParameters& varparams);
typedef Tensor (*karma_property_function_tensor)(const KARMAParameters& varparams);

// ------------------------------------------------------------
// KARMAPhysicsBasePropertyManager class definition
class KARMAPhysicsBasePropertyManager
{
public:

	/**
	*  Copy constructor. Initializes and copies the function pointers from the supplied object.
	*/
	KARMAPhysicsBasePropertyManager (const KARMAPhysicsBasePropertyManager& dummy) ;

	/**
	*  Equality operator. Performs a deep copy of the function pointers.
	*/
	KARMAPhysicsBasePropertyManager& operator = (const KARMAPhysicsBasePropertyManager& dummy) ;

	/**
	* Destoys all the function pointers allocated.
	*/
	virtual ~KARMAPhysicsBasePropertyManager ();

	void attach_exact_solution(karma_property_function m) ;
  
	void attach_exact_gradient(karma_property_function_vector m) ;
  
	void attach_exact_hessian(karma_property_function_tensor m) ;
  
	virtual bool is_ready() const = 0 ;
   
	Real ExactSolution (const KARMAParameters& varparams) const ;
	
	Gradient ExactGradient (const KARMAParameters& varparams) const ;
	
	Tensor ExactHessian (const KARMAParameters& varparams) const ;
	
	bool is_exact_solution_available() const ;

	bool is_exact_gradient_available() const ;

	bool is_exact_hessian_available() const ;
	
	
protected:

	/**
	*  Constructor. Initializes necessary data structures but scope is protected
	*  since we want the Physics to have concrete names for properties and be aware
	*  of the functions that the user supplies. 
	*  Philosophy: Don't take user functions that designer does not want.
	*/
	KARMAPhysicsBasePropertyManager ();

	// very general property container routines; derived class can implement specializations.
	void attach_property_function(const std::string& propname, parameterized_property_function& f) ;
	
	bool is_property_available(const std::string& propname) const  ;

	Real compute_property (const std::string& propname, const KARMAParameters& varparams) const ;
  
  
	// Protected member attributes
	std::map<std::string, parameterized_property_function> _prop_fnptrs ;
	
	typedef std::map<std::string, parameterized_property_function>::iterator property_iterator ;
	
	typedef std::map<std::string, parameterized_property_function>::const_iterator const_property_iterator ;
	
	// Special function pointers - exact solution, the gradient and the hessian
	karma_property_function _exact_solution ;
	karma_property_function_vector _exact_gradient ;
	karma_property_function_tensor _exact_hessian ;
	
};

// Inline functions for KARMAPhysicsBasePropertyManager

inline 
void KARMAPhysicsBasePropertyManager::attach_exact_solution(karma_property_function ex_f)
{
	karma_assert(ex_f != NULL) ;
	
	_exact_solution = ex_f ;
}

inline 
void KARMAPhysicsBasePropertyManager::attach_exact_gradient(karma_property_function_vector ex_g)
{
	karma_assert(ex_g != NULL) ;
	
	_exact_gradient = ex_g ;
}

inline 
void KARMAPhysicsBasePropertyManager::attach_exact_hessian(karma_property_function_tensor ex_h)
{
	karma_assert(ex_h != NULL) ;
	
	_exact_hessian = ex_h ;
}

  
inline
bool KARMAPhysicsBasePropertyManager::is_exact_solution_available() const 
{
	return (_exact_solution != NULL) ;
}

inline
bool KARMAPhysicsBasePropertyManager::is_exact_gradient_available() const 
{
	return (_exact_gradient != NULL) ;
}

inline
bool KARMAPhysicsBasePropertyManager::is_exact_hessian_available() const 
{
	return (_exact_hessian != NULL) ;
}

inline
Real KARMAPhysicsBasePropertyManager::ExactSolution (const KARMAParameters& varparams) const 
{
	karma_assert(is_exact_solution_available()) ;
	
	return (*_exact_solution)(varparams) ;
}

inline
Gradient KARMAPhysicsBasePropertyManager::ExactGradient (const KARMAParameters& varparams) const 
{
	karma_assert(is_exact_gradient_available()) ;
	
	return (*_exact_gradient)(varparams) ;
}

inline
Tensor KARMAPhysicsBasePropertyManager::ExactHessian (const KARMAParameters& varparams) const 
{
	karma_assert(is_exact_hessian_available()) ;
	
	return (*_exact_hessian)(varparams) ;
}


inline
bool KARMAPhysicsBasePropertyManager::is_property_available(const std::string& prop_name) const 
{
	const_property_iterator iter = _prop_fnptrs.find(prop_name);

	return ( iter != _prop_fnptrs.end() ) ;
}

inline
Real KARMAPhysicsBasePropertyManager::compute_property (const std::string& prop_name, const KARMAParameters& varparams) const 
{
	const_property_iterator iter = _prop_fnptrs.find(prop_name);

	if( iter != _prop_fnptrs.end() )
	{
		parameterized_property_function f = iter->second ;
	
		return (*f)(varparams) ;
	}
	else
	{
		KARMA_LOG(ERROR) << "Could not find the function corresponding to property = " << prop_name ;
		karma_error() ;
	}
}

inline
void KARMAPhysicsBasePropertyManager::attach_property_function(const std::string& prop_name, parameterized_property_function& f)
{
	karma_assert(f != NULL) ;
	
	_prop_fnptrs[prop_name] = f ;
}

#endif	// define __karma_physics_base_property_manager_h__
