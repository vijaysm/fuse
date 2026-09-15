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



#ifndef __karma_eigen_solver_h__
#define __karma_eigen_solver_h__

#ifdef LIBMESH_HAVE_SLEPC

// C++ includes

// Local includes
#include "ikarma.h"
//#include "libmesh_common.h"
#include "enum_eigen_solver_type.h"
#include "reference_counted_object.h"
#include "karma_vector.h"
#include "karma_matrix.h"

// forward declarations
template <typename T> class AutoPtr;
template <typename T> class ShellMatrix;

/**
 * SLEPc include files. SLEPc can only be used
 * together with PETSc.
 */
# include <slepceps.h>
# include <slepcversion.h>


// A convenient macro for comparing SLEPc versions.
// Returns 1 if the current SLEPc version is < major.minor.subminor
// and zero otherwise.
#define SLEPC_VERSION_LESS_THAN(major,minor,subminor)			            \
  ((SLEPC_VERSION_MAJOR < (major) ||						    \
    (SLEPC_VERSION_MAJOR == (major) && (SLEPC_VERSION_MINOR < (minor) ||	    \
				  (SLEPC_VERSION_MINOR == (minor) &&		    \
				   SLEPC_VERSION_SUBMINOR < (subminor))))) ? 1 : 0)



/**
 * This class provides an interface to the SLEPc
 * eigenvalue solver library \p www.grycap.upv.es/slepc/.
 */

template <typename T>
class KARMAEigenSolver : public IKARMA<KARMAEigenSolver<T> >
{

public:

  /**
   *  Constructor. Initializes Petsc data structures
   */
  KARMAEigenSolver();
    

  /**
   * Destructor.
   */
  ~KARMAEigenSolver();

  /**
   * @returns true if the data structures are
   * initialized, false otherwise.
   */
  bool initialized () const { return _is_initialized; }
  
  
  /**
   * Release all memory and clear data structures.
   */
  void clear();


  /**
   * Initialize data structures if not done so already.
   */
  void init();

  /**
   * Returns the type of eigensolver to use.
   */
  EigenSolverType eigen_solver_type () const { return _eigen_solver_type; }

  /**
   * Returns the type of the eigen problem.
   */
  EigenProblemType eigen_problem_type () const { return _eigen_problem_type;}

  /**
   * Returns the position of the spectrum to compute.
   */
  PositionOfSpectrum postition_of_spectrum () const
    { return _position_of_spectrum;}

  /**
   * Sets the type of eigensolver to use.
   */
  void set_eigensolver_type (const EigenSolverType est)
    { _eigen_solver_type = est; }

  /**
   * Sets the type of the eigenproblem.
   */
  void set_eigenproblem_type ( EigenProblemType ept) 
    {_eigen_problem_type = ept;}

  /**
   * Sets the position of the spectrum.
   */
  void set_position_of_spectrum (PositionOfSpectrum pos)
    {_position_of_spectrum= pos;}

       
  /**
   * This function calls the SLEPc solver to compute
   * the eigenpairs of the KARMAMatrix matrix_A. \p nev is
   * the number of eigenpairs to be computed and
   * \p ncv is the number of basis vectors to be
   * used in the solution procedure. Return values
   * are the number of converged eigen values and the
   * number of the iterations carried out by the eigen
   * solver.
   */    
  std::pair<unsigned int, unsigned int>  solve_standard (KARMAMatrix &matrix_A,
							 int nev,
							 int ncv,
							 const double tol,
							 const unsigned int m_its);

  /**
   * Same as above except that matrix_A is a ShellMatrix
   * in this case.
   */
  std::pair<unsigned int, unsigned int>  solve_standard (ShellMatrix<T> &shell_matrix,
							 int nev,
							 int ncv,
							 const double tol,
							 const unsigned int m_its);
  

 /**
   * This function calls the SLEPc solver to compute
   * the eigenpairs for the generalized eigenproblem
   * defined by the matrix_A and matrix_B,
   * which are of type KARMAMatrix. The argument 
   * \p nev is the number of eigenpairs to be computed
   * and \p ncv is the number of basis vectors to be
   * used in the solution procedure. Return values
   * are the number of converged eigen values and the
   * number of the iterations carried out by the eigen
   * solver.
   */
  std::pair<unsigned int, unsigned int>  solve_generalized(KARMAMatrix &matrix_A,
							   KARMAMatrix &matrix_B,
							   int nev,
							   int ncv,
							   const double tol,
							   const unsigned int m_its);

  /**
   * Solve generalized eigenproblem when matrix_A is of
   * type ShellMatrix, matrix_B is of type KARMAMatrix.
   */
    std::pair<unsigned int, unsigned int>  solve_generalized(ShellMatrix<T> &matrix_A,
  							   KARMAMatrix &matrix_B,
  							   int nev,
  							   int ncv,
  							   const double tol,
  							   const unsigned int m_its);

  /**
   * Solve generalized eigenproblem when matrix_A is of
   * type KARMAMatrix, matrix_B is of type ShellMatrix.
   * When using this function, one should use the
   * command line options:
   * -st_ksp_type gmres -st_pc_type none
   * or
   * -st_ksp_type gmres -st_pc_type jacobi
   * or similar.
   */
    std::pair<unsigned int, unsigned int>  solve_generalized(KARMAMatrix &matrix_A,
  							   ShellMatrix<T> &matrix_B,
  							   int nev,
  							   int ncv,
  							   const double tol,
  							   const unsigned int m_its);

  /**
   * Solve generalized eigenproblem when both matrix_A and
   * matrix_B are of type ShellMatrix.
   * When using this function, one should use the
   * command line options:
   * -st_ksp_type gmres -st_pc_type none
   * or
   * -st_ksp_type gmres -st_pc_type jacobi
   * or similar.
   */
    std::pair<unsigned int, unsigned int>  solve_generalized(ShellMatrix<T> &matrix_A,
  							   ShellMatrix<T> &matrix_B,
  							   int nev,
  							   int ncv,
  							   const double tol,
  							   const unsigned int m_its);

 
 
  /**
   * This function returns the real and imaginary part of the
   * ith eigenvalue and copies the respective eigenvector to the
   * solution vector. Note that also in case of purely real matrix
   * entries the eigenpair may be complex values.
   */                
  std::pair<Real, Real> get_eigenpair (unsigned int i,
				       KARMAVector &solution_in);

  /**
   * @computes and returns the relative error ||A*x-lambda*x||/|lambda*x|
   * of the ith eigenpair. (or the equivalent for a general eigenvalue problem)
   */                
  Real get_relative_error (unsigned int i);

  
private:

  /**
   * Helper function that actually performs the standard eigensolve.
   */
  std::pair<unsigned int, unsigned int>  _solve_standard_helper (Mat mat,
							 int nev,
							 int ncv,
							 const double tol,
							 const unsigned int m_its);

  /**
   * Helper function that actually performs the generalized eigensolve.
   */
  std::pair<unsigned int, unsigned int>  _solve_generalized_helper (Mat mat_A,
							         Mat mat_B,
							         int nev,
							         int ncv,
							         const double tol,
							         const unsigned int m_its);

  /**
   * Tells Slepc to use the user-specified solver stored in
   * \p _eigen_solver_type
   */
  void set_slepc_solver_type ();

  /**
   * Tells Slepc to deal with the type of problem stored in
   * \p _eigen_problem_type
   */
  void set_slepc_problem_type ();

  /**
   * Tells Slepc to compute the spectrum at the position
   * stored in \p _position_of_spectrum
   */
  void set_slepc_position_of_spectrum();

  /**
   * Internal function if shell matrix mode is used, this just
   * calls the shell matrix's matrix multiplication function.
   * See PetscLinearSolver for a similar implementation.
   */
  static PetscErrorCode _petsc_shell_matrix_mult(Mat mat, Vec arg, Vec dest);

  /**
   * Internal function if shell matrix mode is used, this just
   * calls the shell matrix's get_diagonal function.
   * Required in order to use Jacobi preconditioning.
   */
  static PetscErrorCode _petsc_shell_matrix_get_diagonal(Mat mat, Vec dest);

  /**
   * Eigenproblem solver context
   */
  EPS _eps;

  /**
   * Enum stating which type of eigensolver to use.
   */
  EigenSolverType _eigen_solver_type;

  /**
   * Enum stating which type of eigen problem we deal with.
   */
  EigenProblemType _eigen_problem_type;

  /**
   * Enum stating where to evaluate the spectrum.
   */
  PositionOfSpectrum _position_of_spectrum;	

  /**
   * Flag indicating if the data structures have been initialized.
   */
  bool _is_initialized;


};


/*----------------------- inline functions ----------------------------------*/
template <typename T>
inline
KARMAEigenSolver<T>::KARMAEigenSolver () :
	_eigen_solver_type    (KRYLOVSCHUR),
	_eigen_problem_type   (NHEP),
	_position_of_spectrum (LARGEST_MAGNITUDE),
	_is_initialized       (false)
{
}



template <typename T>
inline
KARMAEigenSolver<T>::~KARMAEigenSolver ()
{
  this->clear ();
}

#endif

#endif // #ifdef __karma_linear_solver_h__
