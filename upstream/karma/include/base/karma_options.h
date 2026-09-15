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


#ifndef __karma_options_h__
#define __karma_options_h__


#define PETSC_OPTIONS
#define SLEPC_OPTIONS
#define LIBMESH_OPTIONS
#define KARMA_OPTIONS

//###########################
//  KARMA configuration file
//###########################

// PETSc configuration options
#ifdef PETSC_OPTIONS

#define PETSC_HELP "-help"         //  unlisted but active

//#### memory checking
#define PETSC_MALLOC_DUMP "-malloc_dump"	// -malloc_dump <optional filename>: dump list of unfreed memory at conclusion
#define PETSC_MALLOC "-malloc"			// use our error checking malloc
#define PETSC_MALLOC_NO "-malloc no"	// don't use error checking malloc
#define PETSC_MALLOC_LOG "-malloc_log"	//  unlisted but active
#define PETSC_MALLOC_INFO "-malloc_info"	// prints total memory usage
#define PETSC_MALLOC_DEBUG "-malloc_debug"	// enables extended checking for memory corruption

#define PETSC_FP_TRAP "-fp_trap"

// print info at end of run

// total memory used
#define PETSC_MEMORY_INFO "-memory_info"

// info about the options
#define PETSC_OPTIONS_TABLE "-options_table"
#define PETSC_OPTIONS_LEFT "-options_left"

// print stats
#define PETSC_TOTAL_FLOPS "-get_total_flops"	//  total flops over all processors

//  print even more info (also during run)
#define PETSC_LOG_SUMMARY "-log_summary"
#define PETSC_INFO "-info"	// <optional filename>: print informative messages about the calculations
#define PETSC_LOG_TRACE "-log_trace"	// //  -log_trace [filename]: prints trace of all PETSc calls
#define PETSC_MEMORY_INFO "-memory_info"	//  print memory usage at end of run

// 
//  shortcuts for analyzing malloc information
// 
//  cat %f | grep bytes | grep -v "src/" > %f.b
//  cat %f | grep bytes | colrm 1 4 | sort -n > %f.c
//  cat %f | grep bytes | grep "src/" > %f.d
//  grep mallocs %f | grep -v " 0 mallocs" | grep -v "is 0" | less


// Options for all PETSc programs:

#define PETSC_ON_ERROR_ABORT "-on_error_abort" 	//  cause an abort when an error is detected. Useful
											//  only when run in the debugger
#define PETSC_ATTACH_DEBUGGER_ON_ERROR "-on_error_attach_debugger" 	//   [gdb,dbx,xxgdb,ups,noxterm]
														//        start the debugger in new xterm
														//        unless noxterm is given
#define PETSC_START_IN_DEBUGGER "-start_in_debugger"	// [gdb,dbx,xxgdb,ups,noxterm]
												//  start all processes in the debugger

// Vector Options -------------------------------------------------

#define PETSC_VEC_VIEW "-vec_view" //   Print vector to stdout (VecView)
#define PETSC_VEC_VIEW_MATLAB_FORMAT "-vec_view_matlab" //   Print vector to stdout in a format Matlab can read (VecView)
#define PETSC_VEC_VIEW_BINARY_FORMAT "-vec_view_binary" //   Save vector to file in binary format (VecView)

// Options for SEQAIJ matrix -------------------------------------------------

#define PETSC_MAT_NO_UNROLL "-mat_no_unroll"	 // <TRUE> Do not optimize for inodes (slower) ((null))
#define PETSC_MAT_NO_INODE "-mat_no_inode"		 // <TRUE> Do not optimize for inodes (slower) ((null))
#define PETSC_MAT_INODE_LIMIT "-mat_inode_limit <5>"	// Do not use inodes larger then this value ((null))

// Matrix Options -------------------------------------------------

#define PETSC_MAT_VIEW "-mat_view"	// Print matrix to stdout (MatView)
#define PETSC_MAT_INFO "-mat_view_info"	// Information on matrix size (MatView)
#define PETSC_MAT_INFO_DETAILED "-mat_view_info_detailed"	// Nonzeros in the matrix (MatView)
#define PETSC_MAT_VIEW_MATLAB_FORMAT "-mat_view_matlab"	//  Print matrix to stdout in a format Matlab can read (MatView)
#define PETSC_MAT_VIEW_BINARY_FORMAT "-mat_view_binary"	// Save matrix to file in binary format (MatView)
#define PETSC_MAT_VIEW_DRAW "-mat_view_draw"	// Draw the matrix nonzero structure (MatView)
 
// Graphics (PetscDraw) Options -------------------------------------------------
#define PETSC_DRAW_TYPE "-draw_type"	//  Type of graphical output:(one of) x null ps (PetscDrawSetType)

// Krylov Method (KSP) Options -------------------------------------------------

#define PETSC_KSP_TYPE "-ksp_type"	// Krylov method:(one of) cg cgne stcg richardson chebychev gmres tcqmr bcgs
									//  bcgsl cgs tfqmr cr lsqr preonly qcg bicg fgmres minres symmlq lgmres lcd (KSPSetType)
#define PETSC_KSP_VIEW "-ksp_view" //  View linear solver parameters (KSPView)
#define PETSC_KSP_MAX_ITER "-ksp_max_it"	// <10000>: Maximum number of iterations (KSPSetTolerances)
#define PETSC_KSP_RTOL "-ksp_rtol"	// <1e-05>: Relative decrease in residual norm (KSPSetTolerances)
#define PETSC_KSP_ATOL "-ksp_atol" 	// <1e-50>: Absolute value of residual norm (KSPSetTolerances)
#define PETSC_KSP_DIVTOL "-ksp_divtol"	//  <10000>: Residual norm increase cause divergence (KSPSetTolerances)
#define PETSC_KSP_KNOLL_INIT_GUESS "-ksp_knoll" //  <FALSE> Use preconditioner applied to b for initial guess (KSPSetInitialGuessKnoll)
#define PETSC_KSP_NORM_TYPE "-ksp_norm_type"  //   <preconditioned> (choose one of) none preconditioned unpreconditioned natural
#define PETSC_KSP_DIAGONAL_SCALE "-ksp_diagonal_scale" // Diagonal scale matrix before building preconditioner (KSPSetDiagonalScale)
#define PETSC_KSP_DIAGONAL_SCALE_FIX "-ksp_diagonal_scale_fix" //  Fix diagonaled scaled matrix after solve (KSPSetDiagonalScaleFix)
#define PETSC_KSP_CONVERGED_REASON "-ksp_converged_reason" //  Print reason for converged or diverged (KSPSolve)
#define PETSC_KSP_CANCEL_MONITORS "-ksp_cancelmonitors" //  Remove any hardwired monitor routines (KSPClearMonitor)
#define PETSC_KSP_MONITOR "-ksp_monitor" //  <stdout>: Monitor (DEFAULT) preconditioned residual norm (KSPSetMonitor)
#define PETSC_KSP_VEC_DRAW_MONITOR "-ksp_vecmonitor" //  Monitor solution graphically (KSPSetMonitor)
#define PETSC_KSP_MONITOR_TRUE_NORM "-ksp_truemonitor" // <stdout>: Monitor preconditioned residual norm (KSPSetMonitor)
//   -ksp_singmonitor <stdout>: Monitor singular values (KSPSetMonitor)
//   -ksp_smonitor <stdout>: Monitor preconditioned residual norm with fewer digits (KSPSetMonitor)
//   -ksp_xmonitor: Monitor graphically preconditioned residual norm (KSPSetMonitor)
//   -ksp_xtruemonitor: Monitor graphically true residual norm (KSPSetMonitor)

//   Pick at most one of PETSC_KSP_PC_SIDE
#define PETSC_KSP_LEFT_PC "-ksp_left_pc"	//  Use left preconditioning (KSPSetPreconditionerSide)
#define PETSC_KSP_RIGHT_PC "-ksp_right_pc" //  Use right preconditioning (KSPSetPreconditionerSide)
#define PETSC_KSP_SYMMETRIC_PC "-ksp_symmetric_pc" //  Use symmetric (factorized) preconditioning (KSPSetPreconditionerSide)
//

#define PETSC_KSP_COMPUTE_SINGULAR_VALUES "-ksp_compute_singularvalues" //  Compute singular values of preconditioned operator (KSPSetComputeSingularValues)
#define PETSC_KSP_COMPUTE_EIGEN_VALUES "-ksp_compute_eigenvalues" //  Compute eigenvalues of preconditioned operator (KSPSetComputeSingularValues)
#define PETSC_KSP_PLOT_EIGEN_VALUES "-ksp_plot_eigenvalues" //  Scatter plot extreme eigenvalues (KSPSetComputeSingularValues)

//   KSP GMRES Options
#define PETSC_KSP_GMRES_RESTART "-ksp_gmres_restart"	//  <30>: Number of Krylov search directions (KSPGMRESSetRestart)
#define PETSC_KSP_GMRES_HAPPY_ENDING "-ksp_gmres_haptol"	// <1e-30>: Tolerance for exact convergence (happy ending) (KSPGMRESSetHapTol)
#define PETSC_KSP_GMRES_PREALLOCATE "-ksp_gmres_preallocate" // Preallocate Krylov vectors (KSPGMRESSetPreAllocateVectors)

//   Pick at most one of Orthogonalization methods
#define PETSC_KSP_GMRES_CLASSICAL_GRAMSCHMIDT "-ksp_gmres_classicalgramschmidt"	//  Classical (unmodified) Gram-Schmidt (fast) (KSPGMRESSetOrthogonalization)
#define PETSC_KSP_GMRES_MODIFIED_GRAMSCHMIDT "-ksp_gmres_modifiedgramschmidt" //  Modified Gram-Schmidt (slow,more stable) (KSPGMRESSetOrthogonalization)

#define PETSC_KSP_GMRES_CGS_REFINEMENT "-ksp_gmres_cgs_refinement_type"	// <REFINE_NEVER> (choose one of) REFINE_NEVER REFINE_IFNEEDED REFINE_ALWAYS
#define PETSC_KSP_GMRES_KRYLOV_MONITOR "-ksp_gmres_krylov_monitor" //  Plot the Krylov directions (KSPSetMonitor)


// Preconditioner (PC) Options -------------------------------------------------

#define PETSC_PC_TYPE "-pc_type" //   Preconditioner:(one of) none jacobi pbjacobi bjacobi sor lu shell mg
								 //       eisenstat ilu icc cholesky asm ksp composite redundant nn mat fieldsplit tfs
								 //			hypre

//   ILU and ICC Options
#define PETSC_PC_FACTOR_LEVELS "-pc_factor_levels"	//  <0>: levels of fill (PCFactorSetLevels)
#define PETSC_PC_FACTOR_FILL "-pc_factor_fill"		//  <1>: Expected fill in factorization (PCFactorSetFill)
#define PETSC_PC_FACTOR_MAT_ORDERING "-pc_factor_mat_ordering_type"	//  Reorder to reduce nonzeros in ICC:(one of) natural nd 1wd rcm qmd rowlength (PCFactorSetMatOrdering)

#define PETSC_PC_FACTOR_SHIFT_NZ "-pc_factor_shift_nonzero" // Shift added to diagonal (PCFactorSetShiftNonzero)
														// When using this option, never use "ksp_type preonly"
														// but always use either ksp_type cg or gmres.
														// Optionally <0>: Shift added to diagonal (PCFactorSetShiftNonzero)

#define PETSC_PC_FACTOR_SHIFT_PD "-pc_factor_shift_positive_definite" //  Manteuffel shift applied to diagonal (PCFactorSetShift)

//	  HYPRE preconditioner options

#define PETSC_PC_HYPRE_TYPE "-pc_hypre_type"	// <pilut> (choose one of) pilut parasails boomeramg euclid

// HYPRE Pilut Options
#define PETSC_PC_HYPRE_PILUT_MAXITER "-pc_hypre_pilut_maxiter"	// <-2>: Number of iterations (None)
#define PETSC_PC_HYPRE_PILUT_DROPTOL "-pc_hypre_pilut_tol"	// <-2>: Drop tolerance (None)

//	HYPRE Euclid Options
#define PETSC_PC_HYPRE_EUCLID_LEVELS "-pc_hypre_euclid_levels"	// <1>: Number of levels of fill ILU(k) (None)
#define PETSC_PC_HYPRE_EUCLID_BLOCK_JACOBI "-pc_hypre_euclid_bj"	//  <FALSE> Use block Jacobi ILU(k) (None)
#define PETSC_PC_HYPRE_EUCLID_STATS "-pc_hypre_euclid_print_statistics" // <FALSE> Print statistics (None)

//	HYPRE BoomerAMG Options
#define PETSC_PC_HYPRE_BOOMERAMG_CYCLE "-pc_hypre_boomeramg_cycle_type"	// <V> (choose one of)  V
#define PETSC_PC_HYPRE_BOOMERAMG_LEVELS "-pc_hypre_boomeramg_max_levels"	// <25>: Number of levels (of grids) allowed (None)
#define PETSC_PC_HYPRE_BOOMERAMG_MAX_ITER "-pc_hypre_boomeramg_max_iter"	// <1>: Maximum iterations used PER hypre call (None)
#define PETSC_PC_HYPRE_BOOMERAMG_TOL "-pc_hypre_boomeramg_tol"	// <0>: Convergence tolerance PER hypre call (0.0 = use a fixed number of iterations) (None)
#define PETSC_PC_HYPRE_BOOMERAMG_TRUNCFACTOR "-pc_hypre_boomeramg_truncfactor"	// <0>: Truncation factor for interpolation (0=no truncation) (None)
#define PETSC_PC_HYPRE_BOOMERAMG_PMAX "-pc_hypre_boomeramg_P_max"	// <0>: Max elements per row for interpolation operator ( 0=unlimited ) (None)
#define PETSC_PC_HYPRE_BOOMERAMG_AGG_LEVELS "-pc_hypre_boomeramg_agg_nl"	// <0>: Number of levels of aggressive coarsening (None)
#define PETSC_PC_HYPRE_BOOMERAMG_AGG_PATHS "-pc_hypre_boomeramg_agg_num_paths"	// <1>: Number of paths for aggressive coarsening (None)
#define PETSC_PC_HYPRE_BOOMERAMG_THRESHOLD "-pc_hypre_boomeramg_strong_threshold"	// <0.25>: Threshold for being strongly connected (None)
#define PETSC_PC_HYPRE_BOOMERAMG_MAX_ROWSUM "-pc_hypre_boomeramg_max_row_sum"	// <0.9>: Maximum row sum (None)

#define PETSC_PC_HYPRE_BOOMERAMG_RELAX_WEIGHT_LEVEL "-pc_hypre_boomeramg_relax_weight_level"	// <1>: Set the relaxation weight for a particular level (weight,level) (None)

#define PETSC_PC_HYPRE_BOOMERAMG_SWEEPS_ALL "-pc_hypre_boomeramg_grid_sweeps_all"	// <1>: Number of sweeps for the up and down grid levels (None)
#define PETSC_PC_HYPRE_BOOMERAMG_RELAX_ALL "-pc_hypre_boomeramg_relax_type_all"	// <symmetric-SOR/Jacobi> (choose one of) Jacobi sequential-Gauss-Seidel  SOR/Jacobi backward-SOR/Jacobi  symmetric-SOR/Jacobi   Gaussian-elimination
#define PETSC_PC_HYPRE_BOOMERAMG_RELAX_WEIGHT_ALL "-pc_hypre_boomeramg_relax_weight_all"	// <1>: Relaxation weight for all levels (0 = hypre estimates, -k = determined with k CG steps) (None)

#define PETSC_PC_HYPRE_BOOMERAMG_OUTER_RELAX_WEIGHT_ALL "-pc_hypre_boomeramg_outer_relax_weight_all"	// <1>: Outer relaxation weight for all levels ( -k = determined with k CG steps) (None)
#define PETSC_PC_HYPRE_BOOMERAMG_OUTER_RELAX_WEIGHT_LEVEL "-pc_hypre_boomeramg_outer_relax_weight_level"	// <1>: Set the outer relaxation weight for a particular level (weight,level) (None)

#define PETSC_PC_HYPRE_BOOMERAMG_SWEEPS_DOWN "-pc_hypre_boomeramg_grid_sweeps_down"	// <1>: Number of sweeps for the down cycles (None)
#define PETSC_PC_HYPRE_BOOMERAMG_RELAX_DOWN "-pc_hypre_boomeramg_relax_type_down"	// <symmetric-SOR/Jacobi> (choose one of) Jacobi sequential-Gauss-Seidel  SOR/Jacobi backward-SOR/Jacobi  symmetric-SOR/Jacobi   Gaussian-elimination

#define PETSC_PC_HYPRE_BOOMERAMG_SWEEPS_UP "-pc_hypre_boomeramg_grid_sweeps_up"	// <1>: Number of sweeps for the up cycles (None)
#define PETSC_PC_HYPRE_BOOMERAMG_RELAX_UP "-pc_hypre_boomeramg_relax_type_up"	// <symmetric-SOR/Jacobi> (choose one of) Jacobi sequential-Gauss-Seidel  SOR/Jacobi backward-SOR/Jacobi  symmetric-SOR/Jacobi   Gaussian-elimination

#define PETSC_PC_HYPRE_BOOMERAMG_SWEEPS_COARSE "-pc_hypre_boomeramg_grid_sweeps_coarse"	// <1>: Number of sweeps for the coarse level (None)
#define PETSC_PC_HYPRE_BOOMERAMG_RELAX_COARSE "-pc_hypre_boomeramg_relax_type_coarse"	// <Gaussian-elimination> (choose one of) Jacobi sequential-Gauss-Seidel  SOR/Jacobi backward-SOR/Jacobi  symmetric-SOR/Jacobi   Gaussian-elimination

#define PETSC_PC_HYPRE_BOOMERAMG_NO_CF "-pc_hypre_boomeramg_no_CF"	// <FALSE> Do not use CF-relaxation (None)
#define PETSC_PC_HYPRE_BOOMERAMG_MEASURE_TYPE "-pc_hypre_boomeramg_measure_type"	// <local> (choose one of) local global
#define PETSC_PC_HYPRE_BOOMERAMG_COARSEN_TYPE "-pc_hypre_boomeramg_coarsen_type"	// <Falgout> (choose one of) CLJP Ruge-Stueben  modifiedRuge-Stueben   Falgout  PMIS  HMIS
#define PETSC_PC_HYPRE_BOOMERAMG_INTERP_TYPE "-pc_hypre_boomeramg_interp_type"	// <classical> (choose one of) classical   direct multipass multipass-wts ext+i ext+i-cc standard standard-wts   FF FF1

#define PETSC_PC_HYPRE_BOOMERAMG_STATS "-pc_hypre_boomeramg_print_statistics"	// Print statistics (None)
#define PETSC_PC_HYPRE_BOOMERAMG_NODAL_COARSEN "-pc_hypre_boomeramg_nodal_coarsen"	// <FALSE> HYPRE_BoomerAMGSetNodal() (None)
#define PETSC_PC_HYPRE_BOOMERAMG_NODAL_RELAXATION "-pc_hypre_boomeramg_nodal_relaxation"	// <FALSE> Nodal relaxation via Schwarz (None)

//	Nonlinear solver (SNES) options -------------------------------------------------
  
#define PETSC_SNES_TYPE "-snes_type"	// Nonlinear solver method:(one of) ls tr test (SNESSetType)
#define PETSC_SNES_VIEW "-snes_view" 	// Print detailed information on solver used (SNESView)
#define PETSC_SNES_STOL "-snes_stol"	// <1e-08>: Stop if step length less then (SNESSetTolerances)
#define PETSC_SNES_ATOL "-snes_atol"	// <1e-50>: Stop if function norm less then (SNESSetTolerances)
#define PETSC_SNES_RTOL "-snes_rtol"	// <1e-08>: Stop if decrease in function norm less then (SNESSetTolerances)
#define PETSC_SNES_MAX_ITER "-snes_max_it"	// <50>: Maximum iterations (SNESSetTolerances)
#define PETSC_SNES_MAX_FUN_EVALS "-snes_max_funcs"	// <10000>: Maximum function evaluations (SNESSetTolerances)
#define PETSC_SNES_MAX_FAILURES "-snes_max_fail" // <1>: Maximum failures (SNESSetTolerances)

#define PETSC_SNES_CONVERGED_REASON "-snes_converged_reason"	// Print reason for converged or diverged (SNESSolve)
#define PETSC_SNES_NO_CONVERGENCE_TEST "-snes_no_convergence_test"	// Don't test for convergence (None)

// Eisenstat-Walker = Inexact Newton iteration
#define PETSC_SNES_USE_EW "-snes_ksp_ew"	// <TRUE> Use Eisentat-Walker linear system convergence test (SNESKSPSetUseEW)
#define PETSC_SNES_EW_VERSION "-snes_ksp_ew_version" // <2>: Version 1, 2 or 3 (SNESKSPSetParametersEW)
#define PETSC_SNES_EW_RTOL0 "-snes_ksp_ew_rtol0" 	// <0.3>: 0 <= rtol0 < 1 (SNESKSPSetParametersEW)
#define PETSC_SNES_EW_RTOLMAX "-snes_ksp_ew_rtolmax"	// <0.9>: 0 <= rtolmax < 1 (SNESKSPSetParametersEW)
#define PETSC_SNES_EW_GAMMA "-snes_ksp_ew_gamma"	// <1>: 0 <= gamma <= 1 (SNESKSPSetParametersEW)
#define PETSC_SNES_EW_ALPHA "-snes_ksp_ew_alpha"	// <1.61803>: 1 < alpha <= 2 (SNESKSPSetParametersEW)
#define PETSC_SNES_EW_ALPHA2 "-snes_ksp_ew_alpha2"	// <1.61803>: alpha2 (SNESKSPSetParametersEW)
#define PETSC_SNES_EW_THRESHOLD "-snes_ksp_ew_threshold"	// <0.1>: 0 < threshold < 1 (SNESKSPSetParametersEW)
//

#define PETSC_SNES_NO_MONITOR_CANCEL "-snes_monitor_cancel"	//  Remove all monitors (SNESMonitorCancel)
#define PETSC_SNES_MONITOR "-snes_monitor"	//  <stdout>: Monitor norm of function (SNESMonitorSet)
#define PETSC_SNES_MONITOR_SHORT "-snes_monitor_short"	// <stdout>: Monitor norm of function (fewer digits) (SNESMonitorSet)
#define PETSC_SNES_MONITOR_SOLUTION "-snes_monitor_solution"	// Plot solution at each iteration (SNESMonitorSolution)
#define PETSC_SNES_MONITOR_UPDATE "-snes_monitor_solution_update"	// Plot correction at each iteration (SNESMonitorSolutionUpdate)
#define PETSC_SNES_MONITOR_RESIDUAL "-snes_monitor_residual"	// Plot residual at each iteration (SNESMonitorResidual)
#define PETSC_SNES_MONITOR_DRAW "-snes_monitor_draw"	//  Plot function norm at each iteration (SNESMonitorLG)

#define PETSC_SNES_FD "-snes_fd"	// Use finite differences (slow) to compute Jacobian (SNESDefaultComputeJacobian)

// SNES Line search options
#define PETSC_SNES_LS "-snes_ls"	// <cubic> (choose one of) basic basicnonorms quadratic cubic
#define PETSC_SNES_LS_ALPHA "-snes_ls_alpha"	// <0.0001>: Function norm must decrease by (None)
#define PETSC_SNES_LS_MAXSTEP "-snes_ls_maxstep"	// <1e+08>: Step must be less than (None)
#define PETSC_SNES_LS_STEPTOL "-snes_ls_steptol"	// <1e-12>: Step must be greater than (None)


#endif

// SLEPc Options
#ifdef SLEPC_OPTIONS

#endif

// KARMA Options
#ifdef LIBMESH_OPTIONS

#endif

// KARMA Options
#ifdef KARMA_OPTIONS

#define KARMA_VERBOSE "-v"
#define KARMA_OPTIONS_FILE "-options_file <VALUE>"	//  reads options from file

#define PETSC_SNES_NO_USE_EW "-snes_ksp_ew false"	// <FALSE> Use Eisentat-Walker linear system convergence test (SNESKSPSetUseEW)

#define KARMA_USER_PRECONDITIONING "-snes_mf -user_precond"	// Use the User-preconditioning with matrix-free
															// jacobian representation.

#endif

// Use PetscOptionsInsertString to isnert into PetscOptions table from a string
// Use PetscOptionsHasName to find if an option is specified

#endif // #define __karma_options_h__
