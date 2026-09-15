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


#ifndef __lapack_utils_h__
#define __lapack_utils_h__

 /*         
  *         -------------------------------------------------
  *                LINEAR LEAST SQUARE SOLVER
  *         -------------------------------------------------
  *
  * 
  * This file has implementation of the LAPACK routine 'dgels', 'dgelsy' and 'dgelss'
  * for C/C++. This program solves the least square problem defined as follows:
  *
  * minimize  || Ax - B ||
  *     x 
  *     
  * please refer to man pages of 'dgels' for details.   
  *
  * void dgels(double **A, double *b, int m, int n, double *x);
  * void dgelsy(double **A, double *b, int m, int n, double *x);
  * void dgelss(double **A, double *b, int m, int n, double *x);
  * void dgelss_pi(double **A, double *b, int m, int n, double *x);
  *
  * 
  * A    : m by n matrix
  * b    : m by 1 matrix 
  * x    : n by 1 matrix for returned values
  *
  *
  *  dgels     : computes minimum norm solution when rank(A) = min(m,n)
  *  dgelsy    : computes minimum norm solution when rank(A) <= min(m,n) 
  *              uses complete orthogonal factorization of A
  *  dgelss    : computes minimum norm solution when rank(A) <= min(m,n) uses SVD
  *  dgelss_pi : computes the solution x = pinv(A)*b which is the NOT the minimum norm solution
  *
  *
  * Least Square Solution for Multiple columns
  *
  * minimize || A X - B ||
  *    X
  *
  *  --------------------------------------------------------------*/

#include <math.h>

#define MAXIMUM(m, n) (m) < (n) ? (n) : (m)
#define MINIMUM(m, n) (m) < (n) ? (m) : (n)

namespace KARMA { namespace Utils {

int dgels(double **A, double *b, int m, int n, double *x);
int dgels(double *A, double *b, int m, int n, double *x);

int dgelsy(double **A, double *b, int m, int n, double *x);
int dgelsy(double *A, double *b, int m, int n, double *x);

int dgelss(double **A, double *b, int m, int n, double *x);
int dgelss(double *A, double *b, int m, int n, double *x);
int dgelss(double **A, double **B, int m, int n, int p, double **X);

int dgelss_pi(double **A, double *b, int m, int n, double *x);
int dgelss_pi(double *A, double *b, int m, int n, double *x);

double* dgels_ctof(double *in, int rows, int cols);
double* dgels_ctof(double **in, int rows, int cols);
 
extern "C" void dgels_(char *trans, int *m, int *n, int *nrhs, double *a, int *lda,
		       double *b, int *ldb, double *work, int *lwork, int *info);

extern "C" void dgelsy_(int *m, int *n, int *nrhs, double *a, int *lda,
		       double *b, int *ldb, int *JPVT, double *RCOND, int *RANK,
             double *work, int *lwork, int *info);
 
extern "C" void dgelss_(int *m, int *n, int *nrhs, double *a, int *lda,
		       double *b, int *ldb, double *s, double *RCOND, int *RANK,
             double *work, int *lwork, int *info);

}	}


#endif // #define __lapack_utils_h__
