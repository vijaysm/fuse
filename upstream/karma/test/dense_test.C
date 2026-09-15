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
// Tests for the Eigen-backed geometry and element-local dense algebra that
// replace libMesh's Point/RealGradient and DenseMatrix/DenseVector.

#include "karma_dense.h"
#include "karma_geometry.h"

#include <cmath>
#include <iostream>
#include <string>

namespace
{
	int failures = 0;

	void check (bool ok, const std::string& what)
	{
		if (!ok)
		{
			std::cerr << "FAIL: " << what << std::endl;
			++failures;
		}
	}

	void close (KARMA::Real a, KARMA::Real b, const std::string& what,
	            KARMA::Real tol = 1.e-12)
	{
		if (std::abs(a - b) > tol)
		{
			std::cerr << "FAIL: " << what
			          << " (got " << a << ", expected " << b << ")" << std::endl;
			++failures;
		}
	}
}

int main ()
{
	using namespace KARMA;

	// ---- geometry -------------------------------------------------------
	{
		Point a(1.0, 2.0, 3.0);
		Point b(4.0, 5.0, 6.0);

		// Component access must read as p(i): the spelling used throughout
		// the assembly kernels.
		close(a(0), 1.0, "Point component 0");
		close(a(2), 3.0, "Point component 2");

		close(dot(a, b), 32.0, "dot product");

		const Vector3 c = cross(a, b);
		close(c(0), -3.0, "cross x");
		close(c(1),  6.0, "cross y");
		close(c(2), -3.0, "cross z");
		close(dot(c, a), 0.0, "cross is orthogonal to a");
		close(dot(c, b), 0.0, "cross is orthogonal to b");

		close(norm(Point(3.0, 4.0, 0.0)), 5.0, "norm");
		close(distance(a, b), std::sqrt(27.0), "distance");
	}

	// ---- zero-copy view of a MOAB-style flat coordinate array ----------
	{
		const Real coords[9] = { 0.0, 0.0, 0.0,
		                         1.0, 0.0, 0.0,
		                         1.0, 1.0, 0.0 };

		const auto p1 = point_at(coords, 1);
		close(p1(0), 1.0, "point_at(1).x");
		close(p1(1), 0.0, "point_at(1).y");

		// It must be a view, not a copy: the address has to land inside the
		// original array, or every element loop pays for a copy.
		check(p1.data() == coords + 3, "point_at aliases the source array");

		const auto p2 = point_at(coords, 2);
		close(distance(p1, p2), 1.0, "distance between mapped points");
	}

	// ---- bounding box ---------------------------------------------------
	{
		BoundingBox box;
		box.extend(Point(-1.0,  0.0, 2.0));
		box.extend(Point( 3.0, -2.0, 5.0));

		close(box.min()(0), -1.0, "bbox min x");
		close(box.max()(1),  0.0, "bbox max y");
		close(box.center()(2), 3.5, "bbox center z");

		check(box.contains(Point(0.0, -1.0, 3.0)), "bbox contains interior point");
		check(!box.contains(Point(9.0,  0.0, 3.0)), "bbox rejects exterior point");

		// x spans 4, y spans 2, z spans 3 -> longest axis is x.
		check(box.longest_axis() == 0, "bbox longest axis");
	}

	// ---- dense vector ---------------------------------------------------
	{
		RealDenseVector v(4);
		check(v.size() == 4, "vector size");

		// resize must leave the contents zeroed; kernels rely on it.
		for (std::size_t i = 0; i < v.size(); ++i)
			close(v(i), 0.0, "vector zero-initialized");

		v(0) = 1.0; v(1) = 2.0; v(2) = 3.0; v(3) = 4.0;
		close(v.sum(), 10.0, "vector sum");
		close(v.l2_norm(), std::sqrt(30.0), "vector l2 norm");

		v.set_zero();
		close(v.sum(), 0.0, "set_zero");

		v.resize(2);
		check(v.size() == 2, "vector resize");
		close(v(0), 0.0, "resize re-zeros");

		const auto s = v.as_span();
		check(s.size() == 2, "span size matches");
		check(s.data() == v.data(), "span aliases vector storage");
	}

	// ---- dense matrix ---------------------------------------------------
	{
		RealDenseMatrix m(2, 3);
		check(m.rows() == 2 && m.cols() == 3, "matrix shape");

		m(1, 2) = 7.0;
		close(m(1, 2), 7.0, "matrix element access");
		close(m(0, 0), 0.0, "matrix zero-initialized");

		m.set_zero();
		close(m(1, 2), 0.0, "matrix set_zero");
	}

	// ---- LU solve against a known system --------------------------------
	{
		//  [ 2 1 1 ] [x]   [ 5 ]              exact solution: (1, -2, 5)
		//  [ 1 3 2 ] [y] = [ 5 ]
		//  [ 1 0 0 ] [z]   [ 1 ]
		RealDenseMatrix A(3, 3);
		A(0,0)=2.0; A(0,1)=1.0; A(0,2)=1.0;
		A(1,0)=1.0; A(1,1)=3.0; A(1,2)=2.0;
		A(2,0)=1.0; A(2,1)=0.0; A(2,2)=0.0;

		RealDenseVector b(3);
		b(0)=5.0; b(1)=5.0; b(2)=1.0;

		RealDenseVector x;
		A.solve(b, x);

		check(x.size() == 3, "solve sizes the result");
		close(x(0),  1.0, "solve x", 1.e-10);
		close(x(1), -2.0, "solve y", 1.e-10);
		close(x(2),  5.0, "solve z", 1.e-10);

		// Round-trip: A*x must reproduce b.
		RealDenseVector Ax;
		A.multiply(x, Ax);
		for (std::size_t i = 0; i < b.size(); ++i)
			close(Ax(i), b(i), "A*x == b", 1.e-10);
	}

	// ---- consistent mass matrix, the real projection use case -----------
	{
		// 1-D linear element mass matrix on [0,h]: (h/6) * [[2,1],[1,2]].
		// Projecting the constant field 1 must return the constant 1.
		const Real h = 0.5;
		RealDenseMatrix M(2, 2);
		M(0,0) = 2.0*h/6.0; M(0,1) = 1.0*h/6.0;
		M(1,0) = 1.0*h/6.0; M(1,1) = 2.0*h/6.0;

		RealDenseVector rhs(2);          // integral of phi_i * 1 = h/2 each
		rhs(0) = h/2.0; rhs(1) = h/2.0;

		RealDenseVector coeff;
		M.solve(rhs, coeff);

		close(coeff(0), 1.0, "L2 projection of a constant, node 0", 1.e-10);
		close(coeff(1), 1.0, "L2 projection of a constant, node 1", 1.e-10);
	}

	// ---- native() escape hatch ------------------------------------------
	{
		RealDenseMatrix m(2, 2);
		m(0,0)=1.0; m(0,1)=2.0; m(1,0)=3.0; m(1,1)=4.0;

		// Reaching Eigen directly must see the same storage.
		close(m.native().determinant(), -2.0, "native() determinant", 1.e-12);
		check(m.native().data() == m.data(), "native() shares storage");
	}

	if (failures == 0)
		std::cout << "dense/geometry tests passed." << std::endl;
	else
		std::cout << "dense/geometry tests: " << failures << " failure(s)." << std::endl;

	return failures == 0 ? 0 : 1;
}
