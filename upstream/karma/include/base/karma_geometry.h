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


#ifndef __karma_geometry_h__
#define __karma_geometry_h__

#include "karma_types.h"

#include <Eigen/Dense>

#include <concepts>
#include <cstddef>
#include <limits>
#include <span>

namespace KARMA
{

	/**
	 * Spatial vector types.
	 *
	 * These are Eigen fixed-size vectors rather than a hand-rolled point
	 * class.  Component access is \p p(0) / \p p(1) / \p p(2), which is the
	 * spelling the assembly kernels already use.
	 *
	 * \p Point and \p Vector3 are the same type by construction: a point and
	 * a displacement have identical representation and identical arithmetic,
	 * and distinguishing them in the type system would buy nothing here while
	 * forcing conversions at every boundary.  \p Gradient names the role a
	 * vector plays when it holds the spatial derivative of a scalar field.
	 */
	using Vector3  = Eigen::Matrix<Real, 3, 1>;
	using Point    = Vector3;
	using Gradient = Vector3;

	using Matrix3 = Eigen::Matrix<Real, 3, 3>;

	/**
	 * A read-only view of contiguous coordinates as \p Point objects.
	 *
	 * MOAB hands back vertex coordinates as a flat \p double array of
	 * \p 3*nverts entries.  Eigen::Map lets that memory be read as points
	 * without copying it.
	 */
	using PointMap = Eigen::Map<const Vector3>;

	/**
	 * @returns the \p i-th point of a flat, MOAB-style coordinate array.
	 *
	 * No bounds checking and no copy: this is called inside element loops.
	 */
	[[nodiscard]] inline PointMap point_at(const Real* coords, std::size_t i) noexcept
	{
		return PointMap(coords + 3 * i);
	}

	/**
	 * Concept satisfied by anything usable as a 3-D spatial vector.
	 *
	 * Constrains the free functions below so that a mistaken argument is
	 * reported at the call site rather than inside an Eigen expression
	 * template, where the diagnostic would be far less legible.
	 */
	template <typename T>
	concept SpatialVector = requires (const T& v) {
		{ v(0) }    -> std::convertible_to<Real>;
		{ v.size() } -> std::convertible_to<Eigen::Index>;
	};

	/**
	 * @returns the dot product of two spatial vectors.
	 *
	 * libMesh spelled this \p a * b via \p RealGradient::operator*.  Eigen
	 * reserves \p operator* for matrix products, so the dot product is named
	 * explicitly.  That is a deliberate improvement: \p a * b for two column
	 * vectors is a compile error in Eigen, so every ported site is flagged by
	 * the compiler rather than silently changing meaning.
	 */
	template <SpatialVector A, SpatialVector B>
	[[nodiscard]] inline Real dot(const A& a, const B& b) noexcept
	{
		return a.dot(b);
	}

	template <SpatialVector A, SpatialVector B>
	[[nodiscard]] inline Vector3 cross(const A& a, const B& b) noexcept
	{
		return a.cross(b);
	}

	template <SpatialVector A>
	[[nodiscard]] inline Real norm(const A& a) noexcept
	{
		return a.norm();
	}

	template <SpatialVector A, SpatialVector B>
	[[nodiscard]] inline Real distance(const A& a, const B& b) noexcept
	{
		return (a - b).norm();
	}

	/**
	 * An axis-aligned bounding box.
	 *
	 * Used for geometric partitioning and for coarse rejection ahead of point
	 * location.  MOAB has \p moab::BoundBox, but this stays in KARMA's own
	 * vocabulary so that geometry code does not have to include MOAB headers.
	 */
	class BoundingBox
	{
	public:

		BoundingBox() noexcept
			: _min(Vector3::Constant( std::numeric_limits<Real>::max())),
			  _max(Vector3::Constant(-std::numeric_limits<Real>::max()))
		{}

		BoundingBox(const Point& lo, const Point& hi) noexcept
			: _min(lo), _max(hi)
		{}

		void extend (const Point& p) noexcept
		{
			_min = _min.cwiseMin(p);
			_max = _max.cwiseMax(p);
		}

		[[nodiscard]] const Point& min () const noexcept { return _min; }
		[[nodiscard]] const Point& max () const noexcept { return _max; }

		[[nodiscard]] Vector3 extent () const noexcept { return _max - _min; }
		[[nodiscard]] Point   center () const noexcept { return 0.5 * (_min + _max); }

		[[nodiscard]] Real diagonal () const noexcept { return extent().norm(); }

		[[nodiscard]] bool contains (const Point& p, Real tol = 0.0) const noexcept
		{
			return (p.array() >= _min.array() - tol).all()
			    && (p.array() <= _max.array() + tol).all();
		}

		/**
		 * @returns the index of the longest axis, which is the coordinate a
		 * recursive-bisection partitioner should split on.
		 */
		[[nodiscard]] int longest_axis () const noexcept
		{
			int axis = 0;
			extent().maxCoeff(&axis);
			return axis;
		}

	private:

		Point _min;
		Point _max;
	};

}

#endif // __karma_geometry_h__
