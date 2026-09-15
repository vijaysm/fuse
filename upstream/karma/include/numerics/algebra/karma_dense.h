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


#ifndef __karma_dense_h__
#define __karma_dense_h__

#include "karma_types.h"

#include <Eigen/Dense>

#include <cassert>
#include <concepts>
#include <cstddef>
#include <span>
#include <type_traits>

namespace KARMA
{

	/**
	 * Element-local dense matrices and vectors.
	 *
	 * These are the small, transient containers an assembly kernel fills for
	 * one element and then scatters into the global system.  They replace
	 * libMesh's \p DenseMatrix / \p DenseVector and are thin adaptors over
	 * Eigen, chosen so that the numerical kernels (LU, norms, products) come
	 * from a maintained library rather than being hand-written here.
	 *
	 * The naming is KARMA's own rather than libMesh's: \p rows()/\p cols()
	 * instead of \p m()/\p n(), \p set_zero() instead of \p zero().  Ported
	 * kernels are updated to match; the compiler catches every site.
	 *
	 * \p native() exposes the underlying Eigen object, so a kernel that wants
	 * real linear algebra (decompositions, block operations, expression
	 * templates) is not boxed in by this wrapper.
	 */

	/**
	 * Scalar types these containers accept.  Restricting the template keeps
	 * an accidental instantiation from producing pages of Eigen diagnostics.
	 */
	template <typename T>
	concept DenseScalar = std::is_floating_point_v<T>
	                   || std::same_as<T, Complex>;


	template <DenseScalar T>
	class DenseVector
	{
	public:

		using Storage = Eigen::Matrix<T, Eigen::Dynamic, 1>;
		using value_type = T;

		DenseVector() = default;

		explicit DenseVector (std::size_t n)
			: _data(Storage::Zero(static_cast<Eigen::Index>(n)))
		{}

		/**
		 * Resize and zero.  Assembly kernels call this once per element, so
		 * it must leave the contents defined.
		 */
		void resize (std::size_t n)
		{
			_data.setZero(static_cast<Eigen::Index>(n));
		}

		void set_zero () noexcept { _data.setZero(); }

		[[nodiscard]] std::size_t size () const noexcept
		{
			return static_cast<std::size_t>(_data.size());
		}

		[[nodiscard]] bool empty () const noexcept { return _data.size() == 0; }

		[[nodiscard]] T& operator() (std::size_t i) noexcept
		{
			return _data[static_cast<Eigen::Index>(i)];
		}

		[[nodiscard]] const T& operator() (std::size_t i) const noexcept
		{
			return _data[static_cast<Eigen::Index>(i)];
		}

		[[nodiscard]] T* data () noexcept { return _data.data(); }
		[[nodiscard]] const T* data () const noexcept { return _data.data(); }

		/** A contiguous view, for handing the buffer to PETSc or MOAB. */
		[[nodiscard]] std::span<T> as_span () noexcept
		{
			return { _data.data(), size() };
		}

		[[nodiscard]] std::span<const T> as_span () const noexcept
		{
			return { _data.data(), size() };
		}

		[[nodiscard]] Storage&       native ()       noexcept { return _data; }
		[[nodiscard]] const Storage& native () const noexcept { return _data; }

		[[nodiscard]] Real l2_norm () const { return _data.norm(); }
		[[nodiscard]] T    sum     () const { return _data.sum();  }

		DenseVector& operator+= (const DenseVector& v) { _data += v._data; return *this; }
		DenseVector& operator-= (const DenseVector& v) { _data -= v._data; return *this; }
		DenseVector& operator*= (T s)                  { _data *= s;       return *this; }

	private:

		Storage _data;
	};


	template <DenseScalar T>
	class DenseMatrix
	{
	public:

		using Storage = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
		using value_type = T;

		DenseMatrix() = default;

		DenseMatrix (std::size_t rows, std::size_t cols)
			: _data(Storage::Zero(static_cast<Eigen::Index>(rows),
			                      static_cast<Eigen::Index>(cols)))
		{}

		void resize (std::size_t rows, std::size_t cols)
		{
			_data.setZero(static_cast<Eigen::Index>(rows),
			              static_cast<Eigen::Index>(cols));
		}

		void set_zero () noexcept { _data.setZero(); }

		[[nodiscard]] std::size_t rows () const noexcept
		{
			return static_cast<std::size_t>(_data.rows());
		}

		[[nodiscard]] std::size_t cols () const noexcept
		{
			return static_cast<std::size_t>(_data.cols());
		}

		[[nodiscard]] T& operator() (std::size_t i, std::size_t j) noexcept
		{
			return _data(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));
		}

		[[nodiscard]] const T& operator() (std::size_t i, std::size_t j) const noexcept
		{
			return _data(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));
		}

		/**
         * Column-major contiguous storage, which is Eigen's default. PETSc
         * dense-block insertion requires an explicit row-major conversion.
		 */
		[[nodiscard]] T* data () noexcept { return _data.data(); }
		[[nodiscard]] const T* data () const noexcept { return _data.data(); }

		[[nodiscard]] Storage&       native ()       noexcept { return _data; }
		[[nodiscard]] const Storage& native () const noexcept { return _data; }

		/**
		 * Solve \p A x = b for this matrix, by partial-pivot LU.
		 *
		 * Used by the element-local L2 projection, where the operator is the
		 * small consistent mass matrix.  Partial pivoting is sufficient there
		 * (mass matrices are symmetric positive definite); a kernel needing
		 * a different decomposition should reach through \p native().
		 */
		void solve (const DenseVector<T>& rhs, DenseVector<T>& x) const
		{
			assert(_data.rows() == _data.cols() && "solve requires a square matrix");
			assert(static_cast<std::size_t>(_data.rows()) == rhs.size() && "rhs size must match matrix rows");

			x.resize(cols());
			x.native() = _data.partialPivLu().solve(rhs.native());
		}

		/** \p y = A x */
		void multiply (const DenseVector<T>& x, DenseVector<T>& y) const
		{
			assert(static_cast<std::size_t>(_data.cols()) == x.size() && "x size must match matrix cols");

			y.resize(rows());
			y.native().noalias() = _data * x.native();
		}

		[[nodiscard]] DenseMatrix transpose () const
		{
			DenseMatrix t;
			t._data = _data.transpose();
			return t;
		}

		[[nodiscard]] Real l2_norm () const { return _data.norm(); }

		DenseMatrix& operator+= (const DenseMatrix& m) { _data += m._data; return *this; }
		DenseMatrix& operator-= (const DenseMatrix& m) { _data -= m._data; return *this; }
		DenseMatrix& operator*= (T s)                  { _data *= s;       return *this; }

	private:

		Storage _data;
	};


	/** The instantiations the assembly kernels actually use. */
	using RealDenseVector = DenseVector<Real>;
	using RealDenseMatrix = DenseMatrix<Real>;

}

#endif // __karma_dense_h__
