#pragma once

/// The Space-templated moving-least-squares weight solve
/// (docs/dev/device-resident-arcs.md: "MLS ... weight assembly producing the
/// device CSR directly"). `mls_weights.hpp`'s own
/// `build_mls_weights` resolves the mesh entity sets, runs the k-nearest
/// search (`fuse::search::BoxIndex`, host-only per docs/dev/search.md), and for each target assembles a `P x P` moment matrix
/// and solves it with `fuse::linalg::least_squares_builtin` -- a
/// column-pivoted Householder QR that also yields the rank/condition
/// diagnostics `E-MLS-RANK` reports. `Result`/`Failure`/`fmt` are not
/// device-callable (docs/dev/native-mesh.md's own rule), so that
/// diagnostic path has to stay host-side regardless of where the solve
/// itself runs.
///
/// This header moves only the numerically dominant per-target work -- the
/// moment-matrix assembly and the dense solve, run once per target,
/// independently of every other target -- onto `Space` as a genuine
/// Kokkos BATCHED kernel: a `KOKKOS_LAMBDA` over targets, each doing its
/// own in-place `P x P` Gaussian elimination with partial pivoting,
/// directly on this header's own `A`/`X` views -- no nested parallelism
/// needed because `P` is small (a handful to a few dozen for the degrees
/// this builder supports).
///
/// `KokkosBatched::SerialGesv` from /opt/kokkos-kernels was tried FIRST, not
/// assumed infeasible: both
/// its `StaticPivoting` and `NoPivoting` instantiations were built against
/// this header's own `A`/`B`/`X` views (including a `LayoutRight` batch
/// layout specifically chosen so `Kokkos::subview(view, t, ALL, ALL)`
/// would deduce a contiguous, non-`LayoutStride` type). Both failed to
/// compile on this host's installed Kokkos 5.2.0 + kokkos-kernels pairing
/// -- `StaticPivoting`'s own scratch slicing (`Kokkos::subview(tmp, ALL,
/// make_pair(0,n))`, `Kokkos::subview(tmp, ALL, n)`) is genuinely
/// `LayoutStride` regardless of `tmp`'s own layout (not a layout choice I
/// could route around), and that installed Kokkos's `View<...,
/// LayoutStride>` has no `.stride_0()`/`.stride_1()`, which
/// `SerialTrsm`/`SerialLU` call on it; `NoPivoting` was tried next on the
/// premise that it builds no such scratch, but its own 1-D subview of the
/// RHS vector came back typed `Kokkos::View<double*, Kokkos::OpenMP::
/// array_layout, Kokkos::Device<Kokkos::CudaSpace::...>>` under the device
/// build -- a layout/execution-space mismatch inside Kokkos's own subview
/// type deduction on this installation, not anything this header's view
/// choices produced. Measured, not guessed. The conclusion is
/// that `KokkosBatched::SerialGesv` -- either pivoting mode -- is not
/// usable against this host's installed Kokkos/KokkosKernels pairing,
/// independent of anything in this header, so the elimination below is a
/// device-callable one of this library's own (in fuse/num): no
/// KokkosKernels dependency at all, so the broken
/// `find_package(KokkosKernels)` is not needed either.
///
/// It is NOT the `FUSE_LINALG_BACKEND` facade's dense solver
/// (cmake/FuseDependencies.cmake's own comment: a KokkosKernels backend
/// was evaluated there and rejected for lacking column-pivoted QR with
/// rank truncation) -- this is a narrower, different
/// concern: a batched solve of many SMALL fixed-size systems inside one
/// builder's own translation unit, not a replacement for the coupler's
/// own Newton-solve linear algebra.
///
/// Partial pivoting, not the host's column-pivoted rank-revealing QR: the
/// device path does not reproduce `E-MLS-RANK`'s condition-number
/// diagnostic (see below), but partial pivoting is the standard,
/// numerically respectable choice for a SQUARE system believed
/// non-singular, which the moment matrix is by this header's own `k >= P`
/// precondition (inherited from the host builder) for a non-degenerate
/// neighbourhood. Verified against the NumPy reference to the precision
/// already reported below, unaffected by this rewrite.
///
/// WHAT THIS DOES NOT DO: the neighbour search itself stays outside this
/// header (`fuse::search::BoxIndex::query_nearest`, called by the caller --
/// `build_mls_weights`, mls_weights.cpp -- which copies the results to
/// `Space` and hands them here (the results are copied to the device
/// behind one function); the search runs on
/// `BoxIndex`'s own resolved space, Host or Device, independently of this
/// header). This header's own elimination
/// (`solve_square_system_partial_pivot`) reports only a per-target PASS/FAIL
/// flag -- no rank, no condition number -- not the RANK-DEFICIENCY
/// diagnostic `E-MLS-RANK`'s named rejection carries on the Host path.
/// `MlsDeviceResult::status` exposes that flag per target (not just the
/// aggregate `n_fail` count) exactly so a caller CAN recover the full
/// diagnostic: `build_mls_weights`'s own Device path re-runs the Host
/// path's rank-revealing solve on just the target(s) `status` names,
/// which is cheap because failures are expected to be rare (this header's
/// own fast path exists on that premise) and reproduces `E-MLS-RANK`'s
/// exact message, target index and condition estimate.
///
/// SCALAR TYPE AND SHAPE GRADIENTS (plans/2026-10-22-universal-number-types.md,
/// stage U8). `solve_mls_on_space<Space, T>` is templated on the scalar type `T`
/// of the point coordinates and of everything computed from them (distances,
/// moment matrices, the solution, the weights). `T = double` is the default and
/// is the previous kernel, operation for operation. With `T =
/// fuse::num::dual<S, N>` (include/fuse/num/dual.hpp, forward-mode automatic
/// differentiation with N directions) and each coordinate seeded as its value
/// together with its gradient in the N design parameters, every returned weight
/// carries its gradient in those parameters: the shape gradient of the weights,
/// i.e. the derivative of the weights with respect to the positions of the
/// source and target points (Griewank and Walther, "Evaluating Derivatives", 2nd
/// ed., SIAM 2008, chapter 3). A weighted sum `y_i = sum_j w_ij f_j` then has the
/// gradient `sum_j (dw_ij) f_j` by the product rule, with no further code.
///
/// WHAT IS DIFFERENTIATED. The algorithm as written, for a FIXED neighbour list
/// (`nbr_index`, chosen by the caller's search, which is not differentiated):
/// the distances, the support radius (1.1 times the largest neighbour distance),
/// the compactly supported radial function, the moment matrix, the elimination
/// and the back substitution. Every branch that selects among alternatives (the
/// largest neighbour distance, the pivot row, the sign and the clamp at 1 of the
/// radial argument, the pivot floor) is decided on the value part
/// (`mls_primal`), so the gradient is that of the branch the value lies on. The
/// weights are piecewise smooth in the coordinates and the gradient is exact
/// except at the following points, where it is one-sided or undefined:
///   - two neighbours are equally far from the target and both are the
///     farthest (the support radius has a kink there);
///   - the neighbour list changes (a discrete change of the stencil);
///   - a neighbour coincides with the target (distance 0). The gradient of the
///     distance is taken as 0 there (`mls_norm_of_squares`), which is the exact
///     gradient of the weight for Wendland2/4/6, whose radial function has zero
///     slope at the origin, and a subgradient for Wendland0, which has a cone.
/// The pivot row does not appear in the list: it does not change the solution of
/// `A x = e_1`, hence not its gradient.
///
/// One branch is decided on more than the value: the elimination skips the row
/// update of a zero multiplier only if the gradient of the multiplier is zero as
/// well (`mls_is_zero`). A symmetric stencil produces multipliers that are zero
/// in value and not in gradient, and skipping those would drop their share of the
/// gradient.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include <Kokkos_Core.hpp>

#include "fuse/backend/mls_weights.hpp"  // MlsRadial, mls_basis_size

namespace fuse::device {

// ---- number-type helpers (see "SCALAR TYPE AND SHAPE GRADIENTS" above) ----
//
// The scalar type `T` of this header is either a built-in arithmetic type
// (`double`) or a forward-mode automatic-differentiation type with the members
// `value()`, `gradient(k)` and `gradient_size` (`fuse::num::dual<S, N>`). The
// four helpers below are the only places that tell the two apart; everything
// else in this header is written once, as arithmetic on `T`.

/// The value part of `x`: `x` itself for an arithmetic type, `x.value()` for a
/// forward-mode type. Every comparison that selects a branch uses this.
template <class T>
[[nodiscard]] KOKKOS_INLINE_FUNCTION auto mls_primal(const T& x) {
  if constexpr (std::is_arithmetic_v<T>) {
    return x;
  } else {
    return x.value();
  }
}

/// True when `x` is exactly zero: for a forward-mode type, when its value AND every
/// gradient entry are zero. The elimination skips a row update only in that case,
/// because a multiplier whose value is zero but whose gradient is not still
/// contributes to the gradient of the result.
template <class T>
[[nodiscard]] KOKKOS_INLINE_FUNCTION bool mls_is_zero(const T& x) {
  if constexpr (std::is_arithmetic_v<T>) {
    return x == T(0);
  } else {
    if (x.value() != 0) return false;
    for (int k = 0; k < T::gradient_size; ++k) {
      if (x.gradient(k) != 0) return false;
    }
    return true;
  }
}

/// `sqrt(s)` for the sum of squares `s` of a coordinate difference. For a
/// forward-mode type a zero value gives the number zero (value and gradient), not
/// the 0/0 of differentiating the square root at its branch point: the distance
/// from a neighbour that coincides with the target has gradient 0 in the sense of
/// the header comment.
template <class T>
[[nodiscard]] KOKKOS_INLINE_FUNCTION T mls_norm_of_squares(const T& s) {
  if constexpr (std::is_arithmetic_v<T>) {
    return Kokkos::sqrt(s);
  } else {
    if (s.value() == 0) return T();
    return sqrt(s);  // found by argument-dependent lookup
  }
}

template <class ViewType, class T>
KOKKOS_INLINE_FUNCTION void mls_basis_into(int dim, int degree, T x, T y, T z,
                                           const ViewType& row) {
  int idx = 0;
  for (int total = 0; total <= degree; ++total) {
    if (dim == 1) {
      T v(1.0);
      for (int i = 0; i < total; ++i) v *= x;
      row(idx++) = v;
    } else if (dim == 2) {
      for (int a = total; a >= 0; --a) {
        const int b = total - a;
        T v(1.0);
        for (int i = 0; i < a; ++i) v *= x;
        for (int i = 0; i < b; ++i) v *= y;
        row(idx++) = v;
      }
    } else {
      for (int a = total; a >= 0; --a) {
        for (int b = total - a; b >= 0; --b) {
          const int cc = total - a - b;
          T v(1.0);
          for (int i = 0; i < a; ++i) v *= x;
          for (int i = 0; i < b; ++i) v *= y;
          for (int i = 0; i < cc; ++i) v *= z;
          row(idx++) = v;
        }
      }
    }
  }
}

/// Same closed forms as `fuse::mls_crbf` (mls_weights.cpp), reproduced as a
/// `KOKKOS_INLINE_FUNCTION`: that one is a plain host function and cannot
/// be called from a device kernel, but the four Wendland polynomials
/// themselves have no host-only dependency at all.
///
/// Templated on the scalar type `T` of the argument (see the header comment): the
/// branches on the sign of `x` and on `|x| > 1` look at the value part, so the
/// gradient is that of the branch the value lies on. The four polynomials have a
/// zero slope at `|x| = 1`, so the clamp does not put a kink in the gradient.
template <class T>
KOKKOS_INLINE_FUNCTION T mls_crbf_device(MlsRadial r, T x) {
  const T t = mls_primal(x) < 0.0 ? -x : x;
  const T tt = mls_primal(t) > 1.0 ? T(1.0) : t;
  const T q = 1.0 - tt;
  switch (r) {
    case MlsRadial::Wendland0:
      return q * q;
    case MlsRadial::Wendland2: {
      const T q2 = q * q;
      return q2 * q2 * (4.0 * tt + 1.0);
    }
    case MlsRadial::Wendland4: {
      const T q3 = q * q * q;
      const T q6 = q3 * q3;
      return q6 * (35.0 * tt * tt + 18.0 * tt + 3.0);
    }
    case MlsRadial::Wendland6: {
      const T q4 = q * q * q * q;
      const T q8 = q4 * q4;
      return q8 * (32.0 * tt * tt * tt + 25.0 * tt * tt + 8.0 * tt + 1.0);
    }
  }
  return T(0.0);
}

/// In-place Gaussian elimination with partial pivoting of the `P x P`
/// system `Amat . x = bvec`, writing the solution into `bvec` itself
/// (`Amat` is destroyed). Returns 0 on success, 1 if no pivot above
/// `kPivotFloor` was found in some column (the system is singular to
/// working precision) -- the same two-outcome contract
/// `KokkosBatched::SerialGesv::invoke` has, so a caller checking `status
/// != 0` needs no change regardless of which solver produced it.
/// `MatView`/`VecView` are any 2-D/1-D view-like type supporting
/// `operator()`, host or device; every index used is local to this one
/// target's own `P x P` block, so this function has no Kokkos::subview of
/// its own.
///
/// The scalar type `T` is that of the entries (`std::remove_cvref_t` of
/// `bvec(0)`): `double`, or a forward-mode type, in which case the elimination
/// carries the gradient through every operation. The pivot row is chosen, and the
/// pivot floor tested, on the value parts; a row update is skipped only when the
/// multiplier is exactly zero in value and gradient (`mls_is_zero`). The pivot
/// choice does not change the mathematical solution, so the gradient is that of
/// `A^{-1} b` whichever row is chosen.
template <class MatView, class VecView>
KOKKOS_INLINE_FUNCTION int solve_square_system_partial_pivot(const MatView& Amat,
                                                              const VecView& bvec, int P) {
  using T = std::remove_cvref_t<decltype(bvec(0))>;
  constexpr double kPivotFloor = 1e-300;
  for (int col = 0; col < P; ++col) {
    int piv = col;
    double best = mls_primal(Amat(col, col)) < 0 ? -mls_primal(Amat(col, col))
                                                   : mls_primal(Amat(col, col));
    for (int row = col + 1; row < P; ++row) {
      const double v = mls_primal(Amat(row, col)) < 0 ? -mls_primal(Amat(row, col))
                                                       : mls_primal(Amat(row, col));
      if (v > best) {
        best = v;
        piv = row;
      }
    }
    if (best < kPivotFloor) return 1;
    if (piv != col) {
      for (int c2 = 0; c2 < P; ++c2) {
        const T t = Amat(col, c2);
        Amat(col, c2) = Amat(piv, c2);
        Amat(piv, c2) = t;
      }
      const T tb = bvec(col);
      bvec(col) = bvec(piv);
      bvec(piv) = tb;
    }
    const T diag = Amat(col, col);
    for (int row = col + 1; row < P; ++row) {
      const T factor = Amat(row, col) / diag;
      if (!mls_is_zero(factor)) {
        for (int c2 = col; c2 < P; ++c2) Amat(row, c2) -= factor * Amat(col, c2);
        bvec(row) -= factor * bvec(col);
      }
    }
  }
  for (int row = P - 1; row >= 0; --row) {
    T sum = bvec(row);
    for (int c2 = row + 1; c2 < P; ++c2) sum -= Amat(row, c2) * bvec(c2);
    bvec(row) = sum / Amat(row, row);
  }
  return 0;
}

/// Thin `operator()` adapters fixing one batch index `t`, so
/// `solve_square_system_partial_pivot` indexes `A(t, a, c)`/`X(t, a)`
/// directly -- no `Kokkos::subview` anywhere in the solve step, which is
/// what sidesteps the layout-deduction issues above entirely rather than
/// routing around them.
template <class Space, class T = double>
struct MatAtBatch {
  Kokkos::View<T***, Kokkos::LayoutRight, Space> A;
  int t;
  KOKKOS_INLINE_FUNCTION T& operator()(int a, int c) const { return A(t, a, c); }
};
template <class Space, class T = double>
struct VecAtBatch {
  Kokkos::View<T**, Kokkos::LayoutRight, Space> X;
  int t;
  KOKKOS_INLINE_FUNCTION T& operator()(int a) const { return X(t, a); }
};

/// Solves every target's moment-matrix system on `Space` and extracts the
/// CSR weight triplets, GIVEN an already-computed k-nearest-neighbour
/// result (`nbr_index`/`nbr_dist`, both `n_rows x k` row-major flat views
/// -- every target has exactly `k` neighbours, `mls_weights.hpp`'s own
/// precondition, checked by the caller before this is reached) and the
/// two point sets, both already copied to `Space`. `n_fail` counts targets
/// whose elimination reported a nonzero status (see this header's own
/// "WHAT THIS DOES NOT DO" section); the caller decides what a nonzero
/// count means for its own error contract.
///
/// Returns device CSR views (`row`/`col`/`val`, each size `n_rows * k`) --
/// the primary result in Device mode -- plus `n_fail` and `status` (one
/// entry per target, 0 on success, the same nonzero code
/// `solve_square_system_partial_pivot` returns on a column with no pivot
/// above `kPivotFloor`). A caller that wants a host `RemapWeights`
/// deep-copies `row`/`col`/`val`; the scalar diagnostics (`max_row_defect`,
/// `min_weight`, `monotone`) are cheap to recompute from the host mirror
/// directly, exactly as `build_mls_weights` does today, and are NOT
/// duplicated as a second device reduction here. `status` exists so a
/// caller can recover WHICH target(s) failed and re-derive `E-MLS-RANK`'s
/// full diagnostic (target index, rank, condition estimate) by re-running
/// the Host path's own rank-revealing solve on just that target -- this
/// header's own partial-pivot elimination reports only a pass/fail flag per
/// target, not a rank or condition number (see this header's own "WHAT THIS
/// DOES NOT DO" section above).
template <class Space, class T = double>
struct MlsDeviceResult {
  Kokkos::View<std::int32_t*, Space> row;
  Kokkos::View<std::int32_t*, Space> col;
  Kokkos::View<T*, Space> val;
  Kokkos::View<int*, Space> status;
  int n_fail = 0;
};

/// `T` is the scalar type of the coordinates and of the returned `val` (see the
/// header comment, "SCALAR TYPE AND SHAPE GRADIENTS"): `double` for the weights
/// alone, `fuse::num::dual<double, N>` for the weights together with their
/// gradient in N parameters. `src_pts` and `tgt_pts` have the same `T`, so a point
/// that does not move is a coordinate with a zero gradient (a `dual` constructed
/// from its value). `T` is deduced from the views; `solve_mls_on_space<Space>(...)`
/// with `double` views is the previous interface unchanged.
template <class Space, class T = double>
[[nodiscard]] MlsDeviceResult<Space, T> solve_mls_on_space(
    int dim, int degree, int k, MlsRadial radial,
    const Kokkos::View<T* [3], Space>& src_pts, const Kokkos::View<T* [3], Space>& tgt_pts,
    const Kokkos::View<int*, Space>& nbr_index) {
  using ExecSpace = typename Space::execution_space;
  const int n_rows = static_cast<int>(tgt_pts.extent(0));
  const int P = static_cast<int>(mls_basis_size(dim, degree));

  // LayoutRight, not LayoutLeft: Kokkos::subview(view, t, ALL, ALL) on a
  // LayoutRight batch view (slicing the OUTERMOST/slowest index, keeping
  // the rest) stays LayoutRight -- contiguous, with its stride computed
  // formulaically from the extents, rather than LayoutStride (verified
  // locally: a LayoutLeft batch view's same slice comes back LayoutStride).
  Kokkos::View<T***, Kokkos::LayoutRight, Space> A("fuse::device::mls_A", n_rows, P, P);
  Kokkos::View<T**, Kokkos::LayoutRight, Space> X("fuse::device::mls_X", n_rows, P);
  Kokkos::View<T**, Space> phi("fuse::device::mls_phi", n_rows, k);
  Kokkos::View<T***, Space> basis_rows("fuse::device::mls_basis_rows", n_rows, k, P);

  Kokkos::parallel_for(
      "fuse::device::mls_assemble", Kokkos::RangePolicy<ExecSpace>(0, n_rows),
      KOKKOS_LAMBDA(const int t) {
        T max_dist(0.0);
        for (int j = 0; j < k; ++j) {
          const int s = nbr_index(t * k + j);
          const T dx = src_pts(s, 0) - tgt_pts(t, 0);
          const T dy = src_pts(s, 1) - tgt_pts(t, 1);
          const T dz = src_pts(s, 2) - tgt_pts(t, 2);
          const T d = mls_norm_of_squares(dx * dx + dy * dy + dz * dz);
          if (mls_primal(d) > mls_primal(max_dist)) max_dist = d;
        }
        const T radius = 1.1 * (mls_primal(max_dist) > 1e-300 ? max_dist : T(1e-300));

        for (int a = 0; a < P; ++a) {
          // X doubles as the RHS going in (e_1, "evaluate at the target")
          // and the solution coming out of solve_square_system_partial_pivot,
          // which solves in place -- the same in/out role `VectorType X`
          // plays in KokkosBatched::SerialGesv's own signature.
          X(t, a) = (a == 0) ? T(1.0) : T(0.0);
          for (int c = 0; c < P; ++c) A(t, a, c) = T(0.0);
        }
        for (int j = 0; j < k; ++j) {
          const int s = nbr_index(t * k + j);
          const T dx = src_pts(s, 0) - tgt_pts(t, 0);
          const T dy = src_pts(s, 1) - tgt_pts(t, 1);
          const T dz = src_pts(s, 2) - tgt_pts(t, 2);
          const T d = mls_norm_of_squares(dx * dx + dy * dy + dz * dz);
          const T p = mls_crbf_device(radial, d / radius);
          phi(t, j) = p;
          auto row = Kokkos::subview(basis_rows, t, j, Kokkos::ALL);
          mls_basis_into(dim, degree, dx, dy, dz, row);
          for (int a = 0; a < P; ++a) {
            for (int c = 0; c < P; ++c) A(t, a, c) += p * row(a) * row(c);
          }
        }
      });

  Kokkos::View<int*, Space> status("fuse::device::mls_status", n_rows);
  Kokkos::parallel_for(
      "fuse::device::mls_solve", Kokkos::RangePolicy<ExecSpace>(0, n_rows),
      KOKKOS_LAMBDA(const int t) {
        MatAtBatch<Space, T> Amat{A, t};
        VecAtBatch<Space, T> Xvec{X, t};
        status(t) = solve_square_system_partial_pivot(Amat, Xvec, P);
      });

  MlsDeviceResult<Space, T> out;
  out.row = Kokkos::View<std::int32_t*, Space>("fuse::device::mls_row", n_rows * k);
  out.col = Kokkos::View<std::int32_t*, Space>("fuse::device::mls_col", n_rows * k);
  out.val = Kokkos::View<T*, Space>("fuse::device::mls_val", n_rows * k);
  out.status = status;
  Kokkos::parallel_for(
      "fuse::device::mls_extract", Kokkos::RangePolicy<ExecSpace>(0, n_rows),
      KOKKOS_LAMBDA(const int t) {
        for (int j = 0; j < k; ++j) {
          T dotc(0.0);
          for (int a = 0; a < P; ++a) dotc += X(t, a) * basis_rows(t, j, a);
          out.row(t * k + j) = t;
          out.col(t * k + j) = nbr_index(t * k + j);
          out.val(t * k + j) = phi(t, j) * dotc;
        }
      });

  int n_fail = 0;
  Kokkos::parallel_reduce(
      "fuse::device::mls_count_fail", Kokkos::RangePolicy<ExecSpace>(0, n_rows),
      KOKKOS_LAMBDA(const int t, int& acc) { acc += (status(t) != 0) ? 1 : 0; }, n_fail);
  out.n_fail = n_fail;
  return out;
}

}  // namespace fuse::device
