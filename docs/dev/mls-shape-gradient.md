# Shape gradients of the MLS weights

The moving-least-squares weights of a point cloud (`fuse/backend/mls_weights.hpp`, user guide:
[](../user/point-cloud-remap.md)) depend smoothly on the coordinates of the source and target points.
The weight kernel, `fuse::device::solve_mls_on_space`
(`fuse/backend/mls_weights_device.hpp`), is a template on the scalar type `T` of those coordinates, so
that with `T = fuse::num::dual<S, N>` (forward-mode automatic differentiation, `fuse/num/dual.hpp`)
every weight comes back with its gradient in `N` parameters: the **shape gradient** of the weights,
the derivative of the weights with respect to where the points are. `T = double`, the default, is the
kernel as it was, operation for operation.

## 1. Using it

```cpp
using D = fuse::num::dual<double, 2>;   // a value and its gradient in two parameters
using Space = Kokkos::HostSpace;        // or the device memory space

Kokkos::View<D * [3], Space> src("src", n_src);   // coordinates (x, y, z) of the sources
Kokkos::View<D * [3], Space> tgt("tgt", 1);       // the target
Kokkos::View<int*, Space> nbr("nbr", k);          // its k neighbours, from your own search
// ... src(i, a) = D(x_ia) for sources that do not move; unused coordinates are D(0.0)
tgt(0, 0) = D::variable(xt, 0);                   // parameter 0: the x of the target
tgt(0, 1) = D::variable(yt, 1);                   // parameter 1: the y of the target
tgt(0, 2) = D(0.0);

auto w = fuse::device::solve_mls_on_space<Space>(/*dim=*/2, /*degree=*/1, k,
                                                 fuse::MlsRadial::Wendland2, src, tgt, nbr);
// w.val(j).value() is the weight of source nbr(j), w.val(j).gradient(0) its derivative with
// respect to xt, and w.val(j).gradient(1) its derivative with respect to yt.
```

The source and target views have the same `T`; a point that does not move is a coordinate with a
zero gradient. The result `MlsDeviceResult<Space, T>` has the same `row`, `col`, `status` and
`n_fail` as before, with `val` of type `T`. A weighted sum `y = sum_j w_j f_j` of a field `f` then has
the gradient `sum_j (dw_j) f_j + sum_j w_j (df_j)` by the product rule: accumulate it in `T`, nothing
else is needed. `T` is deduced from the views, so every existing call with `double` views compiles
and behaves as before (`build_mls_weights` is one of them and is unchanged).

## 2. What is differentiated

The algorithm as it is written, for a fixed neighbour list: the distances, the support radius
(1.1 times the largest neighbour distance), the compactly supported radial function (Wendland), the
`P x P` moment matrix, the Gaussian elimination with partial pivoting and the back substitution, and
the final multiplication by the radial function. Every operation is arithmetic on `T`. Four helpers
in the header are the only places that tell a built-in type from a forward-mode type: `mls_primal`
(the value part), `mls_is_zero` (zero in value and in every gradient entry), `mls_norm_of_squares`
(a square root with a rule at zero) and, through them, the comparisons below.

A branch that selects among alternatives is decided on the value part, as in any differentiated
program (Griewank and Walther, *Evaluating Derivatives*, 2nd ed., SIAM 2008, chapter 3): the
largest neighbour distance, the pivot row, the sign and the clamp at 1 of the radial argument, the
pivot floor. The gradient is then that of the branch the value lies on. The weights are piecewise
smooth in the coordinates, and the gradient is exact except at these points:

| Point | What happens |
| --- | --- |
| two neighbours are equally far from the target and both are the farthest | the support radius has a kink, so the gradient is one-sided (the first of the two in the list is used) |
| the neighbour list changes | a discrete change of the stencil; the list is an input and is not differentiated |
| a neighbour coincides with the target | the distance is 0 and the gradient of a square root there is 0/0; `mls_norm_of_squares` gives the distance the gradient 0, which is the exact gradient for Wendland2, 4 and 6 (their radial function has zero slope at the origin) and a subgradient for Wendland0 (a cone) |

Two choices are not branch points. The **pivot row** does not change the solution of `A x = e_1`, so
it does not change its gradient. The **clamp at 1** of the radial argument is never active: the
farthest neighbour is at `1 / 1.1` of the radius. One branch is decided on more than the value: the
elimination skips the row update of a multiplier only if the multiplier is zero in value and in
gradient (`mls_is_zero`). A stencil that is symmetric about the target has moment-matrix entries
that are exactly zero (the two terms of `sum_j phi_j dx_j` are equal and opposite), whose gradient is
not zero; skipping their row updates on the value alone would drop their share of the gradient.

A target whose elimination fails (`status(t) != 0`, `n_fail > 0`) has no valid weights and no valid
gradient, exactly as for `double`; the caller checks `n_fail`.

## 3. Cost

A `dual<S, N>` is `N + 1` scalars. The scratch views per target hold `P*P + P + k + k*P` numbers of
type `T` (the moment matrix, the solution, the radial values and the basis rows, for `P` basis
functions and `k` neighbours), so a gradient in `N` parameters multiplies the memory of the solve and
the arithmetic of every operation by about `N + 1`. For a two-dimensional degree 2 fit (`P = 6`) with
`k = 10` and `N = 20` that is 112 numbers of 21 doubles, 18.8 kB per target. Forward mode is the right
tool for a few parameters (a design variable, a direction); the gradient with respect to every
coordinate of a large cloud would need reverse mode, which this does not provide (section 6).

## 4. How it is checked

`test/backend/mls_weights_gradient_test.cpp`, a typed test: every case runs on `Kokkos::HostSpace` and, in
a build with a device backend, on the device memory space. The cases and what they show:

1. **The value part equals the double run bit for bit**, in the `FUSE_FP_REPRODUCIBLE=ON`
   configuration the tests are built in. So the gradient is that of the weights the library already
   produces.
2. **The weights sum to 1, and the sum has gradient 0** in every parameter.
3. **The weights reproduce every polynomial of the fit degree, and so does the gradient.** With
   `F = sum_j w_j q(x_j)` and `q` evaluated in `T` at the sources (so the motion of each source
   enters through `q(x_j)`), `F = q(x_t)` in value and in every gradient entry. These two identities
   are derived by hand and use no finite difference. They hold for ANY positive radial values and any
   consistent gradient of them, so they cannot see an error in the gradient of the radial function,
   of the distances or of the support radius; the next case does.
4. **Each gradient entry equals the fourth-order central difference** `(8 (f(+h) - f(-h)) - (f(+2h)
   - f(-2h))) / (12 h)`, `h = 1e-3`, of the double run, for every weight and every parameter.
5. **A symmetric stencil**, sources at `(+-1, 0)`, `(0, +-1.5)`, `(0, 2.2)` around a target at the
   origin, in which the multipliers of one column are zero in value and not in gradient (section 2).
   The farthest source is unique, so the central difference is a valid reference there.
6. **A neighbour coinciding with the target**, finite gradients, and agreement with the central
   difference for Wendland2, 4 and 6.
7. **The gate: the Taylor remainder of the weights, as a function of all the coordinates
   (every source and every target), has order 2.** `fuse::linalg::taylor_remainder_order`
   ([](derivative-gate.md)) with `h0 = 1e-2` and 9 decades, along a fixed direction of max-norm 1.
   The clouds are jittered lattices: 6 x 6 sources and 3 targets in two dimensions (degree 2,
   `k = 10`, all four radial functions), 12 sources and 3 targets in one dimension (degree 2,
   `k = 6`, Wendland4), and 4 x 4 x 4 sources and 2 targets in three dimensions (degree 1,
   `k = 10`, Wendland2). The
   order is read in `(1.8, 2.3)`.
   The control, the same sweep with the motion of the targets left out of the gradient, must read in
   `(0.7, 1.3)`: a gradient that is wrong by the dependence on the target is seen as order 1, not 2.
   Before each sweep the test checks that the gap between the two largest neighbour distances
   exceeds `4 sqrt(dim) h0`, which is the largest closing of that gap along the sweep, so no
   neighbour swaps the farthest place and the weights stay on one smooth branch.
8. **Single precision.** The same kernel with `dual<float, 1>`, against the double run on the same
   coordinates (rounded to single first).

The point clouds come from a 64-bit linear congruential generator in the test, not from the standard
distributions, whose output differs between libraries, so the clouds are the same on every platform.

### Tolerances

None is a value picked by hand; each has the form of the derivative gate's own bounds
([](derivative-gate.md)) and a constant fixed by a measurement, with the headroom stated.

| Check | Bound | Largest measured | Headroom |
| --- | --- | --- | --- |
| identity defects (cases 2, 3, 5, 6) | `1e4 * eps * (sum of the absolute values of the terms added)` | 226 `eps` on the host, 397 `eps` on the device (Wendland6, two dimensions, degree 2, `k = 10`) | 25x |
| gradient against the central difference (cases 4, 5, 6) | `1e-6 * (1 + abs(gradient))` | 1.4e-8 on both | 70x |
| single against double (case 8) | `1e-5 * (1 + abs(double))` | 1.7e-7 on the host, 2.5e-7 on the device | 40x |
| Taylor order, correct gradient | `(1.8, 2.3)` | 1.87 to 1.99 | as in the derivative gate |
| Taylor order, target motion left out | `(0.7, 1.3)` | 0.9999 to 1.0002 | as in the derivative gate |

Measured Taylor orders: 1.903, 1.923, 1.914 and 1.922 for Wendland0, 2, 4 and 6 in two
dimensions, 1.876 in one dimension, 1.980 in three dimensions on the host (OpenMP and Serial, Apple
clang 21, arm64); 1.897, 1.925, 1.920, 1.917, 1.873 and 1.989 on the device. Each sweep has 7
points above a round-off floor of 5e-16 to 1e-15, which is where the remainder stops dropping. The
identity ratios in `eps` range from 0.7 to 397, the largest for the radial function of the highest
degree.

Device results are those of the GPU of the development host, an NVIDIA GeForce GTX 1050 Ti (sm_61),
built with CUDA 12.5 and gcc 13 as the host compiler: the same 18 test instances (9 test functions, each on
`Kokkos::HostSpace` and on `Kokkos::CudaSpace`) pass, and the host-space results on that machine
equal the ones above to every digit printed. The ctest entries are `backend.mls_weights_gradient`
(both spaces in one binary) next to the existing `backend.mls_weights`, `backend.mls_weights.device`
and `backend.mls_weights_device`, which pass unchanged.

## 5. Mutation coverage

`build-evidence/mutations/backend-mls-gradient.toml`, area `backend_mls_gradient`: eleven rows,
each changing only what a dual-number run computes, never the value part, so the double tests of
`mls_weights_device_test.cpp` cannot see them. Each is killed by the new test. The assertions that
kill them differ, which is why the test has all of them (measured by applying each row and reading
which assertions fail):

- **only the symmetric stencil** kills the two rows that skip a zero multiplier on its value alone
  (one at the call site, one in `mls_is_zero`);
- **only the coincident neighbour case** kills the removal of the guard in `mls_norm_of_squares`
  (the gradients become NaN);
- **only the central difference and the Taylor remainder** kill the support radius built from the
  value of the largest distance, the distance returned without its gradient, and the factor
  `(4 t + 1)` of Wendland2 evaluated on the value of `t`: the weights stay exact for any positive
  radial values and any consistent gradient of them, so the identities of cases 2 and 3 hold;
- **the identities, the central difference and the Taylor remainder** kill the moment matrix
  assembled from the value of the radial function, the final weight built from the value of the
  radial function, and the back substitution dividing by the value of the diagonal entry;
- **the polynomial identity, the central difference and the Taylor remainder** kill the
  one-dimensional basis built from the value of the coordinate difference;
- **the polynomial identity and the central difference, but not the Taylor remainder**, kill the
  right-hand-side entry that moves in a pivot swap carrying only its value: the one direction of
  the Taylor sweep does not see it.

## 6. Not done

- **Reverse mode.** The adjoint schedule ([](adjoint-schedule.md)) treats every remap weight as a
  constant of the problem, so a reverse sweep returns gradients with respect to fields and never with
  respect to the positions of points. Forward mode costs `N + 1` times the primal solve for `N`
  parameters; a gradient in every coordinate of a large cloud needs the reverse mode of the solve,
  which is not implemented.
- **The coupler.** `build_mls_weights`, `RemapWeights` and the `Runner` stay `double`: no field or
  channel carries a shape gradient, and the mesh coordinates are not differentiable inputs of a
  coupling. The kernel is reached directly, as in section 1. The weights of the supermesh and
  TempestRemap builders have no gradient at all.
- **The search.** The neighbour list is an input, found by `fuse::search::BoxIndex`, and is not
  differentiated; a gradient across a change of the list does not exist (section 2).
- **Other scalar types.** `T` is `double` or a type with `value()`, `gradient(k)` and
  `gradient_size`; `dual<double, N>` and `dual<float, N>` are what the tests instantiate. A type of
  another kind (the double-double and quad-double of `fuse/num`, posits) has no `value()` for the
  comparisons and is not supported by this kernel.
- **The rank diagnostic.** As for `double`, the device kernel reports only a pass or fail per
  target; `E-MLS-RANK`'s condition estimate comes from the host builder.
