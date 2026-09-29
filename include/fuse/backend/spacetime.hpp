#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "fuse/base/ids.hpp"
#include "fuse/base/rational.hpp"
#include "fuse/base/result.hpp"
#include "fuse/schedule/slab.hpp"

/// Spec §7.3: the `T` of `M = T (x) W`. This header builds the temporal half
/// of a tier-1 (tensor-product) space-time remap: the matrix that carries a
/// field from the source component's time partition onto the target's over one
/// window. `W`, the spatial supermesh operator, is Phase 5's job.
///
/// ---------------------------------------------------------------------------
/// WHICH OPERATOR IS THIS? (read before using the weights)
/// ---------------------------------------------------------------------------
/// **Every weight set built here is CONSERVATIVE, not interpolatory.** They are
/// different matrices and conflating them is a classic coupler bug, so the
/// convention is stated once, here, and never varied:
///
///   * A *coefficient* `x_j` is the MEAN of the field over source step `j`,
///     i.e. `(1/dt_src) * integral over step j`. The output `(T x)_i` is the
///     mean over target step `i`. Both sides are step-averaged quantities;
///     neither is a point sample.
///   * The construction is the classic conservative remap: build, inside each
///     source step, the unique polynomial of degree `p` whose step means over a
///     small stencil of neighbouring steps match the given coefficients, then
///     take the exact mean of that piecewise reconstruction over each target
///     step. The supermesh (`SlabDecomposition`) makes those integrals exact --
///     no quadrature error at the step boundaries, the same reason spatial
///     supermesh projection is exact (Farrell et al., CMAME 2009).
///
/// Two properties follow *structurally* from that construction and are each
/// pinned by a test, because a projection that quietly loses either produces a
/// coupled answer that looks fine and converges at the wrong rate:
///
///   1. **Constant reproduction / partition of unity.**  `sum_j T_ij == 1` for
///      every target step `i`. A field constant in time projects to that
///      constant, to round-off. Note the convention: the row sums are ONE, not
///      the step length, because the coefficients are step *means*, not step
///      integrals.
///   2. **Conservation of the time integral.**  `sum_i T_ij * dt_tgt == dt_src`
///      for every source step `j`, which is exactly the statement that
///      `sum_i (T x)_i dt_tgt == sum_j x_j dt_src` for every `x`: the projected
///      field integrates over the target grid to the source integral over the
///      same span, to round-off.
///
/// Property 2 is a COLUMN sum, and it is the one the space-time conservation
/// check of spec §7.4 rides on. (Task 6.2's brief stated the conservation
/// property as `sum_j T_ij * dt_src == dt_tgt`, a *row* sum. That identity is
/// false for the weights the same brief prescribes: under 4x subcycling every
/// `T_0j` is 1/4 and `dt_src` is 1/4, so the left side is 1/4 while `dt_tgt` is
/// 1. The two true identities are the ones above.)
///
/// The interpolatory (pointwise) temporal operator -- sample `Waveform` at the
/// target step centres -- is a genuinely different matrix and is deliberately
/// NOT built here. It cannot be, in general, without extrapolating: the first
/// target centre can lie before the first source centre, and `Waveform` rejects
/// to evaluate outside its support (`E-WF-OUTSIDE`) rather than silently
/// extrapolate. Conservative projection has no such gap, because the supports
/// are the steps themselves and the two partitions cover the same window.
///
/// ---------------------------------------------------------------------------
/// BASES AND ORDER
/// ---------------------------------------------------------------------------
/// The right-hand column is what `spacetime_order_test.cpp` actually measured
/// on the step means of exp(4t), refining a 3n-against-5n non-nesting pair of
/// partitions from n = 32 to n = 64 -- not what the theory says. It approaches
/// `p + 1` from below on every ladder step; the residual gap is the next term
/// of the exponential's error expansion and shrinks as the ladder refines.
///
///   basis        degree p   reconstruction        claimed   measured
///   ----------   --------   -------------------   -------   --------
///   Constant     0          piecewise constant      1        0.98
///   Linear       1          piecewise linear        2        1.91
///   DG(0)        0          piecewise constant      1        0.98
///   DG(1)        1          piecewise linear        2        1.91
///   DG(2)        2          piecewise quadratic     3        2.96
///   DG(3)        3          piecewise cubic         4        3.95
///   DG(4)        4          piecewise quartic       5        4.93
///   Spline       --         REJECTED (E-ST-BASIS)
///   None         --         REJECTED (E-ST-BASIS)
///
/// The algebraic statement underneath those orders -- that a degree-`p`
/// reconstruction reproduces the step means of a degree-`p` polynomial
/// EXACTLY, not merely to O(h^(p+1)) -- is measured separately and comes out
/// at 0 to 3.6e-15 for p = 0..4.
///
/// `Linear` and `DG(1)` produce the SAME matrix, and that is deliberate rather
/// than an oversight. In `Waveform` they differ (piecewise interpolation of
/// samples versus a global least-squares fit over the window), but a global
/// least-squares fit is the wrong object for a projection matrix: it does not
/// converge under grid refinement at fixed degree (see
/// `WaveformOrder.DgDoesNotConvergeUnderSampleRefinement`), so using it here
/// would cap the projection at O(1). The conservative per-step reconstruction
/// is indexed by degree alone, so `Linear` is simply the `p == 1` member.
///
/// `Spline` is rejected rather than silently downgraded: a conservative
/// spline projection is a global (coupled) reconstruction, not the local one
/// implemented here, and returning a piecewise-cubic-per-step matrix under the
/// name `Spline` would be exactly the silent order/behaviour swap that
/// `Waveform`'s constructor rejects for `Spline(5)`.
///
/// ---------------------------------------------------------------------------
/// STENCILS AT THE ENDS OF THE WINDOW
/// ---------------------------------------------------------------------------
/// The reconstruction in source step `j` uses `p + 1` consecutive steps
/// starting at `j - (p + 1) / 2`, clamped into `[0, n_src - p - 1]`. The
/// stencil is symmetric for even `p` and leans one step towards the past for
/// odd `p`. It is NOT periodic: the window is finite and reaching across its
/// end would be extrapolation from data the other component has not produced.
/// A clamped one-sided stencil at the ends still interpolates degree-`p`
/// polynomials exactly, so the order is unchanged; only the error constant
/// grows.
///
/// If the source partition has fewer than `p + 1` steps the degree is CLAMPED
/// to `n_src - 1` (there is no degree-`p` reconstruction from fewer than
/// `p + 1` averages). `TemporalProjection::effective_degree` reports what was
/// actually used; `degree` reports what was asked for. This mirrors
/// `num::fit_polynomial`, which clamps identically.
///
/// ---------------------------------------------------------------------------
/// ERROR CODES (all returned, never thrown -- `build_temporal_projection`
/// returns a `Result` and is therefore no-throw by contract)
/// ---------------------------------------------------------------------------
///   E-ST-BASIS      `None` or `Spline`
///   E-ST-DEGREE     degree does not match the basis, or exceeds
///                   `kMaxTemporalDegree`
///   E-ST-INTERVAL   a non-positive interval, or an interval that does not
///                   divide the decomposition's period an exact whole number
///                   of times
///   E-ST-SLABS      the decomposition does not tile `[0, period]` with the
///                   given intervals -- an empty list of common-refinement
///                   intervals, a gap, an overlap, a step index out of range,
///                   or a common-refinement interval straddling a step
///                   boundary. This is the check that catches a
///                   `SlabDecomposition` built from intervals other than the
///                   ones passed here.
///   E-ST-SINGULAR   the reconstruction system was singular (defensive; the
///                   stencil geometry makes it non-singular for p <= 4)
///   E-ST-OVERFLOW   exact rational arithmetic overflowed int64 after
///                   reduction
///   E-ST-NULL       `apply_spacetime` given a null buffer
///   E-ST-STEP       `apply_spacetime` given a step index out of range
///   E-ST-SHAPE      `apply_spacetime` given inconsistent spatial dimensions
namespace fuse {

/// Highest temporal reconstruction degree accepted. Matches
/// `Waveform::kMaxDgDegree` so the two halves of a space-time representation
/// admit the same degrees.
inline constexpr int kMaxTemporalDegree = 4;

/// Spec §7.3: the `T` of `M = T (x) W`, in the same COO triplet shape as a
/// spatial remap operator, deliberately -- the tensor product is then two
/// sparse applications rather than a special case.
///
/// Triplets are sorted by `(row, col)` with duplicates merged and exact zeros
/// dropped, so the representation is canonical: two builds from equal inputs
/// are bitwise equal.
struct TemporalProjection {
  std::vector<std::int32_t> row;  ///< target step, ascending
  std::vector<std::int32_t> col;  ///< source step, ascending within a row
  std::vector<double>       val;
  std::size_t               n_src_steps = 0;
  std::size_t               n_tgt_steps = 0;
  TemporalBasis             basis  = TemporalBasis::None;
  int                       degree = 0;
  /// The degree actually used, `min(degree, n_src_steps - 1)`. Differs from
  /// `degree` only when the source partition is too short; see the header
  /// comment.
  int                       effective_degree = 0;
  /// The intervals the matrix was built for. Carried so a caller can form the
  /// conservation sums (`sum_i T_ij dt_tgt == dt_src`) without threading them
  /// separately, and so `apply_spacetime` can be checked against them.
  Rational                  src_interval{0};
  Rational                  dst_interval{0};

  [[nodiscard]] std::size_t nnz() const noexcept { return val.size(); }

  /// `T(tgt_step, src_step)`, or 0 when the entry is structurally absent.
  /// O(log nnz) -- binary search over the canonical ordering.
  [[nodiscard]] double coefficient(std::size_t tgt_step,
                                   std::size_t src_step) const noexcept;
};

/// Non-owning COO view of a spatial remap operator `W`.
///
/// DEVIATION FROM THE BRIEF, stated because it matters to whoever lands Phase
/// 5: the brief has `apply_spacetime` take `const RemapWeights&` from
/// `fuse/backend/remap_plan.hpp`. Neither the type nor the header exists in the
/// tree -- Phase 5's spatial remap has not landed (`grep -r RemapWeights
/// include src` is empty, and no commit on any branch adds it). Rather than
/// squat on the Phase 5 name with a guessed definition that would then have to
/// be reconciled, this takes a non-owning *view* of the same COO triplets.
/// When `RemapWeights` lands, the adapter is one aggregate initialisation and
/// no signature here changes.
struct SpatialWeightsView {
  const std::int32_t* row    = nullptr;  ///< target spatial index
  const std::int32_t* col    = nullptr;  ///< source spatial index
  const double*       val    = nullptr;
  std::size_t         nnz    = 0;
  std::size_t         n_rows = 0;  ///< target spatial size
  std::size_t         n_cols = 0;  ///< source spatial size
};

/// Project the source time partition onto the target's over one window.
///
/// `slabs` must be the intersection of the two partitions -- i.e. the result of
/// `decompose(src_interval, dst_interval, period)` for the same two intervals.
/// That is checked, not assumed: passing a decomposition built from different
/// intervals is `E-ST-SLABS`, because the alternative is a silently wrong
/// matrix whose row sums still happen to look plausible.
///
/// Row sums are unity and column sums carry the time integral; see the header
/// comment for the exact statements.
[[nodiscard]] Result<TemporalProjection> build_temporal_projection(
    const SlabDecomposition& slabs, TemporalBasis basis, int degree,
    Rational src_interval, Rational dst_interval);

/// Tier 1 tensor-product application: `T` in time, `W` in space.
///
///     dst[r, c] += T(tgt_step, src_step) * sum_k W[r, k] * src[k, c]
///
/// `src` is `n_src_space * components` doubles, `dst` is
/// `spatial.n_rows * components`, both space-major (component fastest).
///
/// It ACCUMULATES and does not zero `dst`: the caller loops over the source
/// steps that feed one target step, and zeroing here would make that loop
/// silently keep only the last contribution. Zero `dst` once before the loop.
///
/// Tier 2 (true 4D intersection for moving meshes, spec §7.2) would replace
/// this with a non-factorising operator; the seam is here.
///
/// DEVIATION FROM THE BRIEF: the brief returns `void`. Every precondition
/// here (null buffers, out-of-range steps, a spatial operator whose column
/// count disagrees with `n_src_space`) is a memory-safety precondition, and an
/// unchecked one vanishes under `-O3 -DNDEBUG`. It returns `Result<void>` so
/// the checks survive Release.
[[nodiscard]] Result<void> apply_spacetime(const SpatialWeightsView& spatial,
                                           const TemporalProjection& temporal,
                                           const double* src, double* dst,
                                           std::size_t n_src_space,
                                           int components,
                                           std::size_t src_step,
                                           std::size_t tgt_step);

}  // namespace fuse
