#pragma once

/// One interface for every operation that maps field values linearly from one
/// index set to another.
///
/// WHY THIS EXISTS. A remap from a source mesh to a target mesh, a transport
/// that moves values between ranks, and an interpolation in time are the same
/// kind of object: each takes `cols` values in and produces `rows` values out,
/// each is linear, and each has an adjoint. Before this they were three
/// unrelated pieces of code with three shapes, which meant the properties that
/// matter -- that the adjoint really is the adjoint, that composing two of them
/// is the composition, that a conservative operator conserves -- were checked
/// for one and assumed for the others. They are now checked ONCE, against this
/// interface, by the conformance fixture in `test/backend/operator_conformance`.
///
/// The interface is deliberately NOT templated on an execution space, even
/// though the kernels underneath may use one. Two reasons, both structural: a
/// virtual call cannot be templated, and the execution plan holds operators
/// behind a pointer chosen at init time, so the space would have to be erased
/// again immediately. An implementation is free to run its own apply on a
/// device; what it may not do is make the *caller* choose.
///
/// WHAT AN IMPLEMENTATION PROMISES. `apply` and `apply_transpose` are exact
/// adjoints of one another in the discrete inner product -- see
/// `adjoint_defect()`. That is not a numerical nicety: the monolithic solver
/// path (S7) forms its off-diagonal Jacobian blocks from these operators, and a
/// transpose that is merely approximately the adjoint produces a Newton
/// iteration that converges to the wrong thing while looking healthy.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "fuse/base/result.hpp"

namespace fuse {

/// What an operator claims about itself, so the conformance fixture checks the
/// properties it asserts rather than a fixed list. A claim is a promise the
/// fixture will hold it to -- an operator that claims conservation and does not
/// conserve fails, and one that claims nothing is not thereby exempt from the
/// adjoint identity, which every linear operator has.
struct OperatorClaims {
  /// Row sums are 1: a spatially constant field is reproduced exactly. The
  /// usual property of an interpolation.
  bool preserves_constants = false;
  /// Column sums weighted by the target measure equal the source measure: the
  /// integral of the field is unchanged. The usual property of a conservative
  /// remap, and NOT implied by the above -- the two are the classic pair a
  /// remap can satisfy one of.
  bool conservative = false;
  /// No new extrema: every output lies within the range of the inputs that
  /// contribute to it. Requires non-negative weights summing to 1.
  bool monotone = false;
  /// `apply` does not depend on how ranks are laid out, because it touches no
  /// remote data. A transport operator says false here.
  bool rank_local = true;
};

/// A linear map from `cols()` source values to `rows()` target values, applied
/// to `components` interleaved components at a time.
class LinearOperator {
 public:
  virtual ~LinearOperator() = default;

  LinearOperator() = default;
  LinearOperator(const LinearOperator&) = delete;
  LinearOperator& operator=(const LinearOperator&) = delete;

  /// Local shape. Both are counts of ENTITIES, not of doubles: a call with
  /// `components == 3` reads `cols() * 3` and writes `rows() * 3`.
  [[nodiscard]] virtual std::size_t rows() const noexcept = 0;
  [[nodiscard]] virtual std::size_t cols() const noexcept = 0;

  /// What this operator is, for diagnostics and for the plan dump. Stable
  /// across runs; it appears in reference output.
  [[nodiscard]] virtual std::string_view kind() const noexcept = 0;

  [[nodiscard]] virtual OperatorClaims claims() const noexcept = 0;

  /// y = A x. `x` holds `cols() * components` values, `y` holds
  /// `rows() * components`; both are fully written (no accumulation into y --
  /// an operator that wants accumulation composes with an accumulate step,
  /// which keeps this contract single-valued).
  [[nodiscard]] virtual Result<void> apply(std::span<const double> x,
                                           std::span<double> y,
                                           int components) const = 0;

  /// x = A^T y, with the roles of the two lengths exchanged. Required, not
  /// optional: an operator that cannot supply its adjoint cannot appear in the
  /// monolithic path, and declaring that at the interface is better than
  /// discovering it when a solve stalls.
  [[nodiscard]] virtual Result<void> apply_transpose(std::span<const double> y,
                                                     std::span<double> x,
                                                     int components) const = 0;

  /// Row sums of A (length `rows()`) and column sums (length `cols()`).
  ///
  /// These are derivable from `apply`/`apply_transpose` on a vector of ones --
  /// and the fixture checks they agree -- but they are separately declared
  /// because a sparse operator computes them without touching field data, and
  /// the init-time consistency and conservation checks want them before any
  /// field exists.
  [[nodiscard]] virtual Result<void> row_sums(std::span<double> out) const = 0;
  [[nodiscard]] virtual Result<void> col_sums(std::span<double> out) const = 0;

  /// A hash of everything that determines what this operator computes.
  ///
  /// Two operators with equal signatures must compute the same thing; the
  /// converse is not promised (it is a 64-bit hash). Used to tell whether a
  /// persisted map matches the mesh pair it is being reloaded onto (S5.4), and
  /// to keep a plan dump stable. It must NOT include anything incidental --
  /// allocation addresses, build configuration, the order the triplets were
  /// discovered in -- or a reload that is genuinely valid will be rejected.
  [[nodiscard]] virtual std::uint64_t signature() const = 0;
};

/// ||A^T y · x - y · A x|| relative to the magnitudes involved, for one pair of
/// vectors. Exactly zero for an operator whose transpose is its adjoint, up to
/// the rounding of two different summation orders -- so the fixture compares it
/// against a tolerance scaled by the number of contributions, not against zero.
///
/// This is the one property no operator is exempt from, which is why it lives
/// here as a free function rather than in the test: an operator can be checked
/// in the field, by a user, without the test suite.
[[nodiscard]] Result<double> adjoint_defect(const LinearOperator& a,
                                            std::span<const double> x,
                                            std::span<const double> y,
                                            int components);

/// B then A, as one operator. `inner` is applied first.
///
/// Rejects at construction when the shapes do not meet, rather than at the
/// first apply: a composed operator is built at init time, and a shape mistake
/// there is a setup error that should be reported with the rest of them.
[[nodiscard]] Result<std::unique_ptr<LinearOperator>> compose(
    std::shared_ptr<const LinearOperator> outer,
    std::shared_ptr<const LinearOperator> inner);

}  // namespace fuse
