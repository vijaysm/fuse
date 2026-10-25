#pragma once

/// One dense least-squares solve, behind one declaration, with interchangeable
/// implementations.
///
/// WHAT NEEDS IT. Quasi-Newton acceleration of a coupled iteration solves
/// `min ||V a - r||` at every iteration, where `V` holds the last few
/// difference vectors. The matrix is tall and thin (as many rows as coupled
/// interface values, as many columns as retained iterations, so typically
/// thousands by tens) and its columns become near-dependent as the iteration
/// converges -- which is exactly when the solve has to stay accurate.
///
/// WHY A FACADE. Three implementations are wanted for different reasons, and
/// they must be interchangeable without the acceleration code knowing which is
/// present:
///
///   `Builtin`        the coupler's own Householder QR with column pivoting,
///                    truncated at the numerical rank. No dependency, always
///                    available, and the reference the others are checked
///                    against -- it is the implementation whose rank
///                    truncation behaviour the iteration-count regressions
///                    were recorded with.
///   `Eigen`          Eigen's column-pivoted Householder QR, host only. A
///                    second opinion from a widely used implementation, and
///                    the one to reach for when the built-in factorisation is
///                    suspected.
///
/// A KokkosKernels backend is NOT one of them, and the option refuses that
/// value at configure time rather than accepting it and running something
/// else. KokkosKernels has no column-pivoted QR with rank truncation in the
/// form this solve needs, and a batched dense solve of a rank-deficient system
/// is a different operation with a different answer. When one exists this list
/// grows; until then, asking for it is a configuration error and says so.
///
/// The choice is a build-time option (`FUSE_LINALG_BACKEND`) and not a runtime
/// switch: the backends differ in the last bits, and a coupled iteration's
/// path -- how many iterations it takes, which columns get filtered -- depends
/// on those bits. A run has to be reproducible from its build, so the backend
/// is part of the build.
///
/// EXACTNESS IS NOT CLAIMED. These are floating-point factorisations of an
/// ill-conditioned matrix; two backends will not agree bit for bit and the
/// tests do not ask them to. What is checked is that each satisfies the
/// property the caller depends on: the residual is no larger than the
/// reference's by more than a stated tolerance, and the numerical rank is the
/// same.

#include <cstddef>
#include <vector>

#include "fuse/base/config.hpp"
#include "fuse/base/result.hpp"

namespace fuse::linalg {

/// Which implementation this build carries. Reported, not chosen at run time:
/// a bug report that says "the iteration took 14 steps" is only actionable
/// with this.
enum class Backend { Builtin, Eigen };

[[nodiscard]] Backend configured_backend() noexcept;
[[nodiscard]] const char* backend_name(Backend b) noexcept;

/// Solves `min ||A a - b||_2` for an `m x k` column-major `A`, truncating at
/// the numerical rank, and returns that rank.
///
/// `a` and `b` are overwritten (the factorisation works in place, which is why
/// they are taken by reference and not by const reference). `out` receives `k`
/// coefficients with the dropped columns set to exactly zero, so a caller can
/// tell a filtered column from one that happened to get a small weight.
///
/// Preconditions: `a.size() >= m * k`, `b.size() >= m`, `m >= k`. Violated,
/// they are a caller mistake and throw rather than returning a rank.
[[nodiscard]] std::size_t least_squares(std::vector<double>& a, std::size_t m,
                                        std::size_t k, std::vector<double>& b,
                                        std::vector<double>& out);

/// The built-in factorisation, reachable by name whatever the configured
/// backend is, so a test can compare a backend against it in the same build.
/// This is the oracle, not a fallback: the recorded iteration counts were
/// produced with it.
[[nodiscard]] std::size_t least_squares_builtin(std::vector<double>& a,
                                                std::size_t m, std::size_t k,
                                                std::vector<double>& b,
                                                std::vector<double>& out);

/// The working buffers of one built-in solve.
///
/// Held by whoever drives the iteration so that a steady-state solve touches
/// no allocator -- the property `Runner.AllocatesNothingWhileRunning` checks.
/// Every element is written before it is read on every call, so a buffer
/// carried over from a previous solve of a different size or rank cannot
/// change the answer.
struct LsScratch {
  std::vector<std::size_t> perm;
  std::vector<double>      colnorm2;
  std::vector<double>      reflector;  ///< the Householder vector v
  std::vector<double>      z;          ///< the back-substituted coefficients
};

/// `least_squares_builtin` with the scratch supplied, for the inner loop.
/// Same code, same arithmetic, same order; the buffers are the only difference.
///
/// UNLIKE `least_squares`/`least_squares_builtin`, this entry point does NOT
/// enforce `m >= k`: the accelerator's hot loop can call it
/// with more retained columns (`k`, bounded by `cfg.reuse`) than interface
/// values (`m`, the coupled problem's own size) when the interface is small,
/// and that shape is reachable in ordinary use, not just in principle. The
/// QR loop below is bounded by `steps = min(m, k)` everywhere, so `m < k`
/// does not read or write outside `a`/`b`/`out` -- it returns a rank capped
/// at `m`, with the coefficients of every column past that rank set to zero,
/// exactly like any other rank-deficient system. Confirmed under
/// ASan/UBSan at `m=1,k=8` and `m=2,k=8` (see the detailed review) and
/// pinned by `LeastSquares.IntoAgreesWithBuiltinWhenMIsLessThanK` and
/// `Accelerator.HandlesAnInterfaceSmallerThanTheReuseLimit`.
[[nodiscard]] std::size_t least_squares_into(std::vector<double>& a,
                                             std::size_t m, std::size_t k,
                                             std::vector<double>& b,
                                             std::vector<double>& out,
                                             LsScratch& work);

}  // namespace fuse::linalg
