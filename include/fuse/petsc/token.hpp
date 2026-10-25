#pragma once

/// The PETSc substrate behind the Newton seam (docs/dev/newton-krylov.md
/// Sec. 10): a `fuse::LinearizationToken` whose Jacobian action is PETSc's
/// own matrix-free finite-difference matrix (`MATMFFD`, Walker-Pernice
/// differencing) and whose preconditioner is a PETsc `PC`, built and
/// consumed through `tools/check-layering.py`'s `petsc` layer -- the
/// topmost layer, depending on `fuse::runtime` (for `fuse::LinearizationToken`,
/// fuse/runtime/component.hpp) and `fuse::num` (for `fuse::num::
/// PreconditionerBlock`/`BlockJacobiPreconditioner`, reused rather than
/// reimplemented for the block-Jacobi composition below), depended on by
/// nothing.
///
/// NO PETSC TYPE NAMES THIS FILE (`tools/check-layering.py`'s
/// `FORBIDDEN_PUBLIC` regex matches `petsc` case-sensitively in any public
/// header's `#include` line, at every layer, no exception for this one):
/// every PETSc call lives in `src/petsc/token.cpp`, behind the plain
/// `std::span`-based shapes below. A caller that wants the richer
/// `set_base_point()`/`set_block_preconditioner()` interface holds the
/// concrete `fuse::petsc::MffdToken` this factory actually returns (which
/// IS-A `fuse::LinearizationToken`, usable anywhere the base interface is
/// asked for); a caller that only wants the base `LinearizationToken`
/// interface may simply store it as one.
///
/// BUILD. Compiled into `fuse-coupler-petsc` only when `FUSE_ENABLE_PETSC`
/// is ON (cmake/FuseDependencies.cmake, `PkgConfig::PETSc` from
/// `/opt/petsc/download-moab`); `FUSE_HAVE_PETSC` (fuse/base/config.hpp)
/// is unconditionally `0` in a build without it, and NOTHING in this
/// header is conditional on that macro -- it is PETSc-type-free, so it
/// compiles (as declarations nobody calls) either way. A consumer that
/// calls `make_mffd_token()`/`make_assembled_token()` in a build without
/// PETSC links against nothing (the library target does not exist); that
/// refusal is the caller's own (the Newton-region graph selection returns
/// `E-NEWTON-PETSC-UNAVAILABLE`), not this header's concern.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <vector>

#include "fuse/base/result.hpp"
#include "fuse/num/newton_krylov.hpp"  // PreconditionerBlock, MffdOptions, default_mffd_error_rel
#include "fuse/runtime/component.hpp"  // fuse::LinearizationToken

namespace fuse::petsc {

using fuse::Result;

// ===========================================================================
// PetscInitialize/PetscFinalize ownership.
// ===========================================================================

/// Initialises PETSc on the communicator `comm_f` (an `MPI_Comm_c2f()`
/// handle, exactly `fuse::linalg::make_mpi_reducer`'s own convention,
/// fuse/linalg/comm.hpp) ONLY IF `PetscInitialized()` is currently false.
/// Idempotent and safe to call from every token/driver factory below (and
/// does, at the top of each): the FIRST call in a process does the real
/// work and records that THIS library did it; every later call, from
/// anywhere, is a no-op success. Never finalises MPI itself -- see
/// `finalize_if_owned()`.
///
/// TO BE MEASURED (test/petsc/init_test.cpp, a process of its own -- see
/// that file's own header comment for why it is never shared with
/// another PETSc test binary): PETSc's own documented behaviour is that
/// `PetscInitialize()` calls `MPI_Init()` itself when nothing has called
/// it yet; that test is what checks `MPI_Initialized()` false-then-true
/// across the call in THIS sandbox (no `FUSE_ENABLE_MPI` harness, no
/// prior PETSc use in the process) rather than leaving the claim
/// unverified. `finalize_if_owned()` below relies on PETSc's OWN internal
/// tracking of that fact (it calls `MPI_Finalize()` from inside
/// `PetscFinalize()` if and only if `PetscInitialize()` is the one that
/// called `MPI_Init()`), so this library's own ownership flag only needs
/// to answer "did I call `PetscInitialize()` at all", not "did I call
/// `MPI_Init()`" -- the two can disagree (a process with `FUSE_ENABLE_MPI`
/// that calls `MPI_Init()` itself before any PETSc call would make PETSc
/// skip its own `MPI_Init()`, and then correctly skip `MPI_Finalize()`
/// too), and PETSc's own bookkeeping already gets that right.
[[nodiscard]] Result<void> ensure_initialized(int comm_f);

/// `PetscFinalize()`, ONLY if THIS library's `ensure_initialized()` call
/// was the one that actually initialised PETSc (a no-op, successful,
/// otherwise -- e.g. a process where PETSc was already up before this
/// library ever ran, or one where `ensure_initialized()` was never
/// called). Never finalises an MPI this library did not start, per
/// PETSc's own internal tracking (see `ensure_initialized()`'s comment).
///
/// NOT called by any token/driver destructor and not called by any test
/// except `test/petsc/init_test.cpp`'s own dedicated,
/// single-test binary: `MPI_Finalize()` is irreversible within a
/// process (no `MPI_Init()` after it), so calling this from a test that
/// shares a binary with other PETSc tests would strand every test that
/// runs after it. Harmless to never call in a short-lived test process --
/// the OS reclaims everything at exit -- which is why every OTHER test
/// binary in test/petsc/ relies on process exit instead.
void finalize_if_owned();

/// TEST-ONLY: whether `ensure_initialized()` has recorded this library as
/// PETSc's owner in THIS process (`test/petsc/init_test.cpp`'s own
/// assertion hook -- the ownership flag is otherwise private state).
[[nodiscard]] bool owns_petsc_initialization();

/// `MPI_Comm_c2f(MPI_COMM_SELF)` -- callable before `ensure_initialized()`
/// (needs only `<mpi.h>`, never PETSc itself), for a caller that wants
/// the single-rank communicator (`fuse::petsc::solve()`'s own header
/// comment).
[[nodiscard]] int comm_self_f();

// ===========================================================================
// The MFFD-backed token.
// ===========================================================================

/// `F(x)`, read from and written to host memory -- the identical
/// convention `fuse::FixedPointResidual::evaluate`
/// (fuse/runtime/fixed_point_residual.hpp) and
/// `fuse::linalg::HostApplyFn` (fuse/linalg/derivative_gate.hpp) already
/// use, so a caller in `fuse::runtime` adapts either directly with no
/// extra copy beyond what `std::span` already is.
using ResidualSpanFn =
    std::function<Result<void>(std::span<const double> x, std::span<double> fx)>;

/// `fuse::num::MffdOptions` carries one field this token also needs
/// (`error_rel`) and one this token does not (`min_direction_norm`: PETSc's
/// own `MATMFFD_WP` divides by `||v||` with NO floor of its own -- see
/// `src/petsc/token.cpp`'s header comment for what was measured when `v`
/// is exactly zero, the same corner `fuse::num::MffdJacobianAction`'s own
/// floor exists for). `error_rel` defaults to EXACTLY
/// `fuse::num::default_mffd_error_rel()` (`sqrt(eps_mach)`) -- the same
/// default both the native action and PETSc's own `MATMFFD_WP` already
/// use unset -- so the agreement check compares two actions that
/// differ only in which code computed `h` from the identical formula and
/// the identical inputs, never in the `error_rel` constant itself.
struct MffdTokenOptions {
  double error_rel = fuse::num::default_mffd_error_rel();
};

/// `fuse::LinearizationToken::apply_jacobian_transpose`'s own refusal
/// code for this token: `MATMFFD` never implements `MatMultTranspose` (no
/// transpose exists for a one-sided finite-difference action), so this
/// token refuses rather than asking PETSc for an operation that would
/// itself error out less legibly.
inline constexpr const char* kNoTransposeCode = "E-PETSC-NO-TRANSPOSE";

/// `apply_jacobian()`/`apply_preconditioner()` called before
/// `set_base_point()` ever ran -- the same "programmer error, refused"
/// shape `fuse::num::MffdJacobianAction::apply`'s own
/// `E-NEWTON-MFFD-NO-POINT` uses for the identical mistake.
inline constexpr const char* kNoPointCode = "E-PETSC-MFFD-NO-POINT";

/// `fuse::LinearizationToken` backed by PETSc's `MATMFFD` (Walker-Pernice
/// differencing, `MATMFFD_WP`), with one method beyond the base
/// interface: `set_base_point()`, the point `apply_jacobian()`
/// linearises about (the base interface has no notion of "the point this
/// token was returned for" beyond `signature()`, which only reports
/// WHETHER it moved, never lets a caller MOVE it). A caller holding the
/// concrete type (what `make_mffd_token()` returns) can call it directly;
/// one holding only a `fuse::LinearizationToken*` cannot move the point
/// at all, which is intentional -- `ImplicitComponent::linearization()`'s
/// own contract is that the token is valid only "for the point of the
/// most recent `evaluate_residual()` call", i.e. the POINT is the
/// component's business, not a generic Newton-loop caller's.
class MffdToken : public fuse::LinearizationToken {
 public:
  ~MffdToken() override = default;

  /// Sets the linearisation point to `x` (`x.size() == cols()`).
  /// `fx`, when non-empty, is `F(x)` already known by the caller (so no
  /// extra evaluation is spent to obtain it, matching
  /// `fuse::num::MffdJacobianAction::set_point`'s own "never calls `f`
  /// itself" guarantee); when empty, PETSc's own `MATMFFD` evaluates
  /// `F(x)` internally at the next `MatMult` -- ONE MORE function
  /// evaluation than the native path pays for the SAME base-point change
  /// when the caller does not already have `F(x)` in hand, a
  /// measured difference rather than silently absorbed.
  virtual Result<void> set_base_point(std::span<const double> x,
                                      std::span<const double> fx = {}) = 0;

  /// Installs a block-Jacobi preconditioner, reusing
  /// `fuse::num::BlockJacobiPreconditioner::from_blocks` (fuse/num/
  /// newton_krylov.hpp) for both the exact tiling-coverage check
  /// (`E-NEWTON-PRECOND-COVERAGE`) and the per-block apply -- this class
  /// never reimplements the tiling logic, only adapts it onto a PETSc
  /// `PCSHELL`. An empty `blocks` list (or never calling this at all)
  /// leaves `apply_preconditioner()` at `PCNONE` (the identity), the
  /// same always-legal fallback `LinearizationToken::apply_preconditioner`'s
  /// own comment licenses.
  virtual Result<void> set_block_preconditioner(
      std::vector<fuse::num::PreconditionerBlock> blocks) = 0;

  /// TEST/DIAGNOSTIC-ONLY: the Walker-Pernice step `h` PETSc's own
  /// `MatMFFDGetH` reports from the MOST RECENT `apply_jacobian()` call
  /// (`0.0` before the first one) -- `test/petsc/derivative_gate_test.cpp`'s
  /// own agreement check reads this and `fuse::num::MffdJacobianAction::
  /// trace()`'s matching `h` side by side (`docs/dev/newton-krylov.md`
  /// Sec. 10's own measured pair).
  [[nodiscard]] virtual double last_h() const = 0;
};

/// Builds an `MffdToken` of size `n` (square: `rows() == cols() == n`)
/// over `f`, on the communicator `comm_f`. Calls `ensure_initialized(comm_f)`
/// first. The returned token has NO base point set yet -- `apply_jacobian`/
/// `apply_preconditioner` refuse `kNoPointCode` until `set_base_point()`
/// runs at least once.
[[nodiscard]] Result<std::unique_ptr<MffdToken>> make_mffd_token(
    ResidualSpanFn f, std::size_t n, int comm_f, const MffdTokenOptions& opt = {});

// ===========================================================================
// The assembled token (a real PETSc Mat, for the adjoint-identity test:
// the MFFD token above has no transpose to test against).
// ===========================================================================

/// One nonzero of a `rows x cols` matrix, row/col 0-based.
struct MatrixTriplet {
  std::size_t row = 0;
  std::size_t col = 0;
  double value = 0.0;
};

/// Builds a `fuse::LinearizationToken` over a real, assembled PETSc `Mat`
/// (`MATAIJ`, single-rank `MatCreateSeqAIJ`) from `triplets` -- duplicate
/// (row, col) pairs accumulate (PETSc's own `ADD_VALUES` convention), not
/// an error. `apply_jacobian`/`apply_jacobian_transpose` are `MatMult`/
/// `MatMultTranspose` against the SAME assembled matrix, so (unlike
/// `MffdToken`) BOTH directions are genuinely available -- this is the
/// token `test/petsc/derivative_gate_test.cpp`'s own adjoint-identity
/// check runs against, since the MFFD token has no transpose to compare.
/// `apply_preconditioner` is `PCNONE` (the identity) always; no
/// `set_block_preconditioner` exists for this token (nothing needs one).
[[nodiscard]] Result<std::unique_ptr<fuse::LinearizationToken>> make_assembled_token(
    std::size_t rows, std::size_t cols, const std::vector<MatrixTriplet>& triplets,
    int comm_f);

}  // namespace fuse::petsc
