// src/petsc/token.cpp -- the PETSc substrate's LinearizationToken pair
// (fuse/petsc/token.hpp's own header comment has the design). The ONE
// translation unit (alongside solve.cpp) in fuse::petsc that names
// <petsc.h> -- tools/check-layering.py's FORBIDDEN_PUBLIC rule only
// reaches public headers, so this file, a private .cpp, is where every
// PETSc call actually lives.

#include "fuse/petsc/token.hpp"

#include <petsc.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>

#include "fuse/linalg/comm.hpp"
#include "fuse/linalg/dist_vector.hpp"

namespace fuse::petsc {

namespace {

Result<void> check(PetscErrorCode ierr, const std::string& what) {
  if (ierr) {
    return Failure{"E-PETSC-CALL", what + ": PETSc error code " + std::to_string(ierr)};
  }
  return Result<void>::success();
}

// ---------------------------------------------------------------------------
// The MFFD token's own MatMFFDFn callback: PETSc hands this (x, y) Vecs
// that must stay owned by PETSc's own Vec objects -- the ONLY copy this
// callback performs is into/out of the caller's ResidualSpanFn, through
// VecGetArray/VecGetArrayRead, exactly the boundary fuse::petsc::
// ResidualSpanFn's own header comment describes.
// ---------------------------------------------------------------------------

struct MffdFnContext {
  ResidualSpanFn f;
  std::size_t n = 0;
};

PetscErrorCode mffd_residual_fn(void* raw_ctx, Vec x, Vec y) {
  auto* ctx = static_cast<MffdFnContext*>(raw_ctx);
  const PetscScalar* xarr = nullptr;
  PetscScalar* yarr = nullptr;
  PetscErrorCode ierr = VecGetArrayRead(x, &xarr);
  if (ierr) return ierr;
  ierr = VecGetArray(y, &yarr);
  if (ierr) {
    VecRestoreArrayRead(x, &xarr);
    return ierr;
  }
  std::span<const double> xs(xarr, ctx->n);
  std::span<double> ys(yarr, ctx->n);
  Result<void> r = ctx->f(xs, ys);
  ierr = VecRestoreArrayRead(x, &xarr);
  if (ierr) return ierr;
  ierr = VecRestoreArray(y, &yarr);
  if (ierr) return ierr;
  // A MatMFFDFn signals its own failure by returning a nonzero
  // PetscErrorCode -- the documented mechanism for this callback, distinct
  // from SNESSetFunctionDomainError() (which is the OUTER F's own
  // domain-error signal, not this inner differencing callback's).
  if (!r.ok()) return static_cast<PetscErrorCode>(PETSC_ERR_USER);
  return PETSC_SUCCESS;
}

// ---------------------------------------------------------------------------
// The block-Jacobi PCSHELL: reuses fuse::num::BlockJacobiPreconditioner
// (fuse/num/newton_krylov.hpp) unchanged for both the tiling-coverage
// check (from_blocks) and the per-block apply -- this file never
// reimplements that logic, only crosses the Vec/span boundary around it
// via a throwaway single-rank DistVector pair (the tier-0,
// single-rank scope makes the extra copy's cost irrelevant; see
// fuse/petsc/solve.hpp's own header comment on scope).
// ---------------------------------------------------------------------------

struct BlockPcContext {
  fuse::num::BlockJacobiPreconditioner precond;
  std::size_t n = 0;
};

PetscErrorCode block_pc_apply(PC pc, Vec r, Vec z) {
  void* raw_ctx = nullptr;
  PetscErrorCode ierr = PCShellGetContext(pc, &raw_ctx);
  if (ierr) return ierr;
  auto* ctx = static_cast<BlockPcContext*>(raw_ctx);

  const PetscScalar* rarr = nullptr;
  PetscScalar* zarr = nullptr;
  ierr = VecGetArrayRead(r, &rarr);
  if (ierr) return ierr;
  ierr = VecGetArray(z, &zarr);
  if (ierr) {
    VecRestoreArrayRead(r, &rarr);
    return ierr;
  }

  auto exchanger = fuse::linalg::make_serial_exchanger();
  auto reducer = fuse::linalg::make_serial_reducer();
  fuse::linalg::DistVector rv(ctx->n, exchanger, reducer);
  fuse::linalg::DistVector zv(ctx->n, exchanger, reducer);
  std::copy(rarr, rarr + ctx->n, rv.data());
  Result<void> res = ctx->precond.apply(rv, zv);
  std::copy(zv.data(), zv.data() + ctx->n, zarr);

  ierr = VecRestoreArrayRead(r, &rarr);
  if (ierr) return ierr;
  ierr = VecRestoreArray(z, &zarr);
  if (ierr) return ierr;
  if (!res.ok()) return static_cast<PetscErrorCode>(PETSC_ERR_USER);
  return PETSC_SUCCESS;
}

// ---------------------------------------------------------------------------
// MffdToken implementation.
// ---------------------------------------------------------------------------

class MffdTokenImpl final : public MffdToken {
 public:
  MffdTokenImpl(ResidualSpanFn f, std::size_t n, MPI_Comm comm, MffdTokenOptions opt)
      : ctx_{std::move(f), n}, n_(n), opt_(opt), comm_(comm) {}

  ~MffdTokenImpl() override {
    if (pc_) PCDestroy(&pc_);
    if (base_vec_) VecDestroy(&base_vec_);
    if (fx_vec_) VecDestroy(&fx_vec_);
    if (v_in_) VecDestroy(&v_in_);
    if (v_out_) VecDestroy(&v_out_);
    if (J_) MatDestroy(&J_);
  }

  Result<void> initialize() {
    PetscErrorCode ierr = MatCreateMFFD(comm_, static_cast<PetscInt>(n_), static_cast<PetscInt>(n_),
                                        static_cast<PetscInt>(n_), static_cast<PetscInt>(n_), &J_);
    if (auto r = check(ierr, "MatCreateMFFD"); !r.ok()) return r;
    ierr = MatMFFDSetType(J_, const_cast<char*>(MATMFFD_WP));
    if (auto r = check(ierr, "MatMFFDSetType"); !r.ok()) return r;
    ierr = MatMFFDSetFunction(J_, mffd_residual_fn, &ctx_);
    if (auto r = check(ierr, "MatMFFDSetFunction"); !r.ok()) return r;
    ierr = MatMFFDSetFunctionError(J_, opt_.error_rel);
    if (auto r = check(ierr, "MatMFFDSetFunctionError"); !r.ok()) return r;
    ierr = MatSetUp(J_);
    if (auto r = check(ierr, "MatSetUp"); !r.ok()) return r;

    ierr = VecCreateSeq(comm_, static_cast<PetscInt>(n_), &base_vec_);
    if (auto r = check(ierr, "VecCreateSeq(base)"); !r.ok()) return r;
    ierr = VecCreateSeq(comm_, static_cast<PetscInt>(n_), &v_in_);
    if (auto r = check(ierr, "VecCreateSeq(v_in)"); !r.ok()) return r;
    ierr = VecCreateSeq(comm_, static_cast<PetscInt>(n_), &v_out_);
    if (auto r = check(ierr, "VecCreateSeq(v_out)"); !r.ok()) return r;

    ierr = PCCreate(comm_, &pc_);
    if (auto r = check(ierr, "PCCreate"); !r.ok()) return r;
    ierr = PCSetType(pc_, PCNONE);
    if (auto r = check(ierr, "PCSetType(NONE)"); !r.ok()) return r;
    // PCSetUp refuses "Missing A matrix" for ANY PC type, PCNONE and
    // PCSHELL included, until an operator is attached (the first run of
    // every apply_preconditioner case fails exactly there). The token's own
    // matrix is that operator.
    ierr = PCSetOperators(pc_, J_, J_);
    if (auto r = check(ierr, "PCSetOperators"); !r.ok()) return r;
    return Result<void>::success();
  }

  Result<void> set_base_point(std::span<const double> x, std::span<const double> fx) override {
    if (x.size() != n_) {
      return Failure{"E-PETSC-MFFD-SHAPE", "set_base_point: x has " + std::to_string(x.size()) +
                                               " entries, token is " + std::to_string(n_)};
    }
    PetscScalar* arr = nullptr;
    PetscErrorCode ierr = VecGetArray(base_vec_, &arr);
    if (auto r = check(ierr, "VecGetArray(base)"); !r.ok()) return r;
    std::copy(x.begin(), x.end(), arr);
    ierr = VecRestoreArray(base_vec_, &arr);
    if (auto r = check(ierr, "VecRestoreArray(base)"); !r.ok()) return r;

    Vec fu = nullptr;
    if (!fx.empty()) {
      if (fx.size() != n_) {
        return Failure{"E-PETSC-MFFD-SHAPE", "set_base_point: fx has " + std::to_string(fx.size()) +
                                                 " entries, token is " + std::to_string(n_)};
      }
      if (!fx_vec_) {
        ierr = VecCreateSeq(comm_, static_cast<PetscInt>(n_), &fx_vec_);
        if (auto r = check(ierr, "VecCreateSeq(fx)"); !r.ok()) return r;
      }
      ierr = VecGetArray(fx_vec_, &arr);
      if (auto r = check(ierr, "VecGetArray(fx)"); !r.ok()) return r;
      std::copy(fx.begin(), fx.end(), arr);
      ierr = VecRestoreArray(fx_vec_, &arr);
      if (auto r = check(ierr, "VecRestoreArray(fx)"); !r.ok()) return r;
      fu = fx_vec_;
    }
    ierr = MatMFFDSetBase(J_, base_vec_, fu);
    if (auto r = check(ierr, "MatMFFDSetBase"); !r.ok()) return r;
    has_point_ = true;
    ++generation_;
    return Result<void>::success();
  }

  Result<void> set_block_preconditioner(std::vector<fuse::num::PreconditionerBlock> blocks) override {
    auto built = fuse::num::BlockJacobiPreconditioner::from_blocks(std::move(blocks), n_);
    if (!built.ok()) return built.error();
    block_pc_ctx_ =
        std::make_unique<BlockPcContext>(BlockPcContext{std::move(built).value(), n_});
    PetscErrorCode ierr = PCSetType(pc_, PCSHELL);
    if (auto r = check(ierr, "PCSetType(SHELL)"); !r.ok()) return r;
    ierr = PCShellSetContext(pc_, block_pc_ctx_.get());
    if (auto r = check(ierr, "PCShellSetContext"); !r.ok()) return r;
    ierr = PCShellSetApply(pc_, block_pc_apply);
    if (auto r = check(ierr, "PCShellSetApply"); !r.ok()) return r;
    ierr = PCShellSetName(pc_, "fuse-block-jacobi");
    if (auto r = check(ierr, "PCShellSetName"); !r.ok()) return r;
    ++precond_generation_;
    return Result<void>::success();
  }

  Result<void> apply_jacobian(std::span<const double> x, std::span<double> y) override {
    if (!has_point_) {
      return Failure{kNoPointCode, "apply_jacobian: set_base_point was never called"};
    }
    if (x.size() != n_ || y.size() != n_) {
      return Failure{"E-PETSC-MFFD-SHAPE", "apply_jacobian: shape mismatch"};
    }
    PetscScalar* arr = nullptr;
    PetscErrorCode ierr = VecGetArray(v_in_, &arr);
    if (auto r = check(ierr, "VecGetArray(v_in)"); !r.ok()) return r;
    std::copy(x.begin(), x.end(), arr);
    ierr = VecRestoreArray(v_in_, &arr);
    if (auto r = check(ierr, "VecRestoreArray(v_in)"); !r.ok()) return r;

    ierr = MatMult(J_, v_in_, v_out_);
    if (auto r = check(ierr, "MatMult"); !r.ok()) return r;

    const PetscScalar* oarr = nullptr;
    ierr = VecGetArrayRead(v_out_, &oarr);
    if (auto r = check(ierr, "VecGetArrayRead(v_out)"); !r.ok()) return r;
    std::copy(oarr, oarr + n_, y.begin());
    ierr = VecRestoreArrayRead(v_out_, &oarr);
    if (auto r = check(ierr, "VecRestoreArrayRead(v_out)"); !r.ok()) return r;

    PetscScalar h = 0.0;
    ierr = MatMFFDGetH(J_, &h);
    if (auto r = check(ierr, "MatMFFDGetH"); !r.ok()) return r;
    last_h_ = PetscRealPart(h);
    return Result<void>::success();
  }

  Result<void> apply_jacobian_transpose(std::span<const double>, std::span<double>) override {
    return Failure{kNoTransposeCode,
                   "MATMFFD has no transpose: a one-sided finite-difference action is not "
                   "invertible-by-transpose"};
  }

  Result<void> apply_preconditioner(std::span<const double> r, std::span<double> z) override {
    if (r.size() != n_ || z.size() != n_) {
      return Failure{"E-PETSC-MFFD-SHAPE", "apply_preconditioner: shape mismatch"};
    }
    PetscErrorCode ierr = PCSetUp(pc_);
    if (auto res = check(ierr, "PCSetUp"); !res.ok()) return res;

    PetscScalar* arr = nullptr;
    ierr = VecGetArray(v_in_, &arr);
    if (auto res = check(ierr, "VecGetArray(pc in)"); !res.ok()) return res;
    std::copy(r.begin(), r.end(), arr);
    ierr = VecRestoreArray(v_in_, &arr);
    if (auto res = check(ierr, "VecRestoreArray(pc in)"); !res.ok()) return res;

    ierr = PCApply(pc_, v_in_, v_out_);
    if (auto res = check(ierr, "PCApply"); !res.ok()) return res;

    const PetscScalar* oarr = nullptr;
    ierr = VecGetArrayRead(v_out_, &oarr);
    if (auto res = check(ierr, "VecGetArrayRead(pc out)"); !res.ok()) return res;
    std::copy(oarr, oarr + n_, z.begin());
    ierr = VecRestoreArrayRead(v_out_, &oarr);
    if (auto res = check(ierr, "VecRestoreArrayRead(pc out)"); !res.ok()) return res;
    return Result<void>::success();
  }

  [[nodiscard]] std::size_t rows() const override { return n_; }
  [[nodiscard]] std::size_t cols() const override { return n_; }
  [[nodiscard]] std::uint64_t signature() const override {
    return (static_cast<std::uint64_t>(generation_) << 32) |
           static_cast<std::uint64_t>(precond_generation_);
  }
  [[nodiscard]] double last_h() const override { return last_h_; }

 private:
  MffdFnContext ctx_;
  std::size_t n_ = 0;
  MffdTokenOptions opt_;
  MPI_Comm comm_;
  Mat J_ = nullptr;
  Vec base_vec_ = nullptr;
  Vec fx_vec_ = nullptr;
  Vec v_in_ = nullptr;
  Vec v_out_ = nullptr;
  PC pc_ = nullptr;
  std::unique_ptr<BlockPcContext> block_pc_ctx_;
  bool has_point_ = false;
  std::uint32_t generation_ = 0;
  std::uint32_t precond_generation_ = 0;
  double last_h_ = 0.0;
};

// ---------------------------------------------------------------------------
// The assembled token: a real MATAIJ, so both apply_jacobian and
// apply_jacobian_transpose are genuine (MatMult/MatMultTranspose) -- the
// adjoint-identity check's own counterpart to the MFFD token, which has
// no transpose at all.
// ---------------------------------------------------------------------------

class AssembledTokenImpl final : public fuse::LinearizationToken {
 public:
  AssembledTokenImpl(std::size_t rows, std::size_t cols, MPI_Comm comm)
      : rows_(rows), cols_(cols), comm_(comm) {}

  ~AssembledTokenImpl() override {
    if (pc_) PCDestroy(&pc_);
    if (v_in_c_) VecDestroy(&v_in_c_);
    if (v_out_r_) VecDestroy(&v_out_r_);
    if (v_in_r_) VecDestroy(&v_in_r_);
    if (v_out_c_) VecDestroy(&v_out_c_);
    if (A_) MatDestroy(&A_);
  }

  Result<void> initialize(const std::vector<MatrixTriplet>& triplets) {
    PetscErrorCode ierr =
        MatCreateSeqAIJ(comm_, static_cast<PetscInt>(rows_), static_cast<PetscInt>(cols_),
                        static_cast<PetscInt>(cols_), nullptr, &A_);
    if (auto r = check(ierr, "MatCreateSeqAIJ"); !r.ok()) return r;
    for (const MatrixTriplet& t : triplets) {
      PetscInt row = static_cast<PetscInt>(t.row);
      PetscInt col = static_cast<PetscInt>(t.col);
      PetscScalar value = t.value;
      ierr = MatSetValues(A_, 1, &row, 1, &col, &value, ADD_VALUES);
      if (auto r = check(ierr, "MatSetValues"); !r.ok()) return r;
    }
    ierr = MatAssemblyBegin(A_, MAT_FINAL_ASSEMBLY);
    if (auto r = check(ierr, "MatAssemblyBegin"); !r.ok()) return r;
    ierr = MatAssemblyEnd(A_, MAT_FINAL_ASSEMBLY);
    if (auto r = check(ierr, "MatAssemblyEnd"); !r.ok()) return r;

    ierr = VecCreateSeq(comm_, static_cast<PetscInt>(cols_), &v_in_c_);
    if (auto r = check(ierr, "VecCreateSeq(v_in_c)"); !r.ok()) return r;
    ierr = VecCreateSeq(comm_, static_cast<PetscInt>(rows_), &v_out_r_);
    if (auto r = check(ierr, "VecCreateSeq(v_out_r)"); !r.ok()) return r;
    ierr = VecCreateSeq(comm_, static_cast<PetscInt>(rows_), &v_in_r_);
    if (auto r = check(ierr, "VecCreateSeq(v_in_r)"); !r.ok()) return r;
    ierr = VecCreateSeq(comm_, static_cast<PetscInt>(cols_), &v_out_c_);
    if (auto r = check(ierr, "VecCreateSeq(v_out_c)"); !r.ok()) return r;

    ierr = PCCreate(comm_, &pc_);
    if (auto r = check(ierr, "PCCreate"); !r.ok()) return r;
    ierr = PCSetType(pc_, PCNONE);
    if (auto r = check(ierr, "PCSetType(NONE)"); !r.ok()) return r;
    // PCSetUp refuses "Missing A matrix" for ANY PC type, PCNONE and
    // PCSHELL included, until an operator is attached (the first run of
    // every apply_preconditioner case fails exactly there). The token's own
    // matrix is that operator.
    ierr = PCSetOperators(pc_, A_, A_);
    if (auto r = check(ierr, "PCSetOperators"); !r.ok()) return r;
    return Result<void>::success();
  }

  Result<void> apply_jacobian(std::span<const double> x, std::span<double> y) override {
    if (x.size() != cols_ || y.size() != rows_) {
      return Failure{"E-PETSC-ASSEMBLED-SHAPE", "apply_jacobian: shape mismatch"};
    }
    PetscScalar* arr = nullptr;
    PetscErrorCode ierr = VecGetArray(v_in_c_, &arr);
    if (auto r = check(ierr, "VecGetArray(v_in_c)"); !r.ok()) return r;
    std::copy(x.begin(), x.end(), arr);
    ierr = VecRestoreArray(v_in_c_, &arr);
    if (auto r = check(ierr, "VecRestoreArray(v_in_c)"); !r.ok()) return r;

    ierr = MatMult(A_, v_in_c_, v_out_r_);
    if (auto r = check(ierr, "MatMult"); !r.ok()) return r;

    const PetscScalar* oarr = nullptr;
    ierr = VecGetArrayRead(v_out_r_, &oarr);
    if (auto r = check(ierr, "VecGetArrayRead(v_out_r)"); !r.ok()) return r;
    std::copy(oarr, oarr + rows_, y.begin());
    ierr = VecRestoreArrayRead(v_out_r_, &oarr);
    if (auto r = check(ierr, "VecRestoreArrayRead(v_out_r)"); !r.ok()) return r;
    return Result<void>::success();
  }

  Result<void> apply_jacobian_transpose(std::span<const double> x, std::span<double> y) override {
    if (x.size() != rows_ || y.size() != cols_) {
      return Failure{"E-PETSC-ASSEMBLED-SHAPE", "apply_jacobian_transpose: shape mismatch"};
    }
    PetscScalar* arr = nullptr;
    PetscErrorCode ierr = VecGetArray(v_in_r_, &arr);
    if (auto r = check(ierr, "VecGetArray(v_in_r)"); !r.ok()) return r;
    std::copy(x.begin(), x.end(), arr);
    ierr = VecRestoreArray(v_in_r_, &arr);
    if (auto r = check(ierr, "VecRestoreArray(v_in_r)"); !r.ok()) return r;

    ierr = MatMultTranspose(A_, v_in_r_, v_out_c_);
    if (auto r = check(ierr, "MatMultTranspose"); !r.ok()) return r;

    const PetscScalar* oarr = nullptr;
    ierr = VecGetArrayRead(v_out_c_, &oarr);
    if (auto r = check(ierr, "VecGetArrayRead(v_out_c)"); !r.ok()) return r;
    std::copy(oarr, oarr + cols_, y.begin());
    ierr = VecRestoreArrayRead(v_out_c_, &oarr);
    if (auto r = check(ierr, "VecRestoreArrayRead(v_out_c)"); !r.ok()) return r;
    return Result<void>::success();
  }

  Result<void> apply_preconditioner(std::span<const double> r, std::span<double> z) override {
    if (r.size() != rows_ || z.size() != cols_) {
      return Failure{"E-PETSC-ASSEMBLED-SHAPE", "apply_preconditioner: shape mismatch"};
    }
    if (rows_ != cols_) {
      return Failure{"E-PETSC-PRECOND-SHAPE",
                     "apply_preconditioner: PCNONE identity needs a square operator, this "
                     "token is " +
                         std::to_string(rows_) + "x" + std::to_string(cols_)};
    }
    PetscErrorCode ierr = PCSetUp(pc_);
    if (auto res = check(ierr, "PCSetUp"); !res.ok()) return res;

    PetscScalar* arr = nullptr;
    ierr = VecGetArray(v_in_r_, &arr);
    if (auto res = check(ierr, "VecGetArray(pc in)"); !res.ok()) return res;
    std::copy(r.begin(), r.end(), arr);
    ierr = VecRestoreArray(v_in_r_, &arr);
    if (auto res = check(ierr, "VecRestoreArray(pc in)"); !res.ok()) return res;

    ierr = PCApply(pc_, v_in_r_, v_out_r_);
    if (auto res = check(ierr, "PCApply"); !res.ok()) return res;

    const PetscScalar* oarr = nullptr;
    ierr = VecGetArrayRead(v_out_r_, &oarr);
    if (auto res = check(ierr, "VecGetArrayRead(pc out)"); !res.ok()) return res;
    std::copy(oarr, oarr + cols_, z.begin());
    ierr = VecRestoreArrayRead(v_out_r_, &oarr);
    if (auto res = check(ierr, "VecRestoreArrayRead(pc out)"); !res.ok()) return res;
    return Result<void>::success();
  }

  [[nodiscard]] std::size_t rows() const override { return rows_; }
  [[nodiscard]] std::size_t cols() const override { return cols_; }
  [[nodiscard]] std::uint64_t signature() const override { return 1; }  // never moves

 private:
  std::size_t rows_ = 0;
  std::size_t cols_ = 0;
  MPI_Comm comm_;
  Mat A_ = nullptr;
  Vec v_in_c_ = nullptr;
  Vec v_out_r_ = nullptr;
  Vec v_in_r_ = nullptr;
  Vec v_out_c_ = nullptr;
  PC pc_ = nullptr;
};

}  // namespace

Result<std::unique_ptr<MffdToken>> make_mffd_token(ResidualSpanFn f, std::size_t n, int comm_f,
                                                   const MffdTokenOptions& opt) {
  if (auto r = ensure_initialized(comm_f); !r.ok()) return r.error();
  if (n == 0) return Failure{"E-PETSC-MFFD-SHAPE", "make_mffd_token: n must be >= 1"};
  MPI_Comm comm = MPI_Comm_f2c(comm_f);
  auto impl = std::make_unique<MffdTokenImpl>(std::move(f), n, comm, opt);
  if (auto r = impl->initialize(); !r.ok()) return r.error();
  return std::unique_ptr<MffdToken>(std::move(impl));
}

Result<std::unique_ptr<fuse::LinearizationToken>> make_assembled_token(
    std::size_t rows, std::size_t cols, const std::vector<MatrixTriplet>& triplets, int comm_f) {
  if (auto r = ensure_initialized(comm_f); !r.ok()) return r.error();
  if (rows == 0 || cols == 0) {
    return Failure{"E-PETSC-ASSEMBLED-SHAPE", "make_assembled_token: rows and cols must both be >= 1"};
  }
  MPI_Comm comm = MPI_Comm_f2c(comm_f);
  auto impl = std::make_unique<AssembledTokenImpl>(rows, cols, comm);
  if (auto r = impl->initialize(triplets); !r.ok()) return r.error();
  return std::unique_ptr<fuse::LinearizationToken>(std::move(impl));
}

}  // namespace fuse::petsc
