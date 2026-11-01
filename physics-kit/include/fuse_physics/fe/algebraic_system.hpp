#pragma once

/// The assembled linear algebra system over a fe::DofMap, adapted from
/// KARMA's karma_algebraic_system.h/.C (docs/dev/physics-kit-
/// disposition.md's own D-A: "port-adapt ... algebraic system").
///
/// NOT A LITERAL PORT. KARMA's own KARMAAlgebraicSystem owns a PETSc Mat
/// (preallocated from KARMADofMap's diagonal_nnz/off_diagonal_nnz) and a
/// pair of PETSc Vec, with MatSetValues/VecSetValues called directly from
/// the element loop and a single MatAssemblyEnd/VecAssemblyEnd collective
/// at the end. This port instead collects every element's contribution
/// as a plain (row, col, value) fuse::linalg::Triplet -- INCLUDING a
/// contribution to a row this rank does not own -- and, after the full
/// element loop, exchanges the foreign-row triplets to their owners in
/// ONE collective step before calling
/// fuse::linalg::DistSparseMatrix::from_triplets, exactly item 2's own
/// instruction. No second sparse-matrix type is introduced (item 2's
/// instruction): fuse::linalg::DistSparseMatrix is the sink, read in
/// full before writing this file.
///
/// SCOPE THIS WAVE (S8.3 wave 2, round 1): assemble_matrix/assemble_vector
/// over a caller-supplied per-element contribution callback, and
/// eliminate_dirichlet_rows (symmetric row elimination, tested). A PETSc
/// Mat/Vec sink behind FUSE_HAVE_PETSC is explicitly NOT in this wave
/// (item 2's own instruction) -- named as a follow-up in design_notes.

#include <cstdint>
#include <functional>
#include <optional>
#include <utility>
#include <vector>

#include "fuse/base/result.hpp"
#include "fuse/linalg/comm.hpp"
#include "fuse/linalg/dist_sparse_matrix.hpp"
#include "fuse/linalg/dist_vector.hpp"
#include "fuse_physics/fe/dof_map.hpp"

namespace fuse::physics::fe {

/// Eliminates `dirichlet_rows[i]`'s dof to the prescribed `values[i]`,
/// SYMMETRICALLY: that row becomes exactly {(row, row, 1.0)}, and every
/// OTHER row's triplet naming a dirichlet COLUMN is removed with its
/// value moved into `rhs` at that other row -- so a matrix that was
/// symmetric before elimination stays symmetric after (the column this
/// routine zeroes is removed by the SAME rule the row is, not by a
/// one-sided row-only elimination that would leave pcg's own symmetry
/// assumption violated). KARMA's own `constrain()` (karma_dof_map.h
/// lines 143-144) is a no-op -- this is NOT that, stated in
/// design_notes.
///
/// SCOPE THIS WAVE: `triplets` and `rhs` are the FULL global system (COO
/// triplets in [0, global_n), a dense `rhs` of that same size) -- the
/// single-rank/serial case every tier-0 test here exercises. Extending
/// this to the distributed case (a triplet's dirichlet COLUMN may be
/// owned by a DIFFERENT rank than the row it is being subtracted from,
/// needing a collective exchange of the subtracted amounts before each
/// rank's own `rhs` block is correct) is NOT implemented this round --
/// named as a follow-up (open_questions), not hidden; `assemble_matrix`'s
/// own multi-rank triplet exchange is unaffected (it runs before this
/// function, which refuses on a mismatched size).
///
/// Refuses: `dirichlet_rows.size() != values.size()`; a dirichlet row
/// index or any triplet's row/col outside `[0, global_n)`;
/// `rhs.size() != global_n`.
void eliminate_dirichlet_rows(std::vector<linalg::Triplet>& triplets, std::int64_t global_n,
                               const std::vector<std::int64_t>& dirichlet_rows,
                               const std::vector<double>& values, std::vector<double>& rhs);

/// Owns a DofMap and the communication objects (Reducer/GhostExchanger)
/// its assembly needs. `fortran_communicator` unset selects the
/// single-rank convenience path (fuse::linalg::make_serial_reducer(),
/// matching DofMap's own convention); set, the collective path (built,
/// never run in this sandbox -- host_pending names the MPI build).
class AlgebraicSystem {
 public:
  /// `mesh` is the SAME mesh `dof_map` was built over (both outlive this
  /// object, the same lifetime contract DofMap's own constructor takes) --
  /// DofMap's own public surface names no element COUNT or range (only
  /// per-element queries), so AlgebraicSystem's element loop reads
  /// `mesh.n_elements()` directly rather than adding one to DofMap's
  /// already-tested surface for this file's sole use.
  AlgebraicSystem(const spatial::Mesh& mesh, const DofMap& dof_map,
                   std::optional<int> fortran_communicator = std::nullopt);

  [[nodiscard]] const DofMap& dof_map() const noexcept { return *dof_map_; }
  [[nodiscard]] std::shared_ptr<const linalg::Reducer> reducer() const noexcept { return reducer_; }

  /// A caller's per-OWNED-element contribution: given `element` and its
  /// own GLOBAL dof_indices (dof_map().dof_indices(element, ...), already
  /// computed so every contribution callback shares the same lookup),
  /// append zero or more (row, col, value) triplets to `out` -- rows and
  /// columns need not be rows this rank owns; `assemble_matrix` below
  /// exchanges foreign rows to their owners in ONE collective step
  /// after the whole element loop (item 2's own instruction).
  using ElementContribution = std::function<void(spatial::EntityHandle element,
                                                   const std::vector<std::int64_t>& dof_indices,
                                                   std::vector<linalg::Triplet>& out)>;

  /// Loops every element `dof_map().owns_element_for_assembly()` names,
  /// calls `contribution`, exchanges foreign-row triplets to their
  /// owners (a stated ROUND-1 SIMPLIFICATION -- see the .cpp's own
  /// header comment: every rank's own foreign-triplet buffer is
  /// broadcast to every other rank via the same padded-allgather
  /// technique DofMap's own ghost resolution uses, O(total triplet
  /// count) communication rather than a point-to-point exchange sized
  /// to what each rank actually needs; named as a follow-up, not
  /// hidden), merges the result with its own already-owned triplets,
  /// and calls `fuse::linalg::DistSparseMatrix::from_triplets`.
  [[nodiscard]] Result<linalg::DistSparseMatrix> assemble_matrix(const ElementContribution& contribution) const;

  /// A caller's per-OWNED-element vector contribution: (dof_index, value)
  /// pairs, possibly naming a dof this rank does not own (exchanged the
  /// same way `assemble_matrix` exchanges foreign rows).
  using ElementVectorContribution = std::function<void(
      spatial::EntityHandle element, const std::vector<std::int64_t>& dof_indices,
      std::vector<std::pair<std::int64_t, double>>& out)>;

  /// A zero DistVector over this system's own row distribution and ghost
  /// pattern, ready for `assemble_vector` or `DistSparseMatrix::apply`.
  [[nodiscard]] linalg::DistVector new_vector() const;

  [[nodiscard]] Result<void> assemble_vector(const ElementVectorContribution& contribution,
                                              linalg::DistVector& out) const;

 private:
  const spatial::Mesh* mesh_;
  const DofMap* dof_map_;
  std::shared_ptr<const linalg::Reducer> reducer_;
  std::shared_ptr<const linalg::GhostExchanger> exchanger_;
};

}  // namespace fuse::physics::fe
