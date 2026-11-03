#pragma once

/// Global degree-of-freedom numbering over a fuse_physics::spatial::Mesh,
/// adapted from KARMA's karma_dof_map.h/.C (port-adapted; see
/// docs/dev/physics-kit.md).
///
/// NOT A LITERAL PORT. KARMA's own KARMADofMap keys a DoF by a MOAB
/// GLOBAL_ID or a bit-exact floating-point vertex-coordinate identity
/// (karma_dof_map.C's own geometric_vertex_identity, used only for a
/// mesh-free degenerate case this port's own Mesh has no equivalent of),
/// canonicalizes a shared sub-entity's cyclic vertex order with a
/// brute-force rotation/reflection search over ENTITY IDENTITY tuples,
/// and numbers everything in PetscInt through an ISLocalToGlobalMapping.
/// This port instead:
///   - keys a DoF by (variable, component, entity_dimension, the
///     sub-entity's own CORNER vertex GLOBAL ids, slot) -- exact
///     integers throughout (fuse_physics::spatial::Mesh::vertex_global_id/
///     element_global_id), never a coordinate bit-pattern;
///   - canonicalizes a shared sub-entity's identity by SORTING its corner
///     ids (Mesh::canonical_side_vertex_ids already does exactly this,
///     for exactly this reason -- neighbor() uses it to match a shared
///     side across two elements without caring about local order), not
///     by a rotation/reflection search; a SEPARATE, explicit orientation
///     code (this file's own canonical_orientation) answers "which local
///     order is this" for fe::Basis::apply_orientation, kept apart from
///     the sort-based IDENTITY so one answers "is this the same DoF" and
///     the other answers "how do I read its coefficients", never
///     conflated;
///   - numbers in std::int64_t, never PetscInt (physics-kit has no PETSc
///     dependency at tier 0, quadrature.hpp's own note);
///   - is collective through fuse::agree_outcome
///     when built with a communicator, not an ISLocalToGlobalMapping.
///
/// SCOPE: single-variable and multi-
/// variable numbering, contiguous per-rank ownership, the ghost index
/// list, element dof_indices (all variables, one variable), and the
/// row/column sparsity pattern for AlgebraicSystem. See dof_map.cpp's
/// own header comment for the GLOBAL NUMBERING algorithm (two
/// allgather rounds over fuse::linalg::Reducer, a stated scalability
/// limitation; ghost resolution itself is point-to-point).

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "fuse_physics/fe/basis.hpp"
#include "fuse_physics/spatial/mesh.hpp"

namespace fuse::physics::fe {

/// One field this DofMap numbers. Same fields as KARMA's own VariableSpec
/// (karma_dof_map.h), renamed to this port's own BasisKey vocabulary.
struct Variable {
  std::string name;
  FieldBasisFamily family = FieldBasisFamily::LagrangeNodal;
  unsigned order = 1;
  BasisContinuity continuity = BasisContinuity::cG;
  unsigned components = 1;

  [[nodiscard]] bool operator==(const Variable&) const = default;
};

/// A DoF's stable cross-rank identity: which variable and vector
/// component, which kind of mesh sub-entity it is pinned to
/// (entity_dimension, same vocabulary as DofAssociation), the SORTED
/// corner global ids of that sub-entity (a vertex has exactly one, in
/// corners[0]; the rest are 0 and unused -- corners.size() names how many
/// are live), and the slot distinguishing more than one DoF on the same
/// sub-entity (edge midpoint vs. edge, say, both entity_dimension 1 but
/// different slot). Two DofKeys compare equal iff every field agrees,
/// which is exactly "the same physical DoF, as seen from any element that
/// touches it" -- the identity two different ranks' own local numbering
/// passes must agree on before either can learn the other exists.
struct DofKey {
  unsigned variable = 0;
  unsigned component = 0;
  unsigned entity_dimension = 0;
  std::vector<spatial::GlobalId> corners;  // sorted ascending; size 1 for a vertex dof
  unsigned slot = 0;

  [[nodiscard]] bool operator==(const DofKey&) const = default;
  [[nodiscard]] bool operator<(const DofKey& rhs) const noexcept;
};

/// A local dof's full placement record -- KARMA's own DofStorage,
/// renamed: `index` is the GLOBAL (not local) dof number,
/// `element_local` is true for a dG/SE/interior dof that is never shared.
struct DofRecord {
  std::int64_t index;
  unsigned component;
  DofAssociation association;
  unsigned canonical_slot;
  bool element_local;
};

/// Whether a shared
/// sub-entity (an edge or a face, named by its `corners` -- the LOCAL
/// vertex handles of every one of its corner vertices, in any order)
/// is owned HERE, under the rule every sharing rank can compute from
/// `Mesh::vertex_owner`/`is_vertex_owned` alone, with no further
/// communication: the sub-entity's owner is the MINIMUM over its
/// corners' own owner ranks. Deterministic and gap-free -- unlike the
/// rule "owned here iff every corner is owned here", which
/// left a legitimate partition (a shared face whose corners are split
/// across two or more ranks) owned by NO rank at all. Exposed as a
/// free function (not kept as a DofMap-private implementation detail)
/// so it is DIRECTLY testable against a hand-built Mesh with split
/// vertex ownership, without needing the full collective numbering
/// path (which this environment cannot run under real MPI) to exercise
/// the rule itself. `corners` empty is never owned anywhere.
[[nodiscard]] bool shared_sub_entity_owned_here(const spatial::Mesh& mesh,
                                                 const std::vector<spatial::EntityHandle>& corners);

/// Global DoF numbering over `mesh`'s elements, one or more `variables`.
///
/// OWNERSHIP CONVENTION: a shared sub-entity's DoF is owned by the rank
/// that `shared_sub_entity_owned_here` selects (the MINIMUM over its
/// corner vertices' owner ranks); a
/// vertex DoF's ownership is simply that vertex's own; a dG/SE/interior
/// DoF's ownership is simply its element's own (`is_element_owned`,
/// never shared, so never ambiguous). For a mesh partitioned with a
/// one-layer ghost halo (every element a rank owns has all its OWN
/// corner vertices visible, owned either here or as a ghost) this
/// convention assigns each shared sub-entity to exactly one rank; any
/// disagreement is caught by the global dof-count cross-check below
/// rather than silently miscounted.
class DofMap {
 public:
  /// `fortran_communicator`: unset for the single-rank/tier-0 convenience
  /// path (every refusal below is a plain local throw, since there is no
  /// peer to disagree with); set to an `MPI_Comm_c2f` handle for the
  /// collective path (every refusal routes through `fuse::agree_outcome`,
  /// naming `agreed_code`
  /// "E-PHYSICS-DOFMAP-AGREEMENT-REFUSED" and comm-failure code
  /// "E-PHYSICS-DOFMAP-AGREEMENT-COMM-FAILURE").
  DofMap(const spatial::Mesh& mesh, std::vector<Variable> variables, const BasisRegistry& registry,
         std::optional<int> fortran_communicator = std::nullopt);

  [[nodiscard]] unsigned n_variables() const noexcept { return static_cast<unsigned>(variables_.size()); }
  [[nodiscard]] const Variable& variable(unsigned index) const { return variables_.at(index); }
  [[nodiscard]] std::int64_t n_dofs() const noexcept { return n_dofs_; }
  [[nodiscard]] std::int64_t n_local_dofs() const noexcept { return n_owned_; }
  [[nodiscard]] std::int64_t n_visible_dofs() const noexcept {
    return static_cast<std::int64_t>(local_to_global_.size());
  }
  [[nodiscard]] std::int64_t first_owned_dof() const noexcept { return first_owned_; }
  [[nodiscard]] std::int64_t end_owned_dof() const noexcept { return first_owned_ + n_owned_; }
  /// Every DoF this rank can see (owned, in [0, n_local_dofs()), THEN
  /// ghost), in the SAME local order `dof_indices`' own local-index
  /// overload uses. KARMA's own local_to_global_indices().
  [[nodiscard]] const std::vector<std::int64_t>& local_to_global() const noexcept { return local_to_global_; }
  [[nodiscard]] const std::vector<std::int64_t>& ghost_indices() const noexcept { return ghost_indices_; }
  /// `linalg::RowOwnership` for `DistSparseMatrix::from_triplets`:
  /// `row_lo[r]` for every rank `r`, size `reducer->size() + 1`'s worth of
  /// boundary knowledge packed as `size()` entries ascending (RowOwnership's
  /// own contract -- its header comment).
  [[nodiscard]] const std::vector<std::int64_t>& row_lo_by_rank() const noexcept { return row_lo_by_rank_; }

  /// All variables' GLOBAL dof indices for `element`: variable-major,
  /// then PER BASIS ASSOCIATION (the sub-entity order
  /// Basis::dof_associations() itself returns -- vertices in CN
  /// connectivity order for a P1 basis, then edge/face/interior dofs for
  /// a higher-order one), then component-minor (KARMA's own VariableSpec
  /// loop order kept the vector-component index innermost too,
  /// karma_dof_map.h's class comment -- this port's own basis-association
  /// level simply replaces KARMA's direct "vertex, then edge, then
  /// face..." traversal with whatever order dof_associations() reports,
  /// so a user-registered basis controls its own element ordering).
  void dof_indices(spatial::EntityHandle element, std::vector<std::int64_t>& indices) const;
  /// One variable's own GLOBAL dof indices for `element`.
  void dof_indices(spatial::EntityHandle element, std::vector<std::int64_t>& indices, unsigned variable) const;
  /// One variable's own placement records for `element` -- association,
  /// component, canonical slot, element-local flag -- parallel to the
  /// index-only overload above, for a caller that needs to apply
  /// `Basis::apply_orientation` before assembling (the orientation code
  /// this record's sub-entity presents on THIS element, from THIS
  /// element's own local vertex order, is `orientation_code_for_side`
  /// below).
  [[nodiscard]] std::vector<DofRecord> storage(spatial::EntityHandle element, unsigned variable,
                                                const Basis& basis) const;

  /// Which local-to-canonical permutation (an EdgeOrientation/
  /// TriangleOrientation/QuadrilateralOrientation value, by its
  /// underlying unsigned -- the same vocabulary Basis::apply_orientation
  /// takes) THIS element's own local vertex order presents for its
  /// (entity_dimension, local_entity_index) sub-entity, against the
  /// GLOBAL canonical cyclic order this map picked when it first saw
  /// that sub-entity (the lexicographically smallest rotation/reflection
  /// of its corner GLOBAL ids -- KARMA's own canonical_cycle, ported,
  /// since Mesh::canonical_side_vertex_ids' sorted form this class uses
  /// for IDENTITY throws the cyclic order away on purpose, dof_map.hpp's
  /// own header comment). A caller assembling a shared-DoF basis function
  /// passes this to Basis::apply_orientation before writing the local
  /// coefficient into the canonical (global) slot.
  [[nodiscard]] unsigned orientation_code(spatial::EntityHandle element, unsigned entity_dimension,
                                           unsigned local_entity_index) const;

  /// The DOF count KARMA's own `n_local_dofs`/`n_owned` kept apart from
  /// the raw mesh ownership query are the same single source of truth
  /// here: whichever sub-entity-ownership convention this class's own
  /// header comment states.
  [[nodiscard]] bool owns_element_for_assembly(spatial::EntityHandle element) const noexcept;

  /// NOT PROVIDED: KARMA's own add_element_adjacency/diagonal_nnz/
  /// off_diagonal_nnz existed to preallocate a PETSc Mat's sparsity
  /// before insertion (karma_dof_map.h).
  /// fuse::linalg::DistSparseMatrix::from_triplets infers its sparsity
  /// directly from the assembled triplet list, so this port has no
  /// equivalent method -- a stated deviation, not a
  /// silent drop: AlgebraicSystem's own assemble_matrix is the element-
  /// adjacency consumer, and it reads dof_indices()
  /// per element exactly as KARMA's own assembly loop did, it simply
  /// never needs a separate preallocation pass first.

 private:
  std::vector<Variable> variables_;
  const BasisRegistry* registry_;
  const spatial::Mesh* mesh_;
  std::int64_t n_dofs_ = 0;
  std::int64_t n_owned_ = 0;
  std::int64_t first_owned_ = 0;
  std::vector<std::int64_t> local_to_global_;  // owned entries first, then ghosts -- see local_to_global()
  std::vector<std::int64_t> ghost_indices_;
  std::vector<std::int64_t> row_lo_by_rank_;

  std::map<DofKey, std::int64_t> key_to_global_;  // every key this rank can SEE (owned + ghost)
};

}  // namespace fuse::physics::fe
