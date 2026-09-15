// KARMA: K(C)ode for Analysis of Reactor and other Multiphysics Applications
#ifndef __karma_dof_map_h__
#define __karma_dof_map_h__

#include "karma_basis.h"
#include "karma_dense.h"
#include "karma_field_identity.h"
#include "karma_mesh.h"

#include <petscis.h>
#include <petscsys.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace KARMA
{
namespace detail
{
struct MpiLayout
{
    std::vector< int > displacements;
    std::size_t total_count = 0;
};

[[nodiscard]] int checked_mpi_count( std::size_t count );
[[nodiscard]] int checked_mpi_word_count( std::size_t count, std::size_t words_per_record );
[[nodiscard]] MpiLayout checked_mpi_displacements( std::span< const int > counts, std::size_t words_per_record );
[[nodiscard]] PetscInt checked_petsc_int_size( std::size_t size );
[[nodiscard]] bool collective_validation_succeeded( MPI_Comm communicator, bool local_valid );
}  // namespace detail

struct VariableSpec
{
    std::string name;
    FieldBasisFamily family    = FieldBasisFamily::LagrangeNodal;
    unsigned order             = 1;
    BasisContinuity continuity = BasisContinuity::cG;
    unsigned components        = 1;
};

struct CanonicalSubentity
{
    std::vector< PetscInt > vertex_ids;
    unsigned orientation_code;
};
[[nodiscard]] CanonicalSubentity canonicalize_cyclic_vertex_ids( std::span< const PetscInt > cyclic_vertex_ids );
[[nodiscard]] unsigned canonical_orientation_code( std::span< const PetscInt > cyclic_vertex_ids );

struct DofOrientation
{
    DofAssociation association;
    unsigned orientation_code;
    std::vector< unsigned > local_to_canonical_vertices;
};

struct DofStorage
{
    PetscInt index;
    unsigned component;
    DofAssociation association;
    unsigned canonical_slot;
    bool element_local;
};

/** Stable DoF identity serialized as:
 * variable, component, entity dimension, MOAB entity type, canonical slot,
 * identity count, then three words per identity. Conventional meshes use
 * positive MOAB GLOBAL_ID values in the first identity word; unused words are
 * zero. Intermediate cG entities use canonically ordered vertex identities.
 */
// DofKey now lives in karma_field_identity.h (dependency-free); this header includes it above.

/** Algebraic DoF layout independent of DMMoab numbering.
 *
 * Element indices are variable-major, then component-major, then basis-minor.
 * Shared cG keys represent virtual mesh subentities; dG and SE keys are always
 * element-local. All public and internal algebraic indices use PetscInt.
 */
class KARMADofMap
{
  public:
    using index_type = PetscInt;

    KARMADofMap();
    KARMADofMap( const KARMAMesh& mesh, const std::vector< VariableSpec >& variables,
                 const KARMABasisRegistry& registry );
    ~KARMADofMap();
    KARMADofMap( const KARMADofMap& )            = delete;
    KARMADofMap& operator=( const KARMADofMap& ) = delete;
    KARMADofMap( KARMADofMap&& other ) noexcept;
    KARMADofMap& operator=( KARMADofMap&& other ) noexcept;

    [[nodiscard]] unsigned n_variables() const noexcept
    { return static_cast< unsigned >( _variables.size() ); }
    [[nodiscard]] PetscInt n_dofs() const noexcept
    { return _n_dofs; }
    [[nodiscard]] PetscInt n_local_dofs() const noexcept
    { return _n_owned; }
    [[nodiscard]] PetscInt n_visible_dofs() const noexcept
    { return static_cast< PetscInt >( _local_to_global.size() ); }
    [[nodiscard]] PetscInt first_owned_dof() const noexcept;
    [[nodiscard]] PetscInt end_owned_dof() const noexcept
    { return first_owned_dof() + _n_owned; }
    [[nodiscard]] const std::vector< PetscInt >& local_to_global_indices() const noexcept
    { return _local_to_global; }
    [[nodiscard]] ISLocalToGlobalMapping local_to_global_mapping() const noexcept
    { return _mapping; }
    [[nodiscard]] const std::vector< PetscInt >& ghost_indices() const noexcept
    { return _ghosts; }
    /** Exact global sparsity counts for locally owned rows. */
    [[nodiscard]] const std::vector< PetscInt >& diagonal_nnz() const noexcept
    { return _diagonal_nnz; }
    [[nodiscard]] const std::vector< PetscInt >& off_diagonal_nnz() const noexcept
    { return _off_diagonal_nnz; }
    [[nodiscard]] const std::vector< moab::EntityHandle >& assembly_elements() const noexcept
    { return _assembly_elements; }
    [[nodiscard]] bool owns_element_for_assembly( moab::EntityHandle element ) const noexcept;
    /** Rank-independent ordinal in replicated mode; conventional GLOBAL_ID otherwise. */
    [[nodiscard]] std::uint64_t canonical_element_ordinal( moab::EntityHandle element ) const;

    void dof_indices( moab::EntityHandle element, std::vector< PetscInt >& indices ) const;
    void dof_indices( moab::EntityHandle element, std::vector< PetscInt >& indices, unsigned variable ) const;
    [[nodiscard]] const std::vector< DofKey >& dof_keys( moab::EntityHandle element ) const;
    [[nodiscard]] PetscInt global_index( const DofKey& key ) const;
    [[nodiscard]] std::pair< DofKey, PetscInt > owned_key_index( std::size_t ordinal ) const;
    [[nodiscard]] std::vector< std::pair< DofKey, PetscInt > > owned_key_indices() const;
    void local_dof_indices( moab::EntityHandle element, std::vector< PetscInt >& indices ) const;
    [[nodiscard]] PetscInt vertex_dof( moab::EntityHandle vertex, unsigned variable, unsigned component = 0 ) const;
    [[nodiscard]] std::vector< DofStorage > storage( moab::EntityHandle element, unsigned variable,
                                                     const KARMABasis& basis ) const;
    [[nodiscard]] std::vector< DofOrientation > orientations( moab::EntityHandle element, unsigned variable ) const;

    /** Collective: ranks may submit different locally visible pairs, or (0,0). */
    void add_element_adjacency( moab::EntityHandle first, moab::EntityHandle second );

    void constrain( RealDenseMatrix&, RealDenseVector&, const std::vector< PetscInt >& ) const noexcept {}
    void constrain( RealDenseVector&, const std::vector< PetscInt >& ) const noexcept {}
    [[nodiscard]] bool has_constraints() const noexcept
    { return false; }

  private:
    struct Impl;
    std::unique_ptr< Impl > _impl;
    std::vector< VariableSpec > _variables;
    PetscInt _n_dofs  = 0;
    PetscInt _n_owned = 0;
    std::vector< PetscInt > _local_to_global;
    std::vector< PetscInt > _ghosts;
    std::vector< moab::EntityHandle > _assembly_elements;
    std::vector< PetscInt > _diagonal_nnz;
    std::vector< PetscInt > _off_diagonal_nnz;
    ISLocalToGlobalMapping _mapping = nullptr;
};
}  // namespace KARMA
#endif
