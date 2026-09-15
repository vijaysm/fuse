#ifndef KARMA_COUPLING_FIELD_H
#define KARMA_COUPLING_FIELD_H

#include "karma_dof_map.h"
#include "karma_field_identity.h"
#include "karma_mesh.h"
#include "karma_system.h"

#include <mpi.h>
#include <petscvec.h>
#include <moab/Types.hpp>

#include <span>
#include <string>
#include <variant>
#include <vector>

namespace KARMA
{
enum class FieldLocation
{
    Vertex,
    Element,
    FiniteElementCoefficient
};

enum class FieldScalarRepresentation
{
    NativeReal,
    PetscScalar
};

struct MoabTagBinding
{
    KARMAMesh* mesh = nullptr;
    moab::Tag tag = nullptr;
    FieldLocation location = FieldLocation::Vertex;
    unsigned components = 0;
};

struct PetscFieldBinding
{
    KARMASystem* system = nullptr;
    Vec vector = nullptr;
    unsigned variable = 0;
    VariableSpec specification;
    FieldScalarRepresentation representation = FieldScalarRepresentation::PetscScalar;
};

using FieldStorageBinding = std::variant< MoabTagBinding, PetscFieldBinding >;

// FieldEntryKey now lives in karma_field_identity.h (dependency-free); included above.

class KARMAField
{
  public:
    explicit KARMAField( MoabTagBinding binding );
    explicit KARMAField( PetscFieldBinding binding );

    [[nodiscard]] const FieldStorageBinding& binding() const noexcept { return _binding; }
    [[nodiscard]] FieldLocation location() const noexcept { return _location; }
    [[nodiscard]] unsigned components() const noexcept { return _components; }
    [[nodiscard]] FieldScalarRepresentation scalar_representation() const noexcept
    { return _representation; }
    [[nodiscard]] MPI_Comm communicator() const noexcept { return _communicator; }
    [[nodiscard]] const std::vector< FieldEntryKey >& entries() const noexcept { return _entries; }

    void validate_values() const;
    void commit_petsc_values(std::span<const std::pair<DofKey, Real>> values) const;
    void commit_moab_values(std::span<const std::pair<FieldEntryKey, Real>> values) const;

  private:
    void initialize_moab( const MoabTagBinding& binding );
    void initialize_petsc( const PetscFieldBinding& binding );

    FieldStorageBinding _binding;
    FieldLocation _location = FieldLocation::Vertex;
    FieldScalarRepresentation _representation = FieldScalarRepresentation::NativeReal;
    unsigned _components = 0;
    MPI_Comm _communicator = MPI_COMM_NULL;
    std::vector< FieldEntryKey > _entries;
};
}  // namespace KARMA

#endif
