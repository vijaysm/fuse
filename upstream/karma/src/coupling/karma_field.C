#include "karma_field.h"

#include "karma_exceptions.h"
#include "karma_petsc_field_storage.h"
#include "petsc_raii.h"

#include <algorithm>
#include <cmath>
#include <string>

namespace KARMA
{
namespace
{
void require(bool condition, const char* message)
{
    if (!condition) KARMA_THROW(InvalidArgument(message));
}

MPI_Comm mesh_communicator(const KARMAMesh& mesh)
{
    return mesh.communicator();
}
}

KARMAField::KARMAField(MoabTagBinding binding) : _binding(binding)
{
    initialize_moab(binding);
}

KARMAField::KARMAField(PetscFieldBinding binding) : _binding(binding)
{
    initialize_petsc(binding);
}

void KARMAField::initialize_moab(const MoabTagBinding& binding)
{
    require(binding.mesh != nullptr, "MOAB field binding requires a mesh");
    require(binding.tag != nullptr, "MOAB field binding requires a tag");
    require(binding.components != 0, "field requires at least one component");
    require(binding.location != FieldLocation::FiniteElementCoefficient,
            "MOAB tags cannot bind as finite-element coefficients");
    _location = binding.location;
    _components = binding.components;
    _communicator = mesh_communicator(*binding.mesh);

    int length = 0;
    moab::DataType type = moab::MB_TYPE_OPAQUE;
    require(binding.mesh->moab()->tag_get_length(binding.tag, length) == moab::MB_SUCCESS,
            "failed to query MOAB field tag length");
    require(binding.mesh->moab()->tag_get_data_type(binding.tag, type) == moab::MB_SUCCESS,
            "failed to query MOAB field tag type");
    require(type == moab::MB_TYPE_DOUBLE && length == static_cast<int>(binding.components),
            "MOAB field tag must be a dense real component tuple");

    const moab::Range entities = binding.location == FieldLocation::Vertex ? binding.mesh->vertices()
                                                                             : binding.mesh->elements();
    moab::Tag global_id = nullptr;
    require(binding.mesh->moab()->tag_get_handle("GLOBAL_ID", global_id) == moab::MB_SUCCESS,
            "MOAB field requires conventional GLOBAL_ID");
    for (const auto entity : entities)
    {
        int id = 0;
        require(binding.mesh->moab()->tag_get_data(global_id, &entity, 1, &id) == moab::MB_SUCCESS && id > 0,
                "MOAB field requires positive GLOBAL_ID values");
        for (unsigned component = 0; component < binding.components; ++component)
            _entries.push_back({FieldEntryKey::Kind::MoabGlobalId, static_cast<std::uint64_t>(id), component, {}});
    }
}

void KARMAField::initialize_petsc(const PetscFieldBinding& binding)
{
    require(binding.system != nullptr, "PETSc field binding requires a system");
    require(binding.vector != nullptr, "PETSc field binding requires a vector");
    require(binding.variable < binding.system->n_variables(), "PETSc field variable is out of range");
    require(binding.specification.components != 0, "field requires at least one component");
    require(binding.specification.name == binding.system->variable(binding.variable).name,
            "PETSc field variable specification does not match the system");
    _location = FieldLocation::FiniteElementCoefficient;
    _components = binding.specification.components;
    _representation = binding.representation;
    _communicator = binding.system->communicator();
    binding.system->validate_vector_layout(binding.vector, "field values");
    for (const auto& record : binding.system->dof_map().owned_key_indices())
    {
        if (record.first[0] != binding.variable) continue;
        _entries.push_back({FieldEntryKey::Kind::Dof, 0, static_cast<unsigned>(record.first[1]), record.first});
    }
}

void KARMAField::validate_values() const
{
    if (std::holds_alternative<PetscFieldBinding>(_binding))
    {
        const auto& binding = std::get<PetscFieldBinding>(_binding);
        binding.system->validate_vector_layout(binding.vector, "field values");
        return;
    }
    const auto& binding = std::get<MoabTagBinding>(_binding);
    const auto entities = binding.location == FieldLocation::Vertex ? binding.mesh->vertices() : binding.mesh->elements();
    std::vector<double> values(static_cast<std::size_t>(entities.size()) * binding.components);
    if (binding.mesh->moab()->tag_get_data(binding.tag, entities, values.data()) != moab::MB_SUCCESS)
        KARMA_THROW(InvalidArgument("failed to read MOAB field values"));
    for (const double value : values)
        if (!std::isfinite(value)) KARMA_THROW(InvalidArgument("field contains a nonfinite value"));
}

void KARMAField::commit_moab_values(std::span<const std::pair<FieldEntryKey, Real>> values) const
{
    if (_location == FieldLocation::FiniteElementCoefficient)
        KARMA_THROW(InvalidArgument("MOAB commit requires a MOAB field"));
    const auto& binding = std::get<MoabTagBinding>(_binding);
    if (values.size() != _entries.size()) KARMA_THROW(InvalidArgument("MOAB commit must provide every field entry"));
    std::vector<double> staged(static_cast<std::size_t>(values.size()), 0.0);
    for (std::size_t index = 0; index < values.size(); ++index)
    {
        if (values[index].first != _entries[index] || !std::isfinite(values[index].second))
            KARMA_THROW(InvalidArgument("MOAB commit keys or values are invalid"));
        staged[index] = values[index].second;
    }
    const auto entities = _location == FieldLocation::Vertex ? binding.mesh->vertices() : binding.mesh->elements();
    if (binding.mesh->moab()->tag_set_data(binding.tag, entities, staged.data()) != moab::MB_SUCCESS)
        KARMA_THROW(InvalidArgument("failed to publish MOAB field commit"));
}

void KARMAField::commit_petsc_values(std::span<const std::pair<DofKey, Real>> values) const
{
    if (_location != FieldLocation::FiniteElementCoefficient)
        KARMA_THROW(InvalidArgument("PETSc commit requires finite-element field"));
    const auto& binding = std::get<PetscFieldBinding>(_binding);
    binding.system->validate_vector_layout(binding.vector, "field commit target");

    // Resolution and the per-value checks stay exactly as they were: an
    // unknown key or a nonfinite input throws here, with the same message,
    // before any transaction exists.
    std::vector<PetscInt> indices;
    std::vector<Real> staged;
    indices.reserve(values.size());
    staged.reserve(values.size());
    for (const auto& [key, value] : values)
    {
        if (!std::isfinite(value)) KARMA_THROW(InvalidArgument("field commit contains a nonfinite value"));
        if (key[0] != binding.variable) KARMA_THROW(InvalidArgument("field commit variable differs from binding"));
        indices.push_back(binding.system->dof_map().global_index(key));
        staged.push_back(value);
    }

    // Beyond this point publication goes through the one transactional path
    // every other field write uses. PetscVectorStorage restricts itself to
    // locally owned entries - see karma_petsc_field_storage.h for why an
    // off-process write does not belong behind this interface at all.
    PetscVectorStorage storage(binding.vector, indices);
    FieldTransaction transaction(storage, FieldViewLayout{1, indices.size(), 0}, UpdateMode::Replace);
    for (std::size_t entry = 0; entry != staged.size(); ++entry)
    {
        const auto reason = transaction.stage(entry, 0, staged[entry]);
        if (reason != OutcomeReason::Success) KARMA_THROW(InvalidArgument("field commit staging was refused"));
    }
    const auto outcome = transaction.commit();
    if (!outcome.ok())
        KARMA_THROW(InvalidArgument(outcome.reason == OutcomeReason::InvalidArgument
                                        ? "field commit addresses an entry this rank does not own"
                                        : "field commit publication was refused"));
}
}  // namespace KARMA
