#include "karma_collective_outcome.h"
#include "karma_fem_field_evaluator.h"

#include "karma_exceptions.h"
#include "petsc_raii.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace KARMA
{
namespace
{
/** Reconcile a rank-local failure across `communicator` and throw the agreed
 * message, or return when every rank succeeded.
 *
 * The reduction rule lives in `reconcile_failure_message()`, which this file
 * used to carry its own copy of.
 * This file's previous copy reported an MPI failure as "MPI error while reconciling FEM evaluation failure";
 * the shared primitive reports it uniformly instead.
 */
void reconcile_failure(MPI_Comm communicator, bool failed, const std::string& local_message)
{
    const auto message = reconcile_failure_message(communicator, failed, local_message);
    if (!message.empty()) KARMA_THROW(InvalidArgument(message));
}

std::optional<moab::EntityHandle> local_element_by_global_id(const KARMAMesh& mesh, std::uint64_t global_id)
{
    for (const auto element : mesh.elements())
        if (static_cast<std::uint64_t>(mesh.global_id(element)) == global_id) return element;
    return std::nullopt;
}
}

KARMAFEMFieldEvaluator::KARMAFEMFieldEvaluator(const KARMAField& field, FEMFieldEvaluatorOptions options)
    : _field(field), _options(options)
{
    if (_field.location() != FieldLocation::FiniteElementCoefficient)
        KARMA_THROW(InvalidArgument("FEM field evaluator requires finite-element coefficients"));
    const auto& binding = std::get<PetscFieldBinding>(_field.binding());
    binding.system->validate_vector_layout(binding.vector, "FEM source field");
    if (VecGhostUpdateBegin(binding.vector, INSERT_VALUES, SCATTER_FORWARD) != PETSC_SUCCESS ||
        VecGhostUpdateEnd(binding.vector, INSERT_VALUES, SCATTER_FORWARD) != PETSC_SUCCESS)
        KARMA_THROW(InvalidArgument("failed to refresh source FEM ghost values"));
}

namespace
{
struct EvaluationRequest
{
    std::int32_t origin = -1;
    std::int32_t index = -1;
    std::uint64_t element_id = 0;
    double reference[3]{};
};
struct EvaluationResponse
{
    std::int32_t origin = -1;
    std::int32_t index = -1;
    std::int32_t status = 0;
    std::uint32_t components = 0;
};
}

std::vector<std::vector<Real>> KARMAFEMFieldEvaluator::evaluate_batch(std::span<const LocatedPoint> points) const
{
    const auto& binding = std::get<PetscFieldBinding>(_field.binding());
    const MPI_Comm communicator = binding.system->communicator();
    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(communicator, &rank);
    MPI_Comm_size(communicator, &ranks);
    std::vector<std::vector<Real>> results(points.size());
    std::vector<EvaluationRequest> local_requests;
    std::vector<std::vector<std::byte>> outbound(static_cast<std::size_t>(ranks));
    bool preparation_failed = false;
    std::string preparation_error;
    try
    {
        for (std::size_t index = 0; index < points.size(); ++index)
        {
            if (points[index].status != PointLocationStatus::Located)
                KARMA_THROW(InvalidArgument("cannot evaluate an unlocated source point"));
            if (points[index].source_handle != 0)
                results[index] = evaluate(points[index]);
            else
            {
                const int owner = points[index].source_rank;
                if (owner < 0 || owner >= ranks) KARMA_THROW(InvalidArgument("remote source rank is invalid"));
                EvaluationRequest request;
                request.origin = rank;
                request.index = static_cast<std::int32_t>(index);
                request.element_id = points[index].source_element.global_id;
                for (unsigned dimension = 0; dimension < 3; ++dimension)
                    request.reference[dimension] = points[index].reference_point(static_cast<Eigen::Index>(dimension));
                auto& bytes = outbound[static_cast<std::size_t>(owner)];
                const auto old_size = bytes.size();
                bytes.resize(old_size + sizeof(request));
                std::memcpy(bytes.data() + old_size, &request, sizeof(request));
            }
        }
        if (ranks == 1 && !local_requests.empty())
            KARMA_THROW(InvalidArgument("invalid local remote evaluation request"));
    }
    catch (const std::exception& error) { preparation_failed = true; preparation_error = error.what(); }
    catch (...) { preparation_failed = true; preparation_error = "unknown FEM evaluation preparation failure"; }
    reconcile_failure(communicator, preparation_failed, preparation_error);
    std::vector<int> send_counts(static_cast<std::size_t>(ranks)), send_displacements(static_cast<std::size_t>(ranks));
    std::vector<std::byte> send_buffer;
    for (int destination = 0; destination < ranks; ++destination)
    {
        send_displacements[static_cast<std::size_t>(destination)] = static_cast<int>(send_buffer.size());
        send_counts[static_cast<std::size_t>(destination)] = static_cast<int>(outbound[static_cast<std::size_t>(destination)].size());
        send_buffer.insert(send_buffer.end(), outbound[static_cast<std::size_t>(destination)].begin(),
                           outbound[static_cast<std::size_t>(destination)].end());
    }
    std::vector<int> recv_counts(static_cast<std::size_t>(ranks));
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, communicator);
    std::vector<int> recv_displacements(static_cast<std::size_t>(ranks));
    int recv_size = 0;
    for (int source = 0; source < ranks; ++source)
    {
        recv_displacements[static_cast<std::size_t>(source)] = recv_size;
        recv_size += recv_counts[static_cast<std::size_t>(source)];
    }
    std::vector<std::byte> recv_buffer(static_cast<std::size_t>(recv_size));
    MPI_Alltoallv(send_buffer.data(), send_counts.data(), send_displacements.data(), MPI_BYTE, recv_buffer.data(),
                  recv_counts.data(), recv_displacements.data(), MPI_BYTE, communicator);

    std::vector<std::vector<std::byte>> response_outbound(static_cast<std::size_t>(ranks));
    bool request_failed = false;
    std::string request_error;
    for (int source = 0; source < ranks; ++source)
    {
        const auto begin = static_cast<std::size_t>(recv_displacements[static_cast<std::size_t>(source)]);
        const auto end = begin + static_cast<std::size_t>(recv_counts[static_cast<std::size_t>(source)]);
        for (std::size_t offset = begin; offset < end; offset += sizeof(EvaluationRequest))
        {
            if (end - offset < sizeof(EvaluationRequest))
            {
                request_failed = true;
                request_error = "malformed FEM evaluation request";
                break;
            }
            EvaluationRequest request;
            std::memcpy(&request, recv_buffer.data() + offset, sizeof(request));
            LocatedPoint point;
            point.status = PointLocationStatus::Located;
            point.source_element = {FieldEntryKey::Kind::MoabGlobalId, request.element_id, 0, {}};
            point.reference_point = Point(request.reference[0], request.reference[1], request.reference[2]);
            const auto element = local_element_by_global_id(binding.system->mesh(), request.element_id);
            EvaluationResponse response;
            response.origin = request.origin;
            response.index = request.index;
            response.components = _field.components();
            std::vector<Real> values;
            if (!element)
            {
                response.status = 1;
                response.components = 0;
            }
            else
            {
                point.source_handle = *element;
                try { values = evaluate(point); }
                catch (...) { response.status = 1; response.components = 0; }
            }
            auto& bytes = response_outbound[static_cast<std::size_t>(request.origin)];
            const auto old_size = bytes.size();
            bytes.resize(old_size + sizeof(response) + values.size() * sizeof(Real));
            std::memcpy(bytes.data() + old_size, &response, sizeof(response));
            if (!values.empty()) std::memcpy(bytes.data() + old_size + sizeof(response), values.data(), values.size() * sizeof(Real));
        }
    }
    reconcile_failure(communicator, request_failed, request_error);
    send_buffer.clear();
    for (int destination = 0; destination < ranks; ++destination)
    {
        send_displacements[static_cast<std::size_t>(destination)] = static_cast<int>(send_buffer.size());
        send_counts[static_cast<std::size_t>(destination)] = static_cast<int>(response_outbound[static_cast<std::size_t>(destination)].size());
        send_buffer.insert(send_buffer.end(), response_outbound[static_cast<std::size_t>(destination)].begin(),
                           response_outbound[static_cast<std::size_t>(destination)].end());
    }
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, communicator);
    recv_size = 0;
    for (int source = 0; source < ranks; ++source)
    {
        recv_displacements[static_cast<std::size_t>(source)] = recv_size;
        recv_size += recv_counts[static_cast<std::size_t>(source)];
    }
    recv_buffer.assign(static_cast<std::size_t>(recv_size), std::byte{});
    MPI_Alltoallv(send_buffer.data(), send_counts.data(), send_displacements.data(), MPI_BYTE, recv_buffer.data(),
                  recv_counts.data(), recv_displacements.data(), MPI_BYTE, communicator);
    bool response_failed = false;
    std::string response_error;
    for (int source = 0; source < ranks; ++source)
    {
        const auto begin = static_cast<std::size_t>(recv_displacements[static_cast<std::size_t>(source)]);
        const auto end = begin + static_cast<std::size_t>(recv_counts[static_cast<std::size_t>(source)]);
        for (std::size_t offset = begin; offset < end; )
        {
            if (end - offset < sizeof(EvaluationResponse))
            {
                response_failed = true;
                response_error = "malformed FEM evaluation response";
                break;
            }
            EvaluationResponse response;
            std::memcpy(&response, recv_buffer.data() + offset, sizeof(response));
            offset += sizeof(response);
            const auto value_bytes = static_cast<std::size_t>(response.components) * sizeof(Real);
            if (end - offset < value_bytes || response.index < 0 || static_cast<std::size_t>(response.index) >= results.size())
            {
                response_failed = true;
                response_error = "invalid FEM evaluation response";
                break;
            }
            if (response.status != 0)
            {
                response_failed = true;
                response_error = "owner rank could not evaluate source element";
                offset += value_bytes;
                continue;
            }
            results[static_cast<std::size_t>(response.index)].resize(response.components);
            std::memcpy(results[static_cast<std::size_t>(response.index)].data(), recv_buffer.data() + offset, value_bytes);
            offset += value_bytes;
        }
    }
    reconcile_failure(communicator, response_failed, response_error);
    return results;
}

std::vector<Real> KARMAFEMFieldEvaluator::evaluate(const LocatedPoint& point) const
{
    if (point.status != PointLocationStatus::Located)
        KARMA_THROW(InvalidArgument("cannot evaluate an unlocated source point"));
    const auto& binding = std::get<PetscFieldBinding>(_field.binding());
    const auto& map = binding.system->dof_map();
    const auto source_handle = point.source_handle != 0
                                   ? std::optional<moab::EntityHandle>(point.source_handle)
                                   : local_element_by_global_id(binding.system->mesh(), point.source_element.global_id);
    if (!source_handle) KARMA_THROW(InvalidArgument("source element is remote and has no local ghost handle"));
    const moab::EntityHandle element = *source_handle;
    const moab::EntityHandle* connectivity = nullptr;
    int connectivity_count = 0;
    if (binding.system->mesh().moab()->get_connectivity(element, connectivity, connectivity_count) !=
        moab::MB_SUCCESS)
        KARMA_THROW(InvalidArgument("failed to read source element connectivity"));
    const auto variant = KARMAReferenceTopology::from_moab(
        binding.system->mesh().moab()->type_from_handle(element),
        static_cast<std::size_t>(connectivity_count));
    const auto cell = reference_cell(variant);
    const auto basis = binding.system->basis_registry().create(
        {cell, binding.specification.family, binding.specification.order, binding.specification.continuity});
    const auto storage = map.storage(element, binding.variable, basis);
    std::vector<Real> coefficients(storage.size(), 0.0);
    PetscVecLocalForm local_form(binding.vector);
    PetscVecArrayRead values(local_form.get());
    const auto& local_to_global = map.local_to_global_indices();
    std::map<PetscInt, PetscInt> local_indices;
    for (std::size_t index = 0; index < local_to_global.size(); ++index)
        local_indices.emplace(local_to_global[index], static_cast<PetscInt>(index));
    for (std::size_t index = 0; index < storage.size(); ++index)
    {
        const auto found = local_indices.find(storage[index].index);
        if (found == local_indices.end()) KARMA_THROW(InvalidArgument("source DoF is not locally available"));
        coefficients[index] = PetscRealPart(values.get()[found->second]);
        if (_options.reject_nonfinite && !std::isfinite(coefficients[index]))
            KARMA_THROW(InvalidArgument("source FEM coefficient is nonfinite"));
    }
    const auto local_storage = map.storage(element, binding.variable, basis);
    std::vector<Real> evaluation_coefficients(coefficients);
    for (const auto& orientation : map.orientations(element, binding.variable))
    {
        for (unsigned component = 0; component < _field.components(); ++component)
        {
            std::vector<std::size_t> positions;
            for (std::size_t index = 0; index < local_storage.size(); ++index)
                if (local_storage[index].component == component &&
                    local_storage[index].association.entity_dimension == orientation.association.entity_dimension &&
                    local_storage[index].association.local_entity_index == orientation.association.local_entity_index)
                    positions.push_back(index);
            if (positions.empty()) continue;
            std::vector<Real> canonical(positions.size());
            std::vector<Real> local(positions.size());
            for (std::size_t position = 0; position < positions.size(); ++position)
                canonical[std::min(position, static_cast<std::size_t>(local_storage[positions[position]].canonical_slot))] =
                    coefficients[positions[position]];
            basis.apply_orientation_inverse(orientation.association.entity_dimension,
                                            orientation.association.local_entity_index, orientation.orientation_code,
                                            canonical, local);
            for (std::size_t position = 0; position < positions.size(); ++position)
                evaluation_coefficients[positions[position]] = local[position];
        }
    }
    std::vector<Real> basis_values(basis.size());
    basis.evaluate_values(point.reference_point, basis_values);
    std::vector<Real> result(_field.components(), 0.0);
    for (unsigned component = 0; component < _field.components(); ++component)
        for (std::size_t ordinal = 0; ordinal < basis.size(); ++ordinal)
            result[component] += evaluation_coefficients[component * basis.size() + ordinal] * basis_values[ordinal];
    return result;
}
}  // namespace KARMA
