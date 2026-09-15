#include "karma_operator_split_checkpoint.h"

#include "../physics/common/impls/collective_signature.h"
#include "karma_exceptions.h"
#include "karma_moab_io.h"
#include "karma_transient_reaction_diffusion_adapter.h"

#include <mpi.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace KARMA
{
namespace
{
constexpr std::array<unsigned char, 8> magic{'K', 'A', 'R', 'M', 'A', 'O', 'S', 'R'};
constexpr std::uint32_t metadata_version = 1;
constexpr std::uint64_t maximum_metadata_size = 1024 * 1024;

struct Metadata
{
    std::string signature;
    SplitScheme scheme = SplitScheme::Lie;
    SchedulerPosition position;
    std::uint64_t u_hash = 0;
    std::uint64_t v_hash = 0;
};

void append(std::vector<unsigned char>& bytes, std::uint64_t value)
{
    for (unsigned shift = 0; shift < 64; shift += 8)
        bytes.push_back(static_cast<unsigned char>((value >> shift) & 0xffU));
}

std::uint64_t take(std::span<const unsigned char> bytes, std::size_t& offset)
{
    if (bytes.size() - offset < 8) throw std::invalid_argument("restart metadata is truncated");
    std::uint64_t value = 0;
    for (unsigned shift = 0; shift < 64; shift += 8)
        value |= static_cast<std::uint64_t>(bytes[offset++]) << shift;
    return value;
}

std::vector<unsigned char> encode(const Metadata& metadata)
{
    std::vector<unsigned char> bytes(magic.begin(), magic.end());
    append(bytes, metadata_version);
    append(bytes, static_cast<std::uint64_t>(metadata.scheme));
    append(bytes, std::bit_cast<std::uint64_t>(static_cast<double>(metadata.position.time)));
    append(bytes, metadata.position.step);
    append(bytes, metadata.u_hash);
    append(bytes, metadata.v_hash);
    append(bytes, metadata.signature.size());
    bytes.insert(bytes.end(), metadata.signature.begin(), metadata.signature.end());
    return bytes;
}

Metadata decode(std::span<const unsigned char> bytes)
{
    if (bytes.size() < magic.size() || !std::equal(magic.begin(), magic.end(), bytes.begin()))
        throw std::invalid_argument("restart metadata magic is incompatible");
    std::size_t offset = magic.size();
    if (take(bytes, offset) != metadata_version)
        throw std::invalid_argument("restart metadata version is incompatible");
    const auto scheme = take(bytes, offset);
    if (scheme > static_cast<std::uint64_t>(SplitScheme::Strang))
        throw std::invalid_argument("restart metadata split scheme is invalid");
    Metadata result;
    result.scheme = static_cast<SplitScheme>(scheme);
    result.position.time = std::bit_cast<double>(take(bytes, offset));
    const auto step = take(bytes, offset);
    if (step > std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument("restart metadata step is too large");
    result.position.step = static_cast<std::size_t>(step);
    result.u_hash = take(bytes, offset);
    result.v_hash = take(bytes, offset);
    const auto signature_size = take(bytes, offset);
    if (signature_size > bytes.size() - offset || offset + signature_size != bytes.size())
        throw std::invalid_argument("restart metadata signature is truncated");
    result.signature.assign(reinterpret_cast<const char*>(bytes.data() + offset),
                            static_cast<std::size_t>(signature_size));
    if (!std::isfinite(result.position.time) || result.position.time < 0 ||
        ((result.position.step == 0) != (result.position.time == 0)))
        throw std::invalid_argument("restart metadata scheduler position is invalid");
    return result;
}

std::uint64_t file_hash(const std::string& filename)
{
    std::ifstream input(filename, std::ios::binary);
    if (!input) throw std::invalid_argument("restart bundle file is missing: " + filename);
    std::uint64_t hash = 1469598103934665603ULL;
    char byte = 0;
    while (input.get(byte))
    {
        hash ^= static_cast<unsigned char>(byte);
        hash *= 1099511628211ULL;
    }
    if (!input.eof()) throw std::invalid_argument("failed to read restart bundle file: " + filename);
    return hash;
}

void require_paths(MPI_Comm communicator, const OperatorSplitCheckpointBundle& bundle)
{
    std::string error;
    if (bundle.u_checkpoint.empty() || bundle.v_checkpoint.empty() || bundle.metadata.empty())
        error = "restart bundle paths must be nonempty";
    detail::collective_signature_error(communicator, error);
    const std::string signature = bundle.u_checkpoint + '\n' + bundle.v_checkpoint + '\n' + bundle.metadata;
    int rank = 0;
    MPI_Comm_rank(communicator, &rank);
    std::uint64_t root_size = signature.size();
    MPI_Bcast(&root_size, 1, MPI_UINT64_T, 0, communicator);
    std::string root(root_size, '\0');
    if (rank == 0) root = signature;
    MPI_Bcast(root.data(), static_cast<int>(root.size()), MPI_CHAR, 0, communicator);
    detail::collective_signature_error(communicator,
        root == signature ? std::string_view{} : std::string_view("restart bundle paths differ across ranks"));
}

template<class Function>
void rank_zero_collective(MPI_Comm communicator, Function&& function)
{
    int rank = 0;
    MPI_Comm_rank(communicator, &rank);
    std::string error;
    if (rank == 0)
        try { function(); }
        catch (const std::exception& exception) { error = exception.what(); }
        catch (...) { error = "unknown restart bundle I/O failure"; }
    detail::collective_signature_error(communicator, error);
}

std::vector<unsigned char> read_metadata(MPI_Comm communicator, const std::string& filename)
{
    int rank = 0;
    MPI_Comm_rank(communicator, &rank);
    std::vector<unsigned char> bytes;
    rank_zero_collective(communicator, [&] {
        std::ifstream input(filename, std::ios::binary);
        if (!input) throw std::invalid_argument("restart metadata file is missing: " + filename);
        input.seekg(0, std::ios::end);
        const auto size = input.tellg();
        if (size <= 0 || static_cast<std::uint64_t>(size) > maximum_metadata_size)
            throw std::invalid_argument("restart metadata size is invalid");
        bytes.resize(static_cast<std::size_t>(size));
        input.seekg(0);
        input.read(reinterpret_cast<char*>(bytes.data()), size);
        if (!input) throw std::invalid_argument("failed to read restart metadata");
    });
    std::uint64_t size = bytes.size();
    MPI_Bcast(&size, 1, MPI_UINT64_T, 0, communicator);
    bytes.resize(static_cast<std::size_t>(size));
    MPI_Bcast(bytes.data(), static_cast<int>(bytes.size()), MPI_UNSIGNED_CHAR, 0, communicator);
    return bytes;
}

std::uint64_t collective_hash(MPI_Comm communicator, const std::string& filename)
{
    std::uint64_t hash = 0;
    rank_zero_collective(communicator, [&] { hash = file_hash(filename); });
    MPI_Bcast(&hash, 1, MPI_UINT64_T, 0, communicator);
    return hash;
}

void validate_components(const std::array<TransientReactionDiffusionNativeAdapter*, 2>& components)
{
    if (components[0] == nullptr || components[1] == nullptr || components[0] == components[1])
        throw std::invalid_argument("restart bundle requires two distinct transient adapters");
    if (!components[0]->is_initialized() || !components[1]->is_initialized())
        throw std::invalid_argument("restart bundle requires initialized transient adapters");
}

std::string gathered_union(MPI_Comm communicator, const std::set<std::string>& local)
{
    std::string packed;
    for (const auto& record : local) packed += record + '\n';
    int ranks = 1;
    MPI_Comm_size(communicator, &ranks);
    const int local_size = static_cast<int>(packed.size());
    std::vector<int> sizes(static_cast<std::size_t>(ranks));
    MPI_Allgather(&local_size, 1, MPI_INT, sizes.data(), 1, MPI_INT, communicator);
    std::vector<int> offsets(static_cast<std::size_t>(ranks));
    for (int rank = 1; rank < ranks; ++rank)
        offsets[static_cast<std::size_t>(rank)] = offsets[static_cast<std::size_t>(rank - 1)] +
                                                  sizes[static_cast<std::size_t>(rank - 1)];
    std::string all(static_cast<std::size_t>(offsets.back() + sizes.back()), '\0');
    MPI_Allgatherv(packed.data(), local_size, MPI_CHAR, all.data(), sizes.data(), offsets.data(), MPI_CHAR,
                   communicator);
    std::set<std::string> canonical;
    std::istringstream input(all);
    for (std::string record; std::getline(input, record);)
        if (!record.empty()) canonical.insert(std::move(record));
    std::string result;
    for (const auto& record : canonical) result += record + '\n';
    return result;
}

std::string mesh_signature(const KARMAMesh& mesh)
{
    moab::Tag global_id = nullptr;
    if (mesh.moab()->tag_get_handle("GLOBAL_ID", global_id) != moab::MB_SUCCESS)
        throw std::invalid_argument("restart mesh has no conventional GLOBAL_ID");
    std::set<std::string> records;
    for (const auto vertex : mesh.vertices())
    {
        int id = 0;
        double coordinates[3]{};
        if (mesh.moab()->tag_get_data(global_id, &vertex, 1, &id) != moab::MB_SUCCESS || id <= 0 ||
            mesh.moab()->get_coords(&vertex, 1, coordinates) != moab::MB_SUCCESS)
            throw std::invalid_argument("restart mesh vertex identity is invalid");
        std::ostringstream record;
        record << 'V' << ' ' << id;
        for (const double coordinate : coordinates)
            record << ' ' << std::hex << std::bit_cast<std::uint64_t>(coordinate);
        records.insert(record.str());
    }
    for (const auto element : mesh.elements())
    {
        int id = 0;
        const moab::EntityHandle* connectivity = nullptr;
        int count = 0;
        if (mesh.moab()->tag_get_data(global_id, &element, 1, &id) != moab::MB_SUCCESS || id <= 0 ||
            mesh.moab()->get_connectivity(element, connectivity, count) != moab::MB_SUCCESS)
            throw std::invalid_argument("restart mesh element identity is invalid");
        std::ostringstream record;
        record << 'E' << ' ' << id << ' ' << static_cast<int>(mesh.moab()->type_from_handle(element)) << ' ' << count;
        for (int index = 0; index < count; ++index)
        {
            int vertex_id = 0;
            if (mesh.moab()->tag_get_data(global_id, connectivity + index, 1, &vertex_id) != moab::MB_SUCCESS ||
                vertex_id <= 0)
                throw std::invalid_argument("restart mesh connectivity identity is invalid");
            record << ' ' << vertex_id;
        }
        records.insert(record.str());
    }
    return gathered_union(mesh.communicator(), records);
}

using KeyValues = std::map<DofKey, PetscScalar>;

KeyValues gather_values(const KARMASystem& system, Vec values)
{
    const auto owned = system.dof_map().owned_key_indices();
    std::vector<std::uint64_t> local;
    local.reserve(owned.size() * (DofKey::width + 1));
    for (const auto& [key, index] : owned)
    {
        PetscScalar value = 0;
        if (VecGetValues(values, 1, &index, &value) != PETSC_SUCCESS)
            throw std::invalid_argument("failed to read restart checkpoint value");
        local.insert(local.end(), key.begin(), key.end());
        local.push_back(std::bit_cast<std::uint64_t>(static_cast<double>(PetscRealPart(value))));
    }
    int ranks = 1;
    MPI_Comm_size(system.communicator(), &ranks);
    const int local_size = static_cast<int>(local.size());
    std::vector<int> sizes(static_cast<std::size_t>(ranks));
    MPI_Allgather(&local_size, 1, MPI_INT, sizes.data(), 1, MPI_INT, system.communicator());
    std::vector<int> offsets(static_cast<std::size_t>(ranks));
    for (int rank = 1; rank < ranks; ++rank)
        offsets[static_cast<std::size_t>(rank)] = offsets[static_cast<std::size_t>(rank - 1)] +
                                                  sizes[static_cast<std::size_t>(rank - 1)];
    std::vector<std::uint64_t> all(static_cast<std::size_t>(offsets.back() + sizes.back()));
    MPI_Allgatherv(local.data(), local_size, MPI_UINT64_T, all.data(), sizes.data(), offsets.data(), MPI_UINT64_T,
                   system.communicator());
    constexpr std::size_t width = DofKey::width + 1;
    if (all.size() % width != 0) throw std::invalid_argument("restart checkpoint key records are malformed");
    KeyValues result;
    for (std::size_t offset = 0; offset < all.size(); offset += width)
    {
        DofKey key;
        std::copy_n(all.begin() + static_cast<std::ptrdiff_t>(offset), DofKey::width, key.begin());
        const PetscScalar value = std::bit_cast<double>(all[offset + DofKey::width]);
        const auto [entry, inserted] = result.emplace(key, value);
        if (!inserted && entry->second != value)
            throw std::invalid_argument("restart checkpoint has inconsistent duplicate stable DoF keys");
    }
    return result;
}

PetscOwner<Vec, VecDestroy> stage_by_key(const KARMASystem& loaded, const KARMASystem& destination)
{
    const auto values = gather_values(loaded, loaded.solution());
    const auto owned = destination.dof_map().owned_key_indices();
    PetscOwner<Vec, VecDestroy> staged;
    *staged.put() = destination.create_vector();
    if (VecSet(staged.get(), 0) != PETSC_SUCCESS) throw std::invalid_argument("failed to clear restart staging vector");
    for (const auto& [key, index] : owned)
    {
        const auto found = values.find(key);
        if (found == values.end()) throw std::invalid_argument("restart checkpoint is missing a destination stable DoF key");
        if (VecSetValue(staged.get(), index, found->second, INSERT_VALUES) != PETSC_SUCCESS)
            throw std::invalid_argument("failed to stage restart value by stable DoF key");
    }
    if (values.size() != static_cast<std::size_t>(destination.n_dofs()))
        throw std::invalid_argument("restart checkpoint stable DoF key set differs from destination");
    if (VecAssemblyBegin(staged.get()) != PETSC_SUCCESS || VecAssemblyEnd(staged.get()) != PETSC_SUCCESS)
        throw std::invalid_argument("failed to assemble restart staging vector");
    return staged;
}
}  // namespace

PreparedOperatorSplitRestart::PreparedOperatorSplitRestart(
    std::array<TransientReactionDiffusionNativeAdapter*, 2> components,
    std::array<PetscOwner<Vec, VecDestroy>, 2> staged, SchedulerPosition position,
    OperatorSplitRestartOptions options)
    : components_(components), staged_(std::move(staged)), position_(position), options_(options)
{}

void PreparedOperatorSplitRestart::apply()
{
    if (applied_) throw std::logic_error("prepared operator-split restart was already applied");
    const MPI_Comm communicator = components_[0]->communicator();
    std::array<PetscOwner<Vec, VecDestroy>, 2> snapshots;
    const std::array<Real, 2> times{components_[0]->committed_time(), components_[1]->committed_time()};
    std::string local_error;
    try
    {
        for (std::size_t component = 0; component < components_.size(); ++component)
        {
            if (VecDuplicate(components_[component]->committed_solution(), snapshots[component].put()) != PETSC_SUCCESS ||
                VecCopy(components_[component]->committed_solution(), snapshots[component].get()) != PETSC_SUCCESS)
                throw std::invalid_argument("failed to snapshot live restart destination");
        }
    }
    catch (const std::exception& error) { local_error = error.what(); }
    catch (...) { local_error = "unknown restart snapshot failure"; }
    detail::collective_signature_error(communicator, local_error);

    try
    {
        for (std::size_t component = 0; component < components_.size(); ++component)
        {
            local_error.clear();
            try
            {
                if (options_.before_publication != nullptr) options_.before_publication(component);
            }
            catch (const std::exception& error) { local_error = error.what(); }
            catch (...) { local_error = "unknown restart publication preflight failure"; }
            detail::collective_signature_error(communicator, local_error);

            local_error.clear();
            try { components_[component]->restore_committed_solution(staged_[component].get(), position_.time); }
            catch (const std::exception& error) { local_error = error.what(); }
            catch (...) { local_error = "unknown restart publication failure"; }
            detail::collective_signature_error(communicator, local_error);
        }
        applied_ = true;
    }
    catch (const std::exception& original)
    {
        const std::string original_message = original.what();
        std::string rollback_message;
        for (std::size_t component = 0; component < components_.size(); ++component)
        {
            local_error.clear();
            try { components_[component]->restore_committed_solution(snapshots[component].get(), times[component]); }
            catch (const std::exception& error) { local_error = error.what(); }
            catch (...) { local_error = "unknown restart rollback failure"; }
            try { detail::collective_signature_error(communicator, local_error); }
            catch (const std::exception& error)
            {
                if (rollback_message.empty()) rollback_message = error.what();
            }
        }
        if (!rollback_message.empty())
            throw std::runtime_error("operator-split restart publication failed: " + original_message +
                                     "; additionally rollback failed: " + rollback_message);
        throw std::runtime_error("operator-split restart publication failed: " + original_message +
                                 "; both live components were restored");
    }
}

void write_operator_split_checkpoint_bundle(
    const OperatorSplitCheckpointBundle& bundle, const CompiledOperatorSplitGraph& graph,
    std::array<TransientReactionDiffusionNativeAdapter*, 2> components, SchedulerPosition position)
{
    const MPI_Comm communicator = components[0] ? components[0]->communicator() : MPI_COMM_WORLD;
    require_paths(communicator, bundle);
    std::string error;
    try
    {
        validate_components(components);
        if (!std::isfinite(position.time) || position.time < 0 || position.step == 0)
            throw std::invalid_argument("checkpoint bundle requires a committed noninitial scheduler position");
        for (const auto* component : components)
            if (component->committed_time() != position.time)
                throw std::invalid_argument("component committed time differs from scheduler safe point");
    }
    catch (const std::exception& exception) { error = exception.what(); }
    detail::collective_signature_error(communicator, error);

    components[0]->write_checkpoint(bundle.u_checkpoint);
    components[1]->write_checkpoint(bundle.v_checkpoint);
    Metadata metadata{graph.signature(), graph.metadata().scheme, position,
                      collective_hash(communicator, bundle.u_checkpoint),
                      collective_hash(communicator, bundle.v_checkpoint)};
    rank_zero_collective(communicator, [&] {
        const auto bytes = encode(metadata);
        const std::filesystem::path destination(bundle.metadata);
        const auto temporary = destination.string() + ".tmp";
        {
            std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
            output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
            if (!output) throw std::invalid_argument("failed to write restart metadata");
        }
        std::error_code filesystem_error;
        std::filesystem::rename(temporary, destination, filesystem_error);
        if (filesystem_error) throw std::invalid_argument("failed to publish restart metadata: " + filesystem_error.message());
    });
}

PreparedOperatorSplitRestart prepare_operator_split_restart(
    const OperatorSplitCheckpointBundle& bundle, const CompiledOperatorSplitGraph& graph,
    std::array<TransientReactionDiffusionNativeAdapter*, 2> components, OperatorSplitRestartOptions options)
{
    const MPI_Comm communicator = components[0] ? components[0]->communicator() : MPI_COMM_WORLD;
    require_paths(communicator, bundle);
    std::string error;
    try { validate_components(components); }
    catch (const std::exception& exception) { error = exception.what(); }
    detail::collective_signature_error(communicator, error);

    const Metadata metadata = decode(read_metadata(communicator, bundle.metadata));
    error.clear();
    if (metadata.signature != graph.signature()) error = "restart bundle graph signature differs";
    else if (metadata.scheme != graph.metadata().scheme) error = "restart bundle split scheme differs";
    else if (collective_hash(communicator, bundle.u_checkpoint) != metadata.u_hash)
        error = "restart bundle U checkpoint digest differs";
    else if (collective_hash(communicator, bundle.v_checkpoint) != metadata.v_hash)
        error = "restart bundle V checkpoint digest differs";
    detail::collective_signature_error(communicator, error);

    auto u_loaded = KARMAMoabIO::load_checkpoint(bundle.u_checkpoint, communicator);
    auto v_loaded = KARMAMoabIO::load_checkpoint(bundle.v_checkpoint, communicator);
    error.clear();
    try
    {
        if (mesh_signature(u_loaded.mesh()) != mesh_signature(components[0]->checkpoint_system().mesh()))
            throw std::invalid_argument("restart U mesh geometry or topology differs from destination");
        if (mesh_signature(v_loaded.mesh()) != mesh_signature(components[1]->checkpoint_system().mesh()))
            throw std::invalid_argument("restart V mesh geometry or topology differs from destination");
    }
    catch (const std::exception& exception) { error = exception.what(); }
    detail::collective_signature_error(communicator, error);
    std::array<PetscOwner<Vec, VecDestroy>, 2> staged;
    error.clear();
    try
    {
        staged[0] = stage_by_key(u_loaded.system(), components[0]->checkpoint_system());
        staged[1] = stage_by_key(v_loaded.system(), components[1]->checkpoint_system());
    }
    catch (const std::exception& exception) { error = exception.what(); }
    detail::collective_signature_error(communicator, error);
    return PreparedOperatorSplitRestart(components, std::move(staged), metadata.position, options);
}
}  // namespace KARMA
