#include "karma_mesh.h"
#include "karma_operator_split_checkpoint.h"
#include "karma_operator_split_graph.h"
#include "karma_transient_reaction_diffusion_adapter.h"

#include <petsc.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>

namespace
{
using namespace KARMA;

void require(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

ComponentSpec component_spec(const std::string& id)
{
    return {id, {{"state", PortDirection::Output, false, {id + ".state", "1", 1, FieldRole::Prognostic}},
                 {"coupled", PortDirection::Input, true, {id + ".coupled", "1", 1, FieldRole::Input}}},
            true, true};
}

OperatorSplitGraphSpec graph_spec(SplitScheme scheme = SplitScheme::Strang)
{
    return {"gray-scott-restart-v1", {{{"U", component_spec("U")}, {"V", component_spec("V")}}},
            {{{"U", "state", "V", "coupled"}, {"V", "state", "U", "coupled"}}}, scheme};
}

GrayScottParameters parameters(bool diffusive = false)
{
    GrayScottParameters result;
    result.diffusion_u = diffusive ? 0.01 : 0.0;
    result.diffusion_v = diffusive ? 0.005 : 0.0;
    return result;
}

struct Simulation
{
    KARMAMesh u_mesh{2, PETSC_COMM_WORLD};
    KARMAMesh v_mesh{2, PETSC_COMM_WORLD};
    TransientReactionDiffusionNativeAdapter u;
    TransientReactionDiffusionNativeAdapter v;
    CompiledOperatorSplitGraph graph;

    Simulation(SplitScheme scheme = SplitScheme::Strang, unsigned u_nx = 5, unsigned u_ny = 4,
               bool diffusive = false)
        : u("U", u_mesh, GrayScottSpecies::U, parameters(diffusive), 1.0),
          v("V", v_mesh, GrayScottSpecies::V, parameters(diffusive), 0.2),
          graph(compile_operator_split_graph(graph_spec(scheme), PETSC_COMM_WORLD))
    {
        u_mesh.generate_box(u_nx, u_ny, 0);
        v_mesh.generate_box(7, 3, 0);
    }
};

Vec committed(TransientReactionDiffusionNativeAdapter& component)
{
    return std::get<PetscFieldBinding>(component.output_field("state", FieldVisibility::Committed).binding()).vector;
}

void advance(KARMACouplingScheduler& scheduler, int first, int last)
{
    constexpr double dt = 0.01;
    for (int step = first; step < last; ++step) scheduler.advance(dt * step, dt * (step + 1));
}

using Trajectory = std::map<DofKey, double>;

Trajectory gather(TransientReactionDiffusionNativeAdapter& component)
{
    const auto field = component.output_field("state", FieldVisibility::Committed);
    const auto& binding = std::get<PetscFieldBinding>(field.binding());
    const auto owned = binding.system->dof_map().owned_key_indices();
    std::vector<std::uint64_t> local;
    local.reserve(owned.size() * (DofKey::width + 1));
    for (const auto& [key, index] : owned)
    {
        PetscScalar value = 0.0;
        require(VecGetValues(binding.vector, 1, &index, &value) == PETSC_SUCCESS, "read restart stable-key value");
        local.insert(local.end(), key.begin(), key.end());
        local.push_back(std::bit_cast<std::uint64_t>(static_cast<double>(PetscRealPart(value))));
    }
    int ranks = 1;
    MPI_Comm_size(PETSC_COMM_WORLD, &ranks);
    const int local_size = static_cast<int>(local.size());
    std::vector<int> sizes(static_cast<std::size_t>(ranks));
    MPI_Allgather(&local_size, 1, MPI_INT, sizes.data(), 1, MPI_INT, PETSC_COMM_WORLD);
    std::vector<int> offsets(static_cast<std::size_t>(ranks));
    for (int rank = 1; rank < ranks; ++rank)
        offsets[static_cast<std::size_t>(rank)] = offsets[static_cast<std::size_t>(rank - 1)] +
                                                  sizes[static_cast<std::size_t>(rank - 1)];
    std::vector<std::uint64_t> all(static_cast<std::size_t>(offsets.back() + sizes.back()));
    MPI_Allgatherv(local.data(), local_size, MPI_UINT64_T, all.data(), sizes.data(), offsets.data(), MPI_UINT64_T,
                   PETSC_COMM_WORLD);
    Trajectory result;
    constexpr std::size_t width = DofKey::width + 1;
    require(all.size() % width == 0, "malformed restart stable-key trajectory");
    for (std::size_t offset = 0; offset < all.size(); offset += width)
    {
        DofKey key;
        std::copy_n(all.begin() + static_cast<std::ptrdiff_t>(offset), DofKey::width, key.begin());
        const double value = std::bit_cast<double>(all[offset + DofKey::width]);
        const auto [entry, inserted] = result.emplace(key, value);
        require(inserted || entry->second == value, "inconsistent duplicate restart stable DoF key");
    }
    return result;
}

void compare(const Trajectory& expected, const Trajectory& actual, const char* name)
{
    require(expected.size() == actual.size(), std::string(name) + " restart stable-key cardinality differs");
    double maximum = 0.0;
    for (const auto& [key, value] : expected)
    {
        const auto found = actual.find(key);
        require(found != actual.end(), std::string(name) + " restart is missing a stable DoF key");
        maximum = std::max(maximum, std::abs(found->second - value));
    }
    require(maximum <= 2.0e-10, std::string(name) + " restarted trajectory differs from uninterrupted trajectory");
}

bool bitwise_equal(Vec expected, Vec actual)
{
    PetscInt size = 0;
    const PetscScalar* left = nullptr;
    const PetscScalar* right = nullptr;
    VecGetLocalSize(expected, &size);
    VecGetArrayRead(expected, &left);
    VecGetArrayRead(actual, &right);
    bool equal = true;
    for (PetscInt index = 0; index < size; ++index) equal = equal && left[index] == right[index];
    VecRestoreArrayRead(expected, &left);
    VecRestoreArrayRead(actual, &right);
    return equal;
}

OperatorSplitCheckpointBundle bundle(const std::string& prefix)
{
    return {prefix + "-U.h5m", prefix + "-V.h5m", prefix + "-scheduler.karma"};
}

void remove_bundle(const OperatorSplitCheckpointBundle& files)
{
    int rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    if (rank == 0)
    {
        std::error_code error;
        std::filesystem::remove(files.u_checkpoint, error);
        std::filesystem::remove(files.v_checkpoint, error);
        std::filesystem::remove(files.metadata, error);
    }
    MPI_Barrier(PETSC_COMM_WORLD);
}

void write_safe_point(Simulation& source, const OperatorSplitCheckpointBundle& files)
{
    KARMACouplingScheduler scheduler(source.graph, {&source.u, &source.v});
    advance(scheduler, 0, 2);
    write_operator_split_checkpoint_bundle(files, source.graph, {&source.u, &source.v}, scheduler.position());
}

void set_nonconstant(TransientReactionDiffusionNativeAdapter& component, double offset)
{
    for (const auto visibility : {FieldVisibility::Committed, FieldVisibility::TransactionLocal})
    {
        const auto field = component.output_field("state", visibility);
        const auto& binding = std::get<PetscFieldBinding>(field.binding());
        for (const auto& [key, index] : binding.system->dof_map().owned_key_indices())
        {
            const PetscScalar value = offset + 1.0e-3 * static_cast<double>(key.words[5]);
            require(VecSetValue(binding.vector, index, value, INSERT_VALUES) == PETSC_SUCCESS,
                    "set nonconstant restart state");
        }
        require(VecAssemblyBegin(binding.vector) == PETSC_SUCCESS && VecAssemblyEnd(binding.vector) == PETSC_SUCCESS,
                "assemble nonconstant restart state");
    }
}

void roundtrip(const std::string& prefix)
{
    const auto files = bundle(prefix);
    remove_bundle(files);
    Simulation uninterrupted;
    KARMACouplingScheduler uninterrupted_scheduler(uninterrupted.graph, {&uninterrupted.u, &uninterrupted.v});
    advance(uninterrupted_scheduler, 0, 4);

    Simulation checkpointed;
    write_safe_point(checkpointed, files);
    Simulation restarted;
    restarted.u.initialize();
    restarted.v.initialize();
    auto prepared = prepare_operator_split_restart(files, restarted.graph, {&restarted.u, &restarted.v});
    require(prepared.position().step == 2 && std::abs(prepared.position().time - 0.02) < 1.0e-14,
            "prepared restart position differs from committed safe point");
    prepared.apply();
    KARMACouplingScheduler restarted_scheduler(restarted.graph, {&restarted.u, &restarted.v}, prepared.position());
    advance(restarted_scheduler, 2, 4);

    compare(gather(uninterrupted.u), gather(restarted.u), "U");
    compare(gather(uninterrupted.v), gather(restarted.v), "V");
    require(restarted_scheduler.position().step == 4 &&
                std::abs(restarted_scheduler.position().time - 0.04) < 1.0e-14,
            "restarted scheduler did not continue from persisted position");
    remove_bundle(files);
}

void require_rejected_without_mutation(const std::string& prefix, std::string_view failure)
{
    const auto files = bundle(prefix);
    remove_bundle(files);
    Simulation source;
    write_safe_point(source, files);

    int rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    if (rank == 0 && failure == "missing-v") std::filesystem::remove(files.v_checkpoint);
    if (rank == 0 && failure == "metadata")
    {
        std::ofstream output(files.metadata, std::ios::binary | std::ios::trunc);
        output << "incompatible scheduler metadata";
    }
    MPI_Barrier(PETSC_COMM_WORLD);

    Simulation destination(failure == "incompatible" ? SplitScheme::Lie : SplitScheme::Strang);
    KARMACouplingScheduler initialized(destination.graph, {&destination.u, &destination.v});
    Vec u_before = nullptr;
    Vec v_before = nullptr;
    require(VecDuplicate(committed(destination.u), &u_before) == PETSC_SUCCESS &&
                VecDuplicate(committed(destination.v), &v_before) == PETSC_SUCCESS,
            "allocate mutation snapshots");
    VecCopy(committed(destination.u), u_before);
    VecCopy(committed(destination.v), v_before);
    const auto position_before = initialized.position();

    bool threw = false;
    try { (void)prepare_operator_split_restart(files, destination.graph, {&destination.u, &destination.v}); }
    catch (const std::exception&) { threw = true; }
    int all_threw = threw ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &all_threw, 1, MPI_INT, MPI_MIN, PETSC_COMM_WORLD);
    require(all_threw == 1, "invalid restart bundle was not rejected on every rank");
    require(bitwise_equal(u_before, committed(destination.u)), "invalid restart bundle changed U state");
    require(bitwise_equal(v_before, committed(destination.v)), "invalid restart bundle changed V state");
    require(initialized.position().time == position_before.time && initialized.position().step == position_before.step,
            "invalid restart bundle changed scheduler position");
    VecDestroy(&u_before);
    VecDestroy(&v_before);
    remove_bundle(files);
}

void wrong_mesh(const std::string& prefix)
{
    const auto files = bundle(prefix);
    remove_bundle(files);
    Simulation source(SplitScheme::Strang, 5, 4, true);
    KARMACouplingScheduler source_scheduler(source.graph, {&source.u, &source.v});
    set_nonconstant(source.u, 0.7);
    set_nonconstant(source.v, 0.1);
    advance(source_scheduler, 0, 1);
    write_operator_split_checkpoint_bundle(files, source.graph, {&source.u, &source.v}, source_scheduler.position());

    Simulation destination(SplitScheme::Strang, 4, 5, true);
    KARMACouplingScheduler destination_scheduler(destination.graph, {&destination.u, &destination.v});
    const auto u_before = gather(destination.u);
    bool threw = false;
    try { (void)prepare_operator_split_restart(files, destination.graph, {&destination.u, &destination.v}); }
    catch (const std::exception&) { threw = true; }
    int all_threw = threw ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &all_threw, 1, MPI_INT, MPI_MIN, PETSC_COMM_WORLD);
    require(all_threw == 1, "same-cardinality wrong mesh was accepted");
    compare(u_before, gather(destination.u), "wrong-mesh unchanged U");
    require(destination_scheduler.position().step == 0 && destination_scheduler.position().time == 0.0,
            "wrong-mesh prepare changed scheduler position");
    remove_bundle(files);
}

void apply_failure(const std::string& prefix)
{
    const auto files = bundle(prefix);
    remove_bundle(files);
    Simulation source;
    write_safe_point(source, files);
    Simulation destination;
    KARMACouplingScheduler destination_scheduler(destination.graph, {&destination.u, &destination.v});
    const auto u_before = gather(destination.u);
    const auto v_before = gather(destination.v);
    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &ranks);
    const bool inject = rank == (ranks == 1 ? 0 : 1);
    static bool inject_second_publication = false;
    inject_second_publication = inject;
    OperatorSplitRestartOptions options;
    options.before_publication = [](std::size_t component) {
        if (component == 1 && inject_second_publication)
            throw std::runtime_error("injected second restart publication failure");
    };
    auto prepared = prepare_operator_split_restart(files, destination.graph, {&destination.u, &destination.v}, options);
    bool threw = false;
    std::string diagnostic;
    try { prepared.apply(); }
    catch (const std::exception& error) { threw = true; diagnostic = error.what(); }
    int all_threw = threw ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &all_threw, 1, MPI_INT, MPI_MIN, PETSC_COMM_WORLD);
    require(all_threw == 1, "asymmetric second restart publication failure was not reconciled");
    require(diagnostic.find("rank ") != std::string::npos && diagnostic.find("publication") != std::string::npos,
            "restart publication failure diagnostic is not deterministic and rank-qualified");
    compare(u_before, gather(destination.u), "publication rollback U");
    compare(v_before, gather(destination.v), "publication rollback V");
    require(destination_scheduler.position().step == 0 && destination_scheduler.position().time == 0.0,
            "failed restart apply changed scheduler position");
    remove_bundle(files);
}
}  // namespace

int main(int argc, char** argv)
{
    if (PetscInitialize(&argc, &argv, nullptr, nullptr) != PETSC_SUCCESS) return 1;
    int result = 0;
    try
    {
        require(argc == 3, "usage: gray_scott_restart_test MODE PREFIX");
        const std::string_view mode(argv[1]);
        if (mode == "roundtrip") roundtrip(argv[2]);
        else if (mode == "wrong-mesh") wrong_mesh(argv[2]);
        else if (mode == "apply-failure") apply_failure(argv[2]);
        else if (mode == "missing-v" || mode == "metadata" || mode == "incompatible")
            require_rejected_without_mutation(argv[2], mode);
        else throw std::invalid_argument("unknown restart test mode");
    }
    catch (const std::exception& error)
    {
        std::cerr << "Gray-Scott restart test failure: " << error.what() << '\n';
        result = 1;
    }
    PetscFinalize();
    return result;
}
