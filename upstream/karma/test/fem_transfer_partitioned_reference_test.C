#include "karma_fem_field_evaluator.h"
#include "karma_fem_transfer.h"
#include "karma_moab_io.h"
#include "karma_distributed_point_locator.h"

#include <petsc.h>

#include <bit>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <string_view>
#include <vector>

namespace
{
struct Record
{
    std::uint64_t id = 0;
    std::uint64_t value_bits = 0;
};

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

KARMA::VariableSpec source_spec()
{
    return {"u", KARMA::FieldBasisFamily::LagrangeNodal, 1, KARMA::BasisContinuity::cG, 1};
}

void set_source_values(KARMA::KARMASystem& system)
{
    for (const auto vertex : system.mesh().vertices())
    {
        double coordinates[3]{};
        require(system.mesh().moab()->get_coords(&vertex, 1, coordinates) == moab::MB_SUCCESS,
                "read source coordinates");
        const PetscInt index = system.dof_map().vertex_dof(vertex, 0);
        const PetscScalar value = coordinates[0] + 2.0 * coordinates[1];
        require(VecSetValue(system.solution(), index, value, INSERT_VALUES) == PETSC_SUCCESS, "set source value");
    }
    require(VecAssemblyBegin(system.solution()) == PETSC_SUCCESS && VecAssemblyEnd(system.solution()) == PETSC_SUCCESS,
            "assemble source values");
}

std::map<std::array<std::uint64_t, 3>, double> project(const char* source_file, const char* target_file, bool partitioned)
{
    KARMA::KARMAMesh source_mesh(2, PETSC_COMM_WORLD);
    KARMA::KARMAMesh target_mesh(2, PETSC_COMM_WORLD);
    KARMA::KARMAMoabIO(source_mesh).read_mesh(source_file,
        partitioned ? "PARALLEL=READ_PART;PARTITION=PARALLEL_PARTITION;PARALLEL_RESOLVE_SHARED_ENTS" : "");
    KARMA::KARMAMoabIO(target_mesh).read_mesh(target_file,
        partitioned ? "PARALLEL=READ_PART;PARTITION=PARALLEL_PARTITION;PARALLEL_RESOLVE_SHARED_ENTS" : "");
    source_mesh.update();
    target_mesh.update();
    KARMA::KARMASystem source(source_mesh, "partitioned-transfer-source");
    const auto spec = source_spec();
    source.add_variable(spec);
    source.init();
    set_source_values(source);
    KARMA::KARMAField field(KARMA::PetscFieldBinding{&source, source.solution(), 0, spec,
                                                      KARMA::FieldScalarRepresentation::PetscScalar});
    KARMA::KARMAFEMFieldEvaluator evaluator(field);
    KARMA::KARMADistributedPointLocator locator(source_mesh);
    locator.setup();
    KARMA::KARMABasisRegistry registry;
    const auto basis = registry.create({KARMA::ReferenceCell::Quadrilateral,
                                        KARMA::FieldBasisFamily::LagrangeNodal, 0, KARMA::BasisContinuity::dG});
    const auto& elements = partitioned ? target_mesh.owned_elements() : target_mesh.elements();
    const std::vector<moab::EntityHandle> selected(elements.begin(), elements.end());
    const auto projected = KARMA::KARMAFEMTransfer::project_elements(target_mesh, selected, basis, 4, locator,
                                                                       evaluator, 1, true);
    std::map<std::array<std::uint64_t, 3>, double> result;
    for (std::size_t index = 0; index < selected.size(); ++index)
    {
        const auto centroid = target_mesh.centroid(selected[index]);
        const auto key = std::array<std::uint64_t, 3>{
            std::bit_cast<std::uint64_t>(centroid(0)), std::bit_cast<std::uint64_t>(centroid(1)), 0};
        result.emplace(key, projected[index].coefficients.front());
        if (!partitioned)
        {
        }
    }
    return result;
}

void verify_project_cg_asymmetric(const char* source_file)
{
    KARMA::KARMAMesh source_mesh(2, PETSC_COMM_WORLD);
    KARMA::KARMAMesh target_mesh(2, PETSC_COMM_WORLD);
    KARMA::KARMAMoabIO(source_mesh).read_mesh(
        source_file, "PARALLEL=READ_PART;PARTITION=PARALLEL_PARTITION;PARALLEL_RESOLVE_SHARED_ENTS");
    target_mesh.generate_box(1, 1, 0);
    source_mesh.update();
    target_mesh.update();

    KARMA::KARMASystem source(source_mesh, "asymmetric-projection-source");
    const auto spec = source_spec();
    source.add_variable(spec);
    source.init();
    set_source_values(source);
    KARMA::KARMAField field(KARMA::PetscFieldBinding{&source, source.solution(), 0, spec,
                                                      KARMA::FieldScalarRepresentation::PetscScalar});
    KARMA::KARMAFEMFieldEvaluator evaluator(field);
    KARMA::KARMADistributedPointLocator locator(source_mesh);
    locator.setup();

    KARMA::KARMASystem target(target_mesh, "asymmetric-projection-target");
    target.add_variable(spec);
    target.init();
    int local_assembly = target.dof_map().assembly_elements().empty() ? 0 : 1;
    int global_assembly = 0;
    MPI_Allreduce(&local_assembly, &global_assembly, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD);
    require(global_assembly == 1, "asymmetric cG projection setup is not asymmetric");
    Vec projected = target.create_vector();
    KARMA::KARMAFEMTransfer::project_cg(target, 0, 4, locator, evaluator, projected);
    PetscReal norm = 0.0;
    require(VecNorm(projected, NORM_INFINITY, &norm) == PETSC_SUCCESS && std::isfinite(norm),
            "asymmetric cG projection produced a nonfinite result");
    require(VecSet(projected, 7.0) == PETSC_SUCCESS, "set projection sentinel");
    require(VecSet(source.solution(), std::numeric_limits<PetscScalar>::quiet_NaN()) == PETSC_SUCCESS, "set nonfinite source");
    bool failed = false;
    try { KARMA::KARMAFEMTransfer::project_cg(target, 0, 4, locator, evaluator, projected); }
    catch (const std::exception&) { failed = true; }
    require(failed, "nonfinite cG projection was accepted");
    require(VecNorm(projected, NORM_INFINITY, &norm) == PETSC_SUCCESS && norm == 7.0,
            "failed cG projection modified its target");
    require(VecDestroy(&projected) == PETSC_SUCCESS, "destroy asymmetric cG projection result");

    KARMA::KARMASystem multicomponent_target(target_mesh, "multicomponent-projection-target");
    const KARMA::VariableSpec multicomponent_spec{
        "u2", KARMA::FieldBasisFamily::LagrangeNodal, 1, KARMA::BasisContinuity::cG, 2};
    multicomponent_target.add_variable(multicomponent_spec);
    multicomponent_target.init();
    Vec multicomponent_result = multicomponent_target.create_vector();
    require(VecSet(multicomponent_result, 11.0) == PETSC_SUCCESS, "set multicomponent projection sentinel");
    bool rejected = false;
    try
    {
        KARMA::KARMAFEMTransfer::project_cg(multicomponent_target, 0, 4, locator, evaluator,
                                             multicomponent_result);
    }
    catch (const std::exception& error)
    {
        rejected = std::string_view(error.what()).find("multicomponent") != std::string_view::npos;
    }
    require(rejected, "multicomponent cG target was not explicitly rejected");
    require(VecNorm(multicomponent_result, NORM_INFINITY, &norm) == PETSC_SUCCESS && norm == 11.0,
            "multicomponent rejection modified its target");
    require(VecDestroy(&multicomponent_result) == PETSC_SUCCESS, "destroy multicomponent projection result");
}

void verify_project_cg_asymmetric_uncovered(const char* source_file, const char* target_file)
{
    int rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    KARMA::KARMAMesh source_mesh(2, PETSC_COMM_WORLD);
    KARMA::KARMAMesh target_mesh(2, PETSC_COMM_WORLD);
    const char* parallel_options =
        "PARALLEL=READ_PART;PARTITION=PARALLEL_PARTITION;PARALLEL_RESOLVE_SHARED_ENTS";
    KARMA::KARMAMoabIO(source_mesh).read_mesh(source_file, parallel_options);
    KARMA::KARMAMoabIO(target_mesh).read_mesh(target_file, parallel_options);
    source_mesh.update();
    if (rank == 1)
    {
        const auto& vertices = target_mesh.vertices();
        std::vector<double> coordinates(vertices.size() * 3);
        require(target_mesh.moab()->get_coords(vertices, coordinates.data()) == moab::MB_SUCCESS,
                "read target coordinates for uncovered mutation");
        for (std::size_t index = 0; index < vertices.size(); ++index) coordinates[3 * index] += 2.0;
        require(target_mesh.moab()->set_coords(vertices, coordinates.data()) == moab::MB_SUCCESS,
                "move one rank's target outside source mesh");
    }
    target_mesh.update();

    KARMA::KARMASystem source(source_mesh, "asymmetric-uncovered-source");
    const auto spec = source_spec();
    source.add_variable(spec);
    source.init();
    set_source_values(source);
    KARMA::KARMAField field(KARMA::PetscFieldBinding{&source, source.solution(), 0, spec,
                                                      KARMA::FieldScalarRepresentation::PetscScalar});
    KARMA::KARMAFEMFieldEvaluator evaluator(field);
    KARMA::KARMADistributedPointLocator locator(source_mesh);
    locator.setup();
    KARMA::KARMASystem target(target_mesh, "asymmetric-uncovered-target");
    target.add_variable(spec);
    target.init();
    Vec projected = target.create_vector();
    require(VecSet(projected, 13.0) == PETSC_SUCCESS, "set uncovered projection sentinel");
    std::string message;
    try { KARMA::KARMAFEMTransfer::project_cg(target, 0, 4, locator, evaluator, projected); }
    catch (const std::exception& error) { message = error.what(); }
    require(!message.empty(), "asymmetric uncovered point was not rejected");
    int length = rank == 0 ? static_cast<int>(message.size()) : 0;
    MPI_Bcast(&length, 1, MPI_INT, 0, PETSC_COMM_WORLD);
    std::string reference(static_cast<std::size_t>(length), '\0');
    if (rank == 0) reference = message;
    MPI_Bcast(reference.data(), length, MPI_CHAR, 0, PETSC_COMM_WORLD);
    require(message == reference, "asymmetric uncovered diagnostics differ across ranks");
    require(message.find("uncovered") != std::string::npos || message.find("unlocated") != std::string::npos,
            "asymmetric uncovered diagnostic is missing coverage detail");
    PetscReal norm = 0.0;
    require(VecNorm(projected, NORM_INFINITY, &norm) == PETSC_SUCCESS && norm == 13.0,
            "asymmetric uncovered failure modified its target");
    require(VecDestroy(&projected) == PETSC_SUCCESS, "destroy uncovered projection result");
}

void write_reference(const char* source_file, const char* target_file, const char* reference_file)
{
    const auto values = project(source_file, target_file, false);
    std::ofstream output(reference_file, std::ios::binary | std::ios::trunc);
    require(output.good(), "open transfer reference");
    const std::uint64_t count = values.size();
    output.write(reinterpret_cast<const char*>(&count), sizeof(count));
    for (const auto& [key, value] : values)
    {
        const auto bits = std::bit_cast<std::uint64_t>(value);
        output.write(reinterpret_cast<const char*>(key.data()), static_cast<std::streamsize>(key.size() * sizeof(key[0])));
        output.write(reinterpret_cast<const char*>(&bits), sizeof(bits));
    }
}

std::map<std::array<std::uint64_t, 3>, std::uint64_t> read_reference(const char* filename)
{
    std::ifstream input(filename, std::ios::binary);
    require(input.good(), "open transfer reference for reading");
    std::uint64_t count = 0;
    input.read(reinterpret_cast<char*>(&count), sizeof(count));
    std::map<std::array<std::uint64_t, 3>, std::uint64_t> reference;
    for (std::uint64_t index = 0; index < count; ++index)
    {
        std::array<std::uint64_t, 3> key{};
        std::uint64_t bits = 0;
        input.read(reinterpret_cast<char*>(key.data()), static_cast<std::streamsize>(key.size() * sizeof(key[0])));
        input.read(reinterpret_cast<char*>(&bits), sizeof(bits));
        reference.emplace(key, bits);
    }
    return reference;
}

int verify_partitioned(const char* source_file, const char* target_file, const char* reference_file)
{
    const auto reference = read_reference(reference_file);
    const auto values = project(source_file, target_file, true);
    std::vector<std::uint64_t> local_words;
    local_words.reserve(values.size() * 4);
    for (const auto& [key, value] : values)
    {
        local_words.insert(local_words.end(), key.begin(), key.end());
        local_words.push_back(std::bit_cast<std::uint64_t>(value));
    }
    int ranks = 1;
    MPI_Comm_size(PETSC_COMM_WORLD, &ranks);
    const int local_count = static_cast<int>(local_words.size());
    std::vector<int> counts(static_cast<std::size_t>(ranks));
    MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, PETSC_COMM_WORLD);
    std::vector<int> offsets(counts.size());
    for (std::size_t index = 1; index < offsets.size(); ++index) offsets[index] = offsets[index - 1] + counts[index - 1];
    const int total = offsets.back() + counts.back();
    std::vector<std::uint64_t> gathered(static_cast<std::size_t>(total));
    MPI_Allgatherv(local_words.data(), local_count, MPI_UINT64_T, gathered.data(), counts.data(), offsets.data(),
                   MPI_UINT64_T, PETSC_COMM_WORLD);
    std::map<std::array<std::uint64_t, 3>, std::uint64_t> partitioned;
    for (std::size_t index = 0; index + 3 < gathered.size(); index += 4)
        partitioned.emplace(std::array<std::uint64_t, 3>{gathered[index], gathered[index + 1], gathered[index + 2]},
                            gathered[index + 3]);
    int local_ok = partitioned.size() == reference.size() ? 1 : 0;
    for (const auto& [key, bits] : reference)
    {
        const auto found = partitioned.find(key);
        const bool match = found != partitioned.end() && found->second == bits;
        local_ok = local_ok && match;
    }
    int global_ok = 0;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, PETSC_COMM_WORLD);
    return global_ok ? 0 : 1;
}
}

int main(int argc, char** argv)
{
    if (PetscInitializeNoArguments() != PETSC_SUCCESS) return 1;
    int result = 1;
    try
    {
        require(argc >= 3, "invalid transfer fixture arguments");
        const std::string_view mode(argv[1]);
        if (mode == "--write-reference")
        {
            require(argc == 5, "invalid reference fixture arguments");
            write_reference(argv[2], argv[3], argv[4]);
            result = 0;
        }
        else if (mode == "--verify-partitioned")
        {
            require(argc == 5, "invalid partitioned fixture arguments");
            result = verify_partitioned(argv[2], argv[3], argv[4]);
        }
        else if (mode == "--verify-project-cg-asymmetric")
        {
            require(argc == 3, "invalid asymmetric projection arguments");
            verify_project_cg_asymmetric(argv[2]);
            result = 0;
        }
        else if (mode == "--verify-project-cg-asymmetric-uncovered")
        {
            require(argc == 4, "invalid asymmetric uncovered projection arguments");
            verify_project_cg_asymmetric_uncovered(argv[2], argv[3]);
            result = 0;
        }
        else
            require(false, "unknown transfer fixture mode");
    }
    catch (const std::exception& error)
    {
        std::cerr << "transfer fixture failure: " << error.what() << '\n';
    }
    PetscFinalize();
    return result;
}
