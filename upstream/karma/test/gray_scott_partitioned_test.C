#include "karma_distributed_point_locator.h"
#include "karma_mesh.h"
#include "karma_moab_io.h"
#include "karma_operator_split_graph.h"
#include "karma_transient_reaction_diffusion_adapter.h"

#include <petsc.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace
{
using namespace KARMA;

constexpr const char* parallel_read_options =
    "PARALLEL=READ_PART;PARTITION=PARALLEL_PARTITION;PARALLEL_RESOLVE_SHARED_ENTS";
constexpr double trajectory_tolerance = 2.0e-10;

void require(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

OperatorSplitGraphSpec graph()
{
    const auto component = [](const std::string& id) {
        return ComponentSpec{id, {{"state", PortDirection::Output, false, {id + ".state", "1", 1, FieldRole::Prognostic}},
                                   {"coupled", PortDirection::Input, true, {id + ".coupled", "1", 1, FieldRole::Input}}},
                             true, true};
    };
    return {"gray-scott-partitioned-v1", {{{"U", component("U")}, {"V", component("V")}}},
            {{{"U", "state", "V", "coupled"}, {"V", "state", "U", "coupled"}}}, SplitScheme::Strang};
}

GrayScottParameters parameters()
{
    GrayScottParameters result;
    result.diffusion_u = 0.0;
    result.diffusion_v = 0.0;
    return result;
}

void write_fixture(const char* filename, unsigned nx, unsigned ny)
{
    KARMAMesh mesh(2, PETSC_COMM_WORLD);
    mesh.generate_box(nx, ny, 0);
    KARMAMoabIO(mesh).write_mesh(filename);
}

struct FieldRecord
{
    DofKey key;
    double value = 0.0;
};

using Trajectory = std::map<DofKey, double>;

Trajectory gather_field(const KARMAField& field)
{
    const auto& binding = std::get<PetscFieldBinding>(field.binding());
    const auto owned = binding.system->dof_map().owned_key_indices();
    std::vector<std::uint64_t> local;
    local.reserve(owned.size() * (DofKey::width + 1));
    for (const auto& [key, index] : owned)
    {
        PetscScalar value = 0.0;
        require(VecGetValues(binding.vector, 1, &index, &value) == PETSC_SUCCESS, "read owned stable-key value");
        require(std::isfinite(PetscRealPart(value)), "trajectory contains a nonfinite value");
        local.insert(local.end(), key.begin(), key.end());
        local.push_back(std::bit_cast<std::uint64_t>(static_cast<double>(PetscRealPart(value))));
    }

    int ranks = 1;
    MPI_Comm_size(PETSC_COMM_WORLD, &ranks);
    const int local_count = static_cast<int>(local.size());
    std::vector<int> counts(static_cast<std::size_t>(ranks));
    MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, PETSC_COMM_WORLD);
    std::vector<int> offsets(static_cast<std::size_t>(ranks));
    for (int rank = 1; rank < ranks; ++rank) offsets[static_cast<std::size_t>(rank)] = offsets[static_cast<std::size_t>(rank - 1)] + counts[static_cast<std::size_t>(rank - 1)];
    const int total = offsets.back() + counts.back();
    std::vector<std::uint64_t> gathered(static_cast<std::size_t>(total));
    MPI_Allgatherv(local.data(), local_count, MPI_UINT64_T, gathered.data(), counts.data(), offsets.data(), MPI_UINT64_T,
                   PETSC_COMM_WORLD);

    constexpr std::size_t record_width = DofKey::width + 1;
    require(gathered.size() % record_width == 0, "malformed gathered stable-key trajectory");
    Trajectory result;
    for (std::size_t offset = 0; offset < gathered.size(); offset += record_width)
    {
        DofKey key;
        std::copy_n(gathered.begin() + static_cast<std::ptrdiff_t>(offset), DofKey::width, key.begin());
        const auto [found, inserted] = result.emplace(key, std::bit_cast<double>(gathered[offset + DofKey::width]));
        require(inserted || found->second == std::bit_cast<double>(gathered[offset + DofKey::width]), "duplicate stable DoF key");
    }
    return result;
}

struct Reference
{
    std::string graph_id;
    double time = 0.0;
    std::uint64_t step = 0;
    Trajectory u;
    Trajectory v;
};

void write_trajectory(std::ofstream& output, const Trajectory& trajectory)
{
    const std::uint64_t count = trajectory.size();
    output.write(reinterpret_cast<const char*>(&count), sizeof(count));
    for (const auto& [key, value] : trajectory)
    {
        output.write(reinterpret_cast<const char*>(key.words.data()), static_cast<std::streamsize>(sizeof(key.words)));
        const auto bits = std::bit_cast<std::uint64_t>(value);
        output.write(reinterpret_cast<const char*>(&bits), sizeof(bits));
    }
}

Trajectory read_trajectory(std::ifstream& input)
{
    std::uint64_t count = 0;
    input.read(reinterpret_cast<char*>(&count), sizeof(count));
    Trajectory result;
    for (std::uint64_t entry = 0; entry < count; ++entry)
    {
        DofKey key;
        std::uint64_t bits = 0;
        input.read(reinterpret_cast<char*>(key.words.data()), static_cast<std::streamsize>(sizeof(key.words)));
        input.read(reinterpret_cast<char*>(&bits), sizeof(bits));
        require(input.good(), "read stable-key trajectory record");
        require(result.emplace(key, std::bit_cast<double>(bits)).second, "duplicate reference stable DoF key");
    }
    return result;
}

void write_reference(const Reference& reference, const char* filename)
{
    std::ofstream output(filename, std::ios::binary | std::ios::trunc);
    require(output.good(), "open Gray-Scott trajectory reference");
    const std::uint64_t id_size = reference.graph_id.size();
    output.write(reinterpret_cast<const char*>(&id_size), sizeof(id_size));
    output.write(reference.graph_id.data(), static_cast<std::streamsize>(id_size));
    output.write(reinterpret_cast<const char*>(&reference.time), sizeof(reference.time));
    output.write(reinterpret_cast<const char*>(&reference.step), sizeof(reference.step));
    write_trajectory(output, reference.u);
    write_trajectory(output, reference.v);
    require(output.good(), "write Gray-Scott trajectory reference");
}

Reference read_reference(const char* filename)
{
    std::ifstream input(filename, std::ios::binary);
    require(input.good(), "open Gray-Scott trajectory reference");
    std::uint64_t id_size = 0;
    input.read(reinterpret_cast<char*>(&id_size), sizeof(id_size));
    Reference result;
    result.graph_id.resize(static_cast<std::size_t>(id_size));
    input.read(result.graph_id.data(), static_cast<std::streamsize>(id_size));
    input.read(reinterpret_cast<char*>(&result.time), sizeof(result.time));
    input.read(reinterpret_cast<char*>(&result.step), sizeof(result.step));
    result.u = read_trajectory(input);
    result.v = read_trajectory(input);
    require(input.good(), "read Gray-Scott trajectory reference");
    return result;
}

class FailOnceAdapter final : public KARMANativeComponent
{
  public:
    FailOnceAdapter(TransientReactionDiffusionNativeAdapter& wrapped, std::string phase, bool inject)
        : wrapped_(wrapped), phase_(std::move(phase)), inject_(inject) {}
    const ComponentSpec& specification() const noexcept override { return wrapped_.specification(); }
    MPI_Comm communicator() const noexcept override { return wrapped_.communicator(); }
    void initialize() override { wrapped_.initialize(); }
    KARMAField input_field(PortId port) override { return wrapped_.input_field(std::move(port)); }
    KARMAField output_field(PortId port, FieldVisibility visibility) override { return wrapped_.output_field(std::move(port), visibility); }
    void checkpoint() override { wrapped_.checkpoint(); }
    void bind_trial_input(PortId port, Vec values) override
    {
        fail("exchange");
        wrapped_.bind_trial_input(std::move(port), values);
    }
    void advance(const TimeInterval& interval) override
    {
        fail("advance");
        wrapped_.advance(interval);
    }
    void commit() override
    {
        fail("commit");
        wrapped_.commit();
    }
    void release_checkpoint() noexcept override { wrapped_.release_checkpoint(); }
    bool restore() noexcept override
    {
        if (inject_ && phase_ == "restore" && !failed_)
        {
            failed_ = true;
            return false;
        }
        return wrapped_.restore();
    }
    void disarm() noexcept { failed_ = true; }
  private:
    void fail(const char* phase)
    {
        if (inject_ && phase_ == phase && !failed_)
        {
            failed_ = true;
            throw std::runtime_error("asymmetric injected " + phase_ + " failure");
        }
    }
    TransientReactionDiffusionNativeAdapter& wrapped_;
    std::string phase_;
    bool inject_ = false;
    bool failed_ = false;
};

struct RunResult
{
    Reference reference;
    std::array<std::size_t, 2> setup_counts{};
};

bool remote_points(const KARMAMesh& source, const KARMAMesh& target)
{
    KARMADistributedPointLocator locator(source);
    locator.setup();
    std::vector<Point> points;
    for (const auto element : target.owned_elements()) points.push_back(target.centroid(element));
    const auto locations = locator.locate(points);
    bool local_remote = false;
    for (const auto& location : locations)
        local_remote = local_remote || (location.status == PointLocationStatus::Located && location.source_handle == 0);
    int result = local_remote ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &result, 1, MPI_INT, MPI_MAX, PETSC_COMM_WORLD);
    return result != 0;
}

RunResult run(const char* u_file, const char* v_file, bool partitioned, const std::string& failure = {})
{
    const bool retry = failure.starts_with("retry-");
    const std::string phase = retry ? failure.substr(6) : failure;
    KARMAMesh u_mesh(2, PETSC_COMM_WORLD);
    KARMAMesh v_mesh(2, PETSC_COMM_WORLD);
    KARMAMoabIO(u_mesh).read_mesh(u_file, partitioned ? parallel_read_options : "");
    KARMAMoabIO(v_mesh).read_mesh(v_file, partitioned ? parallel_read_options : "");
    u_mesh.update();
    v_mesh.update();
    TransientReactionDiffusionNativeAdapter raw_u("U", u_mesh, GrayScottSpecies::U, parameters(), 1.0);
    TransientReactionDiffusionNativeAdapter raw_v("V", v_mesh, GrayScottSpecies::V, parameters(), .2);
    int rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    FailOnceAdapter u(raw_u, phase, rank == 1);
    FailOnceAdapter v(raw_v, phase, rank == 1);
    const auto compiled = compile_operator_split_graph(graph(), PETSC_COMM_WORLD);
    KARMACouplingScheduler scheduler(compiled, {&u, &v});

    if (!phase.empty())
    {
        bool threw = false;
        try { scheduler.advance(0.0, 0.01); }
        catch (const std::exception&) { threw = true; }
        int global_threw = threw ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &global_threw, 1, MPI_INT, MPI_MIN, PETSC_COMM_WORLD);
        if (phase == "restore")
        {
            require(global_threw == 1, "asymmetric restore failure was not reconciled on every rank");
            require(scheduler.state() == SchedulerState::Failed, "restore failure did not terminally fail scheduler");
            return {};
        }
        require(global_threw == 1, "asymmetric failure was not reconciled on every rank");
        require(scheduler.position().step == 0 && scheduler.position().time == 0.0, "recoverable failure changed scheduler position");
        require(retry, "failure mode must be explicitly registered as a retry case");
        u.disarm();
        v.disarm();
    }

    for (int step = 0; step < 2; ++step) scheduler.advance(0.01 * step, 0.01 * (step + 1));
    require(scheduler.position().step == 2 && std::abs(scheduler.position().time - .02) < 1.e-14,
            "scheduler did not complete partitioned trajectory");
    const auto u_field = raw_u.output_field("state", FieldVisibility::Committed);
    const auto v_field = raw_v.output_field("state", FieldVisibility::Committed);
    return {{graph().id, scheduler.position().time, scheduler.position().step, gather_field(u_field), gather_field(v_field)},
            scheduler.connection_setup_counts()};
}

void compare(const Trajectory& reference, const Trajectory& actual, const char* name)
{
    require(reference.size() == actual.size(), std::string(name) + " stable-key cardinality differs");
    double maximum = 0.0;
    for (const auto& [key, expected] : reference)
    {
        const auto found = actual.find(key);
        require(found != actual.end(), std::string(name) + " missing stable DoF key");
        maximum = std::max(maximum, std::abs(found->second - expected));
    }
    require(maximum <= trajectory_tolerance, std::string(name) + " trajectory differs beyond independent-reduction tolerance");
}

void serial_reference(const char* u_file, const char* v_file, const char* reference_file)
{
    const auto result = run(u_file, v_file, false);
    require(result.setup_counts[0] == 1 && result.setup_counts[1] == 1, "serial scheduler did not compile each connection once");
    write_reference(result.reference, reference_file);
}

void verify_partitioned(const char* u_file, const char* v_file, const char* reference_file)
{
    const auto reference = read_reference(reference_file);
    const auto result = run(u_file, v_file, true);
    require(reference.graph_id == graph().id, "reference graph identity differs (stable graph signature)");
    require(reference.time == result.reference.time && reference.step == result.reference.step, "reference scheduler position differs");
    compare(reference.u, result.reference.u, "U");
    compare(reference.v, result.reference.v, "V");
    require(result.setup_counts[0] == 1 && result.setup_counts[1] == 1, "partitioned scheduler rebuilt a compiled connection");
}

void verify_remote(const char* u_file, const char* v_file)
{
    KARMAMesh u_mesh(2, PETSC_COMM_WORLD), v_mesh(2, PETSC_COMM_WORLD);
    KARMAMoabIO(u_mesh).read_mesh(u_file, parallel_read_options);
    KARMAMoabIO(v_mesh).read_mesh(v_file, parallel_read_options);
    u_mesh.update();
    v_mesh.update();
    require(remote_points(u_mesh, v_mesh), "V-to-U transfer has no remotely routed source evaluation point");
    require(remote_points(v_mesh, u_mesh), "U-to-V transfer has no remotely routed source evaluation point");
}

}  // namespace

int main(int argc, char** argv)
{
    if (PetscInitialize(&argc, &argv, nullptr, nullptr) != PETSC_SUCCESS) return 1;
    int result = 1;
    try
    {
        require(argc >= 2, "missing partitioned Gray-Scott mode");
        const std::string_view mode(argv[1]);
        if (mode == "--write-fixture")
        {
            require(argc == 5, "fixture mode requires filename nx ny");
            write_fixture(argv[2], static_cast<unsigned>(std::stoul(argv[3])), static_cast<unsigned>(std::stoul(argv[4])));
        }
        else if (mode == "--write-reference")
        {
            require(argc == 5, "reference mode requires U V reference files");
            serial_reference(argv[2], argv[3], argv[4]);
        }
        else if (mode == "--verify-partitioned")
        {
            require(argc == 5, "partitioned mode requires U V reference files");
            verify_partitioned(argv[2], argv[3], argv[4]);
        }
        else if (mode == "--verify-remote")
        {
            require(argc == 4, "remote mode requires U and V files");
            verify_remote(argv[2], argv[3]);
        }
        else if (mode.starts_with("--failure-"))
        {
            require(argc == 4, "failure mode requires U and V files");
            run(argv[2], argv[3], true, std::string(mode.substr(10)));
        }
        else if (mode.starts_with("--retry-"))
        {
            require(argc == 4, "retry mode requires U and V files");
            run(argv[2], argv[3], true, "retry-" + std::string(mode.substr(8)));
        }
        else throw std::invalid_argument("unknown partitioned Gray-Scott mode");
        result = 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "partitioned Gray-Scott test failure: " << error.what() << '\n';
    }
    PetscFinalize();
    return result;
}
