#include "karma_mesh.h"
#include "karma_operator_split_checkpoint.h"
#include "karma_operator_split_graph.h"
#include "karma_transient_reaction_diffusion_adapter.h"

#include <petsc.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace
{
struct Options
{
    KARMA::SplitScheme scheme = KARMA::SplitScheme::Strang;
    double dt = 0.01;
    double final_time = 0.1;
    KARMA::GrayScottParameters parameters;
    double perturbation = 0.05;
    double perturbation_radius = 0.12;
    std::array<unsigned, 2> u_resolution{16, 16};
    std::array<unsigned, 2> v_resolution{19, 17};
    std::string u_mesh;
    std::string v_mesh;
    std::string output_dir{"gray-scott-output"};
    int checkpoint_cadence = 10;
};

void reconcile_local_error(const std::string& local_error, const char* epoch)
{
    const int local_failed = local_error.empty() ? 0 : 1;
    int any_failed = 0;
    if (MPI_Allreduce(&local_failed, &any_failed, 1, MPI_INT, MPI_MAX, PETSC_COMM_WORLD) != MPI_SUCCESS)
        throw std::runtime_error(std::string("MPI failure during Gray-Scott ") + epoch + " reconciliation");
    if (!any_failed) return;

    int rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    const int candidate = local_failed ? rank : std::numeric_limits<int>::max();
    int failing_rank = 0;
    MPI_Allreduce(&candidate, &failing_rank, 1, MPI_INT, MPI_MIN, PETSC_COMM_WORLD);
    int message_size = rank == failing_rank ? static_cast<int>(local_error.size()) : 0;
    MPI_Bcast(&message_size, 1, MPI_INT, failing_rank, PETSC_COMM_WORLD);
    std::string message(static_cast<std::size_t>(message_size), '\0');
    if (rank == failing_rank) message = local_error;
    MPI_Bcast(message.data(), message_size, MPI_CHAR, failing_rank, PETSC_COMM_WORLD);
    throw std::invalid_argument(std::string(epoch) + " failed on rank " + std::to_string(failing_rank) + ": " + message);
}

std::array<unsigned, 2> resolution(const std::string& text, const char* option)
{
    const auto comma = text.find(',');
    if (comma == std::string::npos || text.find(',', comma + 1) != std::string::npos)
        throw std::invalid_argument(std::string(option) + " must have the form NX,NY");
    std::size_t first_end = 0;
    std::size_t second_end = 0;
    const auto first = std::stoul(text.substr(0, comma), &first_end);
    const auto second_text = text.substr(comma + 1);
    const auto second = std::stoul(second_text, &second_end);
    if (first_end != comma || second_end != second_text.size() || first == 0 || second == 0)
        throw std::invalid_argument(std::string(option) + " entries must be positive integers");
    return {static_cast<unsigned>(first), static_cast<unsigned>(second)};
}

std::string string_option(const char* name, const std::string& fallback)
{
    std::array<char, PETSC_MAX_PATH_LEN> value{};
    PetscBool set = PETSC_FALSE;
    if (PetscOptionsGetString(nullptr, nullptr, name, value.data(), value.size(), &set) != PETSC_SUCCESS)
        throw std::runtime_error(std::string("failed to read ") + name);
    return set ? std::string(value.data()) : fallback;
}

Options options()
{
    Options result;
    const auto scheme = string_option("-gs_scheme", "strang");
    if (scheme == "lie") result.scheme = KARMA::SplitScheme::Lie;
    else if (scheme == "strang") result.scheme = KARMA::SplitScheme::Strang;
    else throw std::invalid_argument("-gs_scheme must be lie or strang");

    auto real_option = [](const char* name, double& value) {
        if (PetscOptionsGetReal(nullptr, nullptr, name, &value, nullptr) != PETSC_SUCCESS)
            throw std::invalid_argument(std::string("invalid real value for ") + name);
    };
    real_option("-gs_dt", result.dt);
    real_option("-gs_final_time", result.final_time);
    real_option("-gs_diffusion_u", result.parameters.diffusion_u);
    real_option("-gs_diffusion_v", result.parameters.diffusion_v);
    real_option("-gs_feed", result.parameters.feed_rate);
    real_option("-gs_kill", result.parameters.kill_rate);
    real_option("-gs_perturbation", result.perturbation);
    real_option("-gs_perturbation_radius", result.perturbation_radius);
    PetscInt cadence = result.checkpoint_cadence;
    if (PetscOptionsGetInt(nullptr, nullptr, "-gs_checkpoint_cadence", &cadence, nullptr) != PETSC_SUCCESS)
        throw std::invalid_argument("invalid integer value for -gs_checkpoint_cadence");
    result.checkpoint_cadence = static_cast<int>(cadence);
    result.u_resolution = resolution(string_option("-gs_u_resolution", "16,16"), "-gs_u_resolution");
    result.v_resolution = resolution(string_option("-gs_v_resolution", "19,17"), "-gs_v_resolution");
    result.u_mesh = string_option("-gs_u_mesh", "");
    result.v_mesh = string_option("-gs_v_mesh", "");
    result.output_dir = string_option("-gs_output_dir", result.output_dir);

    const auto finite_nonnegative = [](double value) { return std::isfinite(value) && value >= 0; };
    if (!std::isfinite(result.dt) || result.dt <= 0) throw std::invalid_argument("-gs_dt must be finite and positive");
    if (!std::isfinite(result.final_time) || result.final_time <= 0)
        throw std::invalid_argument("-gs_final_time must be finite and positive");
    if (!finite_nonnegative(result.parameters.diffusion_u) ||
        !finite_nonnegative(result.parameters.diffusion_v) ||
        (result.parameters.diffusion_u == 0 && result.parameters.diffusion_v == 0))
        throw std::invalid_argument("diffusion coefficients must be finite and nonnegative, with at least one nonzero");
    if (!finite_nonnegative(result.parameters.feed_rate) || !finite_nonnegative(result.parameters.kill_rate))
        throw std::invalid_argument("feed and kill rates must be finite and nonnegative");
    if (!finite_nonnegative(result.perturbation) || !std::isfinite(result.perturbation_radius) ||
        result.perturbation_radius <= 0 || result.perturbation_radius > 0.5)
        throw std::invalid_argument("perturbation must be nonnegative and its radius must be in (0,0.5]");
    if (result.checkpoint_cadence < 0) throw std::invalid_argument("checkpoint cadence must be nonnegative");
    if (result.output_dir.empty() && result.checkpoint_cadence != 0)
        throw std::invalid_argument("checkpoint output directory must be nonempty");
    const double steps = result.final_time / result.dt;
    if (std::abs(steps - std::round(steps)) > 1.e-10 * std::max(1.0, std::abs(steps)))
        throw std::invalid_argument("final time must be an integer multiple of dt");
    return result;
}

std::string option_signature(const Options& value, bool help_requested)
{
    std::ostringstream signature;
    signature.imbue(std::locale::classic());
    signature << std::hexfloat << static_cast<int>(value.scheme) << '|' << value.dt << '|'
              << value.final_time << '|' << value.parameters.diffusion_u << '|'
              << value.parameters.diffusion_v << '|' << value.parameters.feed_rate << '|'
              << value.parameters.kill_rate << '|' << value.perturbation << '|'
              << value.perturbation_radius << '|' << value.u_resolution[0] << '|'
              << value.u_resolution[1] << '|' << value.v_resolution[0] << '|'
              << value.v_resolution[1] << '|' << value.u_mesh.size() << ':' << value.u_mesh << '|'
              << value.v_mesh.size() << ':' << value.v_mesh << '|' << value.output_dir.size() << ':'
              << value.output_dir << '|' << value.checkpoint_cadence << '|' << help_requested;
    return signature.str();
}

void require_matching_options(const Options& value, bool help_requested)
{
    const std::string local = option_signature(value, help_requested);
    int rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    std::uint64_t root_size = rank == 0 ? local.size() : 0;
    MPI_Bcast(&root_size, 1, MPI_UINT64_T, 0, PETSC_COMM_WORLD);
    std::string root(static_cast<std::size_t>(root_size), '\0');
    if (rank == 0) root = local;
    MPI_Bcast(root.data(), static_cast<int>(root.size()), MPI_CHAR, 0, PETSC_COMM_WORLD);
    reconcile_local_error(local == root ? "" : "effective options differ across ranks", "option signature");
}

void preflight_mesh_file(const std::string& filename, const char* species)
{
    std::string error;
    if (!filename.empty())
    {
        std::ifstream input(filename, std::ios::binary);
        if (!input) error = std::string(species) + " mesh file is not readable: " + filename;
    }
    reconcile_local_error(error, "mesh input preflight");
}

void preflight_output_directory(const Options& configuration)
{
    if (configuration.checkpoint_cadence == 0)
    {
        reconcile_local_error("", "output directory preflight");
        return;
    }
    int rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    std::string error;
    std::error_code filesystem_error;
    std::filesystem::create_directories(configuration.output_dir, filesystem_error);
    if (filesystem_error)
        error = "cannot create output directory " + configuration.output_dir + ": " + filesystem_error.message();
    else
    {
        const auto probe = std::filesystem::path(configuration.output_dir) /
                           (".karma-write-probe-rank-" + std::to_string(rank));
        std::ofstream output(probe, std::ios::binary | std::ios::trunc);
        if (!output) error = "output directory is not writable: " + configuration.output_dir;
        else
        {
            output << "probe";
            output.close();
            if (!output) error = "output directory write probe failed: " + configuration.output_dir;
            std::filesystem::remove(probe, filesystem_error);
            if (error.empty() && filesystem_error)
                error = "output directory probe cleanup failed: " + filesystem_error.message();
        }
    }
    reconcile_local_error(error, "output directory preflight");
}

KARMA::OperatorSplitGraphSpec graph(KARMA::SplitScheme scheme)
{
    const auto make_component = [](const std::string& id) {
        const KARMA::FieldSpec output{id + ".state", "1", 1, KARMA::FieldRole::Prognostic};
        const KARMA::FieldSpec input{id + ".coupled", "1", 1, KARMA::FieldRole::Input};
        return KARMA::ComponentSpec{id, {{"state", KARMA::PortDirection::Output, false, output},
                                          {"coupled", KARMA::PortDirection::Input, true, input}}, true, true};
    };
    return {"gray-scott-example", {{{"U", make_component("U")}, {"V", make_component("V")}}},
            {{{"U", "state", "V", "coupled"}, {"V", "state", "U", "coupled"}}}, scheme};
}

std::unique_ptr<KARMA::KARMAMesh> make_mesh(const std::string& filename,
                                            std::array<unsigned, 2> generated)
{
    if (!filename.empty()) return std::make_unique<KARMA::KARMAMesh>(2, filename, PETSC_COMM_WORLD);
    auto mesh = std::make_unique<KARMA::KARMAMesh>(2, PETSC_COMM_WORLD);
    mesh->generate_box(generated[0], generated[1], 0);
    return mesh;
}

struct FieldDiagnostics
{
    PetscReal minimum = 0;
    PetscReal maximum = 0;
    PetscReal norm = 0;
    PetscReal mass = 0;
};

FieldDiagnostics field_diagnostics(KARMA::TransientReactionDiffusionNativeAdapter& component)
{
    Vec state = component.committed_solution();
    PetscInt index = 0;
    FieldDiagnostics result;
    if (VecMin(state, &index, &result.minimum) != PETSC_SUCCESS ||
        VecMax(state, &index, &result.maximum) != PETSC_SUCCESS ||
        VecNorm(state, NORM_2, &result.norm) != PETSC_SUCCESS)
        throw std::runtime_error("failed to compute field extrema or norm");
    Vec weighted = nullptr;
    if (VecDuplicate(state, &weighted) != PETSC_SUCCESS) throw std::runtime_error("failed to allocate mass work vector");
    const auto cleanup = [&] { VecDestroy(&weighted); };
    if (MatMult(component.mass(), state, weighted) != PETSC_SUCCESS || VecSum(weighted, &result.mass) != PETSC_SUCCESS)
    {
        cleanup();
        throw std::runtime_error("failed to compute field mass");
    }
    cleanup();
    return result;
}

void write_checkpoint(const Options& configuration, const KARMA::CompiledOperatorSplitGraph& compiled,
                      KARMA::KARMACouplingScheduler& scheduler,
                      KARMA::TransientReactionDiffusionNativeAdapter& u,
                      KARMA::TransientReactionDiffusionNativeAdapter& v)
{
    const auto stem = configuration.output_dir + "/step-" + std::to_string(scheduler.position().step);
    KARMA::write_operator_split_checkpoint_bundle(
        {stem + "-U.h5m", stem + "-V.h5m", stem + ".karma"}, compiled, {&u, &v}, scheduler.position());
}
}  // namespace

int main(int argc, char** argv)
{
    static constexpr const char help[] =
        "Spatial Gray-Scott operator-split demonstration.\n"
        "  -gs_scheme <lie|strang> -gs_dt <real> -gs_final_time <real>\n"
        "  -gs_diffusion_u/-gs_diffusion_v/-gs_feed/-gs_kill <real>\n"
        "  -gs_perturbation/-gs_perturbation_radius <real>\n"
        "  -gs_u_mesh/-gs_v_mesh <h5m> or -gs_u_resolution/-gs_v_resolution NX,NY\n"
        "  -gs_output_dir <path> -gs_checkpoint_cadence <steps; 0 disables>\n";
    if (PetscInitialize(&argc, &argv, nullptr, help) != PETSC_SUCCESS) return 1;
    int status = 0;
    try
    {
        PetscBool petsc_help_requested = PETSC_FALSE;
        std::string parse_error;
        Options configuration;
        try
        {
            if (PetscOptionsHasName(nullptr, nullptr, "-help", &petsc_help_requested) != PETSC_SUCCESS)
                throw std::runtime_error("failed to query -help");
            configuration = options();
        }
        catch (const std::exception& error) { parse_error = error.what(); }
        reconcile_local_error(parse_error, "option parsing");
        const bool help_requested = petsc_help_requested == PETSC_TRUE;
        require_matching_options(configuration, help_requested);
        if (help_requested)
        {
            PetscFinalize();
            return 0;
        }
        preflight_mesh_file(configuration.u_mesh, "U");
        preflight_mesh_file(configuration.v_mesh, "V");
        preflight_output_directory(configuration);
        auto u_mesh = make_mesh(configuration.u_mesh, configuration.u_resolution);
        auto v_mesh = make_mesh(configuration.v_mesh, configuration.v_resolution);
        KARMA::TransientReactionDiffusionNativeAdapter u("U", *u_mesh, KARMA::GrayScottSpecies::U,
                                                         configuration.parameters, 1.0);
        KARMA::TransientReactionDiffusionNativeAdapter v("V", *v_mesh, KARMA::GrayScottSpecies::V,
                                                         configuration.parameters, 0.0);
        const auto compiled = KARMA::compile_operator_split_graph(graph(configuration.scheme), PETSC_COMM_WORLD);
        KARMA::KARMACouplingScheduler scheduler(compiled, {&u, &v});
        const double amplitude = configuration.perturbation;
        const double radius_squared = configuration.perturbation_radius * configuration.perturbation_radius;
        const auto inside = [radius_squared](const KARMA::Point& point) {
            const double dx = point[0] - 0.5;
            const double dy = point[1] - 0.5;
            return dx * dx + dy * dy <= radius_squared;
        };
        u.set_initial_condition([=](const KARMA::Point& point) { return inside(point) ? 1.0 - amplitude : 1.0; });
        v.set_initial_condition([=](const KARMA::Point& point) { return inside(point) ? amplitude : 0.0; });
        int rank = 0;
        MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
        if (rank == 0)
            std::cout << "Gray-Scott " << (configuration.scheme == KARMA::SplitScheme::Lie ? "Lie" : "Strang")
                      << ": U mesh=" << (configuration.u_mesh.empty() ? "generated" : configuration.u_mesh)
                      << ", V mesh=" << (configuration.v_mesh.empty() ? "generated" : configuration.v_mesh)
                      << ", dt=" << configuration.dt << ", final-time=" << configuration.final_time
                      << ", Du=" << configuration.parameters.diffusion_u
                      << ", Dv=" << configuration.parameters.diffusion_v
                      << ", feed=" << configuration.parameters.feed_rate
                      << ", kill=" << configuration.parameters.kill_rate << '\n';

        const auto total_started = std::chrono::steady_clock::now();
        const auto steps = static_cast<std::size_t>(std::llround(configuration.final_time / configuration.dt));
        for (std::size_t step = 0; step < steps; ++step)
        {
            const auto started = std::chrono::steady_clock::now();
            scheduler.advance(step * configuration.dt, (step + 1) * configuration.dt);
            if (configuration.checkpoint_cadence > 0 &&
                scheduler.position().step % static_cast<std::size_t>(configuration.checkpoint_cadence) == 0)
                write_checkpoint(configuration, compiled, scheduler, u, v);
            const auto u_field = field_diagnostics(u);
            const auto v_field = field_diagnostics(v);
            const auto transfers = scheduler.connection_diagnostics();
            if (rank == 0)
            {
                std::cout << std::setprecision(8) << "step=" << scheduler.position().step
                          << " time=" << scheduler.position().time
                          << " U[min,max,norm,mass]=[" << u_field.minimum << ',' << u_field.maximum << ','
                          << u_field.norm << ',' << u_field.mass << "] V[min,max,norm,mass]=["
                          << v_field.minimum << ',' << v_field.maximum << ',' << v_field.norm << ',' << v_field.mass
                          << "] SNES[U,V]=[" << static_cast<int>(u.diagnostics().reason) << ','
                          << static_cast<int>(v.diagnostics().reason) << "] transfer-residual=["
                          << transfers[0].residual_norm << ',' << transfers[1].residual_norm << "] coverage=["
                          << transfers[0].covered_points << '/' << transfers[0].total_points << ','
                          << transfers[1].covered_points << '/' << transfers[1].total_points << "] timing[scheduler,U,V]=["
                          << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() << ','
                          << u.diagnostics().solve_seconds << ',' << v.diagnostics().solve_seconds << "]\n";
            }
        }
        if (configuration.checkpoint_cadence > 0 &&
            scheduler.position().step % static_cast<std::size_t>(configuration.checkpoint_cadence) != 0)
            write_checkpoint(configuration, compiled, scheduler, u, v);
        if (rank == 0)
        {
            const auto transfers = scheduler.connection_diagnostics();
            std::cout << "completed committed-time=" << scheduler.position().time
                      << " steps=" << scheduler.position().step << " transfer-KSP=["
                      << static_cast<int>(transfers[0].solver_reason) << ','
                      << static_cast<int>(transfers[1].solver_reason) << "] setup/apply-seconds=["
                      << transfers[0].setup_seconds + transfers[1].setup_seconds << ','
                      << transfers[0].apply_seconds + transfers[1].apply_seconds << "] total-seconds="
                      << std::chrono::duration<double>(std::chrono::steady_clock::now() - total_started).count() << '\n';
        }
    }
    catch (const std::exception& error)
    {
        int rank = 0;
        MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
        if (rank == 0) std::cerr << "Gray-Scott option/runtime error: " << error.what() << '\n';
        status = 1;
    }
    PetscFinalize();
    return status;
}
