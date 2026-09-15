#include "karma_operator_split_graph.h"

#include <petsc.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
KARMA::ComponentSpec component_spec(const std::string& id);

struct FakeComponent final : KARMA::KARMANativeComponent
{
    explicit FakeComponent(std::string name, MPI_Comm communicator)
        : name(std::move(name)), communicator_(communicator)
    {
        component = component_spec(this->name);
        component.supports_fractional_steps = true;
        component.supports_checkpoint_restore = true;
    }

    const KARMA::ComponentSpec& specification() const noexcept override { return component; }
    MPI_Comm communicator() const noexcept override { return communicator_; }
    void initialize() override { initialized = true; events.push_back(name + ":init"); }
    KARMA::KARMAField input_field(KARMA::PortId) override { throw std::logic_error("fake field unavailable"); }
    KARMA::KARMAField output_field(KARMA::PortId, KARMA::FieldVisibility) override
    { throw std::logic_error("fake field unavailable"); }
    void checkpoint() override { checkpoint_value = value; checkpoint_time = time; events.push_back(name + ":checkpoint"); }
    void bind_trial_input(KARMA::PortId, Vec) override { input_bound = true; }
    void advance(const KARMA::TimeInterval& interval) override
    {
        if (fail_advance && name == "V") throw std::runtime_error("injected advance failure");
        intervals.push_back(interval);
        value += interval.end - interval.begin + (input_bound ? 1.0 : 0.0);
        time = interval.end;
        events.push_back(name + ":advance");
    }
    void commit() override
    {
        if (fail_commit && name == "V") throw std::runtime_error("injected commit failure");
        committed_value = value; committed_time = time; events.push_back(name + ":commit");
    }
    void release_checkpoint() noexcept override { events.push_back(name + ":release"); }
    bool restore() noexcept override
    {
        events.push_back(name + ":restore");
        if (fail_restore && name == "V") return false;
        value = checkpoint_value;
        committed_value = checkpoint_value;
        time = checkpoint_time;
        committed_time = checkpoint_time;
        input_bound = false;
        return true;
    }

    std::string name;
    MPI_Comm communicator_;
    KARMA::ComponentSpec component;
    std::vector<std::string> events;
    std::vector<KARMA::TimeInterval> intervals;
    double value = 0;
    double committed_value = 0;
    double checkpoint_value = 0;
    double time = 0;
    double committed_time = 0;
    double checkpoint_time = 0;
    bool initialized = false;
    bool input_bound = false;
    bool fail_advance = false;
    bool fail_commit = false;
    bool fail_restore = false;
};

std::array<FakeComponent*, 2> active_components;

Vec make_staging(const KARMA::ConnectionSpec&, KARMA::KARMANativeComponent&, KARMA::KARMANativeComponent&)
{
    Vec vector = nullptr;
    if (VecCreateMPI(PETSC_COMM_WORLD, 1, PETSC_DECIDE, &vector) != PETSC_SUCCESS) return nullptr;
    return vector;
}

void exchange(const KARMA::ConnectionSpec& connection, KARMA::KARMANativeComponent& source,
              KARMA::KARMANativeComponent& target, KARMA::FieldVisibility visibility, Vec)
{
    auto* source_fake = dynamic_cast<FakeComponent*>(&source);
    auto* target_fake = dynamic_cast<FakeComponent*>(&target);
    if (source_fake == nullptr || target_fake == nullptr) throw std::runtime_error("wrong fake component");
    target_fake->input_bound = true;
    target_fake->events.push_back("exchange:" + connection.source_component + "->" + connection.target_component +
                                  (visibility == KARMA::FieldVisibility::TransactionLocal ? ":local" : ":committed"));
}

KARMA::ComponentSpec component_spec(const std::string& id)
{
    const KARMA::FieldSpec output{id + ".state", "1", 1, KARMA::FieldRole::Prognostic};
    const KARMA::FieldSpec input{id + ".input", "1", 1, KARMA::FieldRole::Input};
    return {id, {{"state", KARMA::PortDirection::Output, false, output},
                 {"input", KARMA::PortDirection::Input, true, input}}, true, true};
}

KARMA::OperatorSplitGraphSpec graph(KARMA::SplitScheme scheme)
{
    KARMA::OperatorSplitGraphSpec result;
    result.id = "scheduler-test";
    result.components = {{{"U", component_spec("U")}, {"V", component_spec("V")}}};
    result.connections = {{{"U", "state", "V", "input"}, {"V", "state", "U", "input"}}};
    result.scheme = scheme;
    return result;
}

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

void run_success(KARMA::SplitScheme scheme)
{
    FakeComponent u("U", PETSC_COMM_WORLD), v("V", PETSC_COMM_WORLD);
    active_components = {&u, &v};
    const auto compiled = KARMA::compile_operator_split_graph(graph(scheme), PETSC_COMM_WORLD);
    KARMA::SchedulerOptions options;
    options.staging_factory = make_staging;
    options.exchange_hook = exchange;
    KARMA::KARMACouplingScheduler scheduler(compiled, {&u, &v}, options);
    scheduler.advance(0.0, 0.4);
    const auto position = scheduler.position();
    require(position.step == 1 && std::abs(position.time - 0.4) < 1.e-14, "scheduler position is wrong");
    require(std::find(u.events.begin(), u.events.end(), "U:commit") != u.events.end() &&
                std::find(v.events.begin(), v.events.end(), "V:commit") != v.events.end(),
            "commit did not execute");
    require(u.intervals.size() == (scheme == KARMA::SplitScheme::Lie ? 1u : 2u), "U interval count is wrong");
    require(v.intervals.size() == 1u, "V interval count is wrong");
    require(std::abs(u.intervals.front().begin - 0.0) < 1.e-14, "U interval start is wrong");
    require(std::abs(u.intervals.front().end - (scheme == KARMA::SplitScheme::Lie ? 0.4 : 0.2)) < 1.e-14,
            "U interval end is wrong");
    require(std::abs(v.intervals.front().begin - 0.0) < 1.e-14 && std::abs(v.intervals.front().end - 0.4) < 1.e-14,
            "V physical interval is wrong");
    if (scheme == KARMA::SplitScheme::Strang)
        require(u.events.end() != std::find(u.events.begin(), u.events.end(), "exchange:V->U:local"),
                "Strang did not expose transaction-local state");
}

void run_restore_failure()
{
    FakeComponent u("U", PETSC_COMM_WORLD), v("V", PETSC_COMM_WORLD);
    v.fail_advance = true;
    v.fail_restore = true;
    const auto compiled = KARMA::compile_operator_split_graph(graph(KARMA::SplitScheme::Lie), PETSC_COMM_WORLD);
    KARMA::SchedulerOptions options;
    options.staging_factory = make_staging;
    options.exchange_hook = exchange;
    KARMA::KARMACouplingScheduler scheduler(compiled, {&u, &v}, options);
    bool failed = false;
    try { scheduler.advance(0, 1); }
    catch (const std::exception& error)
    {
        const std::string message(error.what());
        failed = message.find("injected advance failure") != std::string::npos &&
                 message.find("restore") != std::string::npos;
    }
    require(failed, "restore failure did not preserve the original step failure");
    require(scheduler.state() == KARMA::SchedulerState::Failed,
            "restore failure did not make the scheduler terminal");
    const auto u_events = u.events.size();
    const auto v_events = v.events.size();
    bool terminal_rejected = false;
    try { scheduler.advance(0, 1); }
    catch (const std::logic_error& error)
    {
        terminal_rejected = std::string(error.what()) ==
                            "scheduler is in a terminal failed state after a rollback failure";
    }
    require(terminal_rejected, "terminal scheduler accepted another advance");
    require(u.events.size() == u_events && v.events.size() == v_events,
            "terminal scheduler touched components during rejected advance");
}

void require_collective_message(const std::string& message, const char* missing_message)
{
    int rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    int length = rank == 0 ? static_cast<int>(message.size()) : 0;
    MPI_Bcast(&length, 1, MPI_INT, 0, PETSC_COMM_WORLD);
    std::string reference(static_cast<std::size_t>(length), '\0');
    if (rank == 0) reference = message;
    MPI_Bcast(reference.data(), length, MPI_CHAR, 0, PETSC_COMM_WORLD);
    require(!message.empty(), missing_message);
    require(message == reference, "collective failure messages differ across ranks");
}

void run_asymmetric_graph_metadata()
{
    int rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    auto specification = graph(KARMA::SplitScheme::Lie);
    if (rank == 1) specification.connections[0].source_port = "missing";
    std::string message;
    try { (void)KARMA::compile_operator_split_graph(specification, PETSC_COMM_WORLD); }
    catch (const std::exception& error) { message = error.what(); }
    require_collective_message(message, "asymmetric graph metadata was not rejected collectively");
    require(message.find("rank 1") != std::string::npos && message.find("unknown port") != std::string::npos,
            "collective graph diagnostic did not identify the malformed rank");
}

void run_asymmetric_valid_graph()
{
    int rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    auto specification = graph(rank == 0 ? KARMA::SplitScheme::Lie : KARMA::SplitScheme::Strang);
    std::string message;
    try { (void)KARMA::compile_operator_split_graph(specification, PETSC_COMM_WORLD); }
    catch (const std::exception& error) { message = error.what(); }
    require_collective_message(message, "rank-different valid graphs were not rejected collectively");
    require(message.find("rank 1") != std::string::npos && message.find("signature") != std::string::npos,
            "collective graph diagnostic did not identify the differing successful rank");
}

void run_asymmetric_runtime_metadata()
{
    int rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    FakeComponent u("U", PETSC_COMM_WORLD), v("V", PETSC_COMM_WORLD);
    if (rank == 1) v.component.ports[0].field.units = "malformed";
    const auto compiled = KARMA::compile_operator_split_graph(graph(KARMA::SplitScheme::Lie), PETSC_COMM_WORLD);
    KARMA::SchedulerOptions options;
    options.staging_factory = make_staging;
    options.exchange_hook = exchange;
    std::string message;
    try { KARMA::KARMACouplingScheduler scheduler(compiled, {&u, &v}, options); }
    catch (const std::exception& error) { message = error.what(); }
    require_collective_message(message, "asymmetric runtime metadata was not rejected collectively");
    require(message.find("rank 1") != std::string::npos && message.find("metadata") != std::string::npos,
            "collective runtime metadata diagnostic did not identify the malformed rank");
    require(u.events.empty() && v.events.empty(), "runtime metadata failure reached component initialization");
}

void run_reordered_runtime_ports()
{
    FakeComponent u("U", PETSC_COMM_WORLD), v("V", PETSC_COMM_WORLD);
    std::reverse(u.component.ports.begin(), u.component.ports.end());
    std::reverse(v.component.ports.begin(), v.component.ports.end());
    const auto compiled = KARMA::compile_operator_split_graph(graph(KARMA::SplitScheme::Lie), PETSC_COMM_WORLD);
    KARMA::SchedulerOptions options;
    options.staging_factory = make_staging;
    options.exchange_hook = exchange;
    KARMA::KARMACouplingScheduler scheduler(compiled, {&u, &v}, options);
    const auto report = scheduler.report();
    require(report.state == KARMA::SchedulerState::Ready && report.schedule.size() == compiled.instructions().size(),
            "runtime report omitted scheduler state or schedule");
    require(report.connections[0].setup_count == 0 && report.connections[0].coverage == "unknown",
            "hook-backed runtime report did not mark unavailable setup/coverage unknown");
}

void run_second_commit_failure()
{
    FakeComponent u("U", PETSC_COMM_WORLD), v("V", PETSC_COMM_WORLD);
    v.fail_commit = true;
    const auto compiled = KARMA::compile_operator_split_graph(graph(KARMA::SplitScheme::Lie), PETSC_COMM_WORLD);
    KARMA::SchedulerOptions options;
    options.staging_factory = make_staging;
    options.exchange_hook = exchange;
    KARMA::KARMACouplingScheduler scheduler(compiled, {&u, &v}, options);
    bool failed = false;
    try { scheduler.advance(0, 1); } catch (const std::exception&) { failed = true; }
    require(failed, "second-component commit failure was not reported");
    require(scheduler.position().step == 0 && scheduler.position().time == 0,
            "second-component commit failure changed position");
    require(u.committed_value == 0 && v.committed_value == 0 && u.time == 0 && v.time == 0,
            "second-component commit failure did not restore both components");
}

void expect_constructor_rejection(std::array<FakeComponent*, 2> components,
                                  const KARMA::CompiledOperatorSplitGraph& compiled)
{
    try
    {
        KARMA::SchedulerOptions options;
        options.staging_factory = make_staging;
        options.exchange_hook = exchange;
        KARMA::KARMACouplingScheduler scheduler(
            compiled, {components[0], components[1]}, options);
    }
    catch (const std::exception&)
    {
        return;
    }
    throw std::runtime_error("invalid runtime scheduler configuration was accepted");
}

void run_runtime_validation()
{
    FakeComponent u("U", PETSC_COMM_WORLD), v("V", PETSC_COMM_WORLD);
    const auto compiled = KARMA::compile_operator_split_graph(graph(KARMA::SplitScheme::Lie), PETSC_COMM_WORLD);
    u.component.supports_checkpoint_restore = false;
    expect_constructor_rejection({&u, &v}, compiled);
    auto missing_capability = graph(KARMA::SplitScheme::Lie);
    missing_capability.components[0].component.supports_checkpoint_restore = false;
    bool compile_rejected = false;
    try { (void)KARMA::compile_operator_split_graph(missing_capability); }
    catch (const std::invalid_argument&) { compile_rejected = true; }
    require(compile_rejected, "compile accepted a component without checkpoint restore");
    u.component.supports_checkpoint_restore = true;
    v.component.id = "wrong";
    expect_constructor_rejection({&u, &v}, compiled);
    v.component.id = "V";
    expect_constructor_rejection({&u, &u}, compiled);
}

void run_invalid_interval(bool asymmetric = false)
{
    int rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    FakeComponent u("U", PETSC_COMM_WORLD), v("V", PETSC_COMM_WORLD);
    const auto compiled = KARMA::compile_operator_split_graph(graph(KARMA::SplitScheme::Lie), PETSC_COMM_WORLD);
    KARMA::SchedulerOptions options;
    options.staging_factory = make_staging;
    options.exchange_hook = exchange;
    KARMA::KARMACouplingScheduler scheduler(compiled, {&u, &v}, options);
    bool failed = false;
    const double left = asymmetric && rank == 1 ? 1.0 : 0.0;
    const double right = asymmetric ? (rank == 0 ? 0.0 : 1.0) : 0.0;
    try { scheduler.advance(left, right); } catch (const std::exception&) { failed = true; }
    require(failed, "invalid interval was not rejected");
    require(u.events.size() == 1 && v.events.size() == 1 &&
                u.events.front() == "U:init" && v.events.front() == "V:init",
            "invalid interval reached a component collective");
}

void run_fixed_point_hook_rejection()
{
    FakeComponent u("U", PETSC_COMM_WORLD), v("V", PETSC_COMM_WORLD);
    const auto compiled = KARMA::compile_operator_split_graph(graph(KARMA::SplitScheme::Lie), PETSC_COMM_WORLD);
    KARMA::SchedulerOptions options;
    options.staging_factory = make_staging;
    options.exchange_hook = exchange;
    KARMA::KARMACouplingScheduler scheduler(compiled, {&u, &v}, options);
    bool rejected = false;
    try { (void)scheduler.advance_fixed_point(0, 1); }
    catch (const std::exception& error)
    {
        rejected = std::string(error.what()).find("exchange hooks") != std::string::npos;
    }
    require(rejected, "fixed-point scheduler accepted an exchange hook");
    require(scheduler.state() == KARMA::SchedulerState::Ready,
            "fixed-point hook rejection left scheduler executing");
    require(std::find(u.events.begin(), u.events.end(), "U:checkpoint") == u.events.end() &&
                std::find(v.events.begin(), v.events.end(), "V:checkpoint") == v.events.end(),
            "fixed-point hook rejection reached checkpoint acquisition");
}

void run_failure(bool asymmetric)
{
    int rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    FakeComponent u("U", PETSC_COMM_WORLD), v("V", PETSC_COMM_WORLD);
    v.fail_advance = asymmetric ? rank == 1 : true;
    const auto compiled = KARMA::compile_operator_split_graph(graph(KARMA::SplitScheme::Lie), PETSC_COMM_WORLD);
    KARMA::SchedulerOptions options;
    options.staging_factory = make_staging;
    options.exchange_hook = exchange;
    KARMA::KARMACouplingScheduler scheduler(compiled, {&u, &v}, options);
    bool failed = false;
    try { scheduler.advance(0, 1); } catch (const std::exception&) { failed = true; }
    require(failed, "injected scheduler failure was not reported");
    require(scheduler.position().step == 0 && scheduler.position().time == 0, "failed step changed position");
    require(u.committed_value == 0 && v.committed_value == 0 && u.time == 0 && v.time == 0,
            "failed step did not restore component state");
}
}  // namespace

int main(int argc, char** argv)
{
    if (PetscInitialize(&argc, &argv, nullptr, nullptr) != PETSC_SUCCESS) return 1;
    int status = 0;
    try
    {
        const std::string mode = argc > 1 ? argv[1] : "";
        if (mode == "--asymmetric-advance-failure") run_failure(true);
        else if (mode == "--asymmetric-interval") run_invalid_interval(true);
        else if (mode == "--asymmetric-graph-metadata") run_asymmetric_graph_metadata();
        else if (mode == "--asymmetric-valid-graph") run_asymmetric_valid_graph();
        else if (mode == "--asymmetric-runtime-metadata") run_asymmetric_runtime_metadata();
        else
        {
            run_success(KARMA::SplitScheme::Lie);
            run_success(KARMA::SplitScheme::Strang);
            run_failure(false);
            run_second_commit_failure();
            run_restore_failure();
            run_runtime_validation();
            run_reordered_runtime_ports();
            run_invalid_interval();
            run_fixed_point_hook_rejection();
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        status = 1;
    }
    PetscFinalize();
    return status;
}
