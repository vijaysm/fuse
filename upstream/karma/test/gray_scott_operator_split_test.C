#include "karma_mesh.h"
#include "karma_operator_split_graph.h"
#include "karma_transient_reaction_diffusion_adapter.h"
#include <petsc.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace KARMA;

OperatorSplitGraphSpec graph(SplitScheme scheme) {
    const auto spec = [](const std::string& id) {
        return ComponentSpec{id, {{"state", PortDirection::Output, false,
                                   {id + ".state", "1", 1, FieldRole::Prognostic}},
                                  {"coupled", PortDirection::Input, true,
                                   {id + ".coupled", "1", 1, FieldRole::Input}}}, true, true};
    };
    return {"gray-scott-test", {{{"U", spec("U")}, {"V", spec("V")}}},
            {{{"U", "state", "V", "coupled"}, {"V", "state", "U", "coupled"}}}, scheme};
}

GrayScottParameters parameters() { GrayScottParameters p; p.diffusion_u = p.diffusion_v = 0; return p; }

double vector_norm(Vec vector) {
    PetscReal result = 0;
    if (VecNorm(vector, NORM_INFINITY, &result) != PETSC_SUCCESS || !std::isfinite(result))
        throw std::runtime_error("nonfinite PETSc state");
    return result;
}

// Temporal isolation uses homogeneous states, so this exact hook removes only
// the nonmatching-mesh transfer cost. It still fills scheduler-owned staging and
// lets the scheduler perform the normal bind/advance/commit sequence.
void exact_constant_exchange(const ConnectionSpec& connection, KARMANativeComponent& source,
                             KARMANativeComponent& target, FieldVisibility visibility, Vec staging) {
    const auto source_field = source.output_field(connection.source_port, visibility);
    const auto& source_binding = std::get<PetscFieldBinding>(source_field.binding());
    const auto source_vector = source_binding.vector;
    const PetscScalar* values = nullptr;
    if (VecGetArrayRead(source_vector, &values) != PETSC_SUCCESS)
        throw std::runtime_error("failed to read constant-transfer source");
    VecRestoreArrayRead(source_vector, &values);
    PetscReal global_sum = 0.0;
    PetscInt global_size = 0;
    if (VecSum(source_vector, &global_sum) != PETSC_SUCCESS || VecGetSize(source_vector, &global_size) != PETSC_SUCCESS ||
        global_size == 0)
        throw std::runtime_error("failed to obtain constant-transfer source value");
    const PetscReal global_value = global_sum / static_cast<PetscReal>(global_size);
    if (VecSet(staging, global_value) != PETSC_SUCCESS)
        throw std::runtime_error("failed to stage exact constant transfer");
    (void)target;
}

std::array<double, 2> homogeneous_reference(double final_time) {
    std::array<double, 2> y{1., .2};
    constexpr double dt = 1.e-5;
    const int steps = static_cast<int>(std::llround(final_time / dt));
    const GrayScottModel model(parameters());
    const auto rhs = [&model](const std::array<double, 2>& x) {
        const auto r = model.reaction(x[0], x[1]);
        return std::array<double, 2>{r.u, r.v};
    };
    for (int n = 0; n < steps; ++n) {
        const auto k1 = rhs(y);
        const auto k2 = rhs({y[0] + .5 * dt * k1[0], y[1] + .5 * dt * k1[1]});
        const auto k3 = rhs({y[0] + .5 * dt * k2[0], y[1] + .5 * dt * k2[1]});
        const auto k4 = rhs({y[0] + dt * k3[0], y[1] + dt * k3[1]});
        for (int i = 0; i < 2; ++i) y[i] += dt * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6;
    }
    return y;
}

std::array<double, 2> run(SplitScheme scheme, double dt, bool exact_constant_transfer = false) {
    KARMAMesh u_mesh(2, PETSC_COMM_WORLD), v_mesh(2, PETSC_COMM_WORLD);
    u_mesh.generate_box(2, 2, 0); v_mesh.generate_box(3, 2, 0);
    TransientReactionDiffusionNativeAdapter u("U", u_mesh, GrayScottSpecies::U, parameters(), 1.0);
    TransientReactionDiffusionNativeAdapter v("V", v_mesh, GrayScottSpecies::V, parameters(), .2);
    const auto compiled = compile_operator_split_graph(graph(scheme), PETSC_COMM_WORLD);
    SchedulerOptions options;
    if (exact_constant_transfer) options.exchange_hook = exact_constant_exchange;
    KARMACouplingScheduler scheduler(compiled, {&u, &v}, options);
    const double final_time = exact_constant_transfer ? .05 : .1;
    const double step_count = final_time / dt;
    if (std::abs(step_count - std::round(step_count)) > 1.e-12)
        throw std::runtime_error("temporal horizon must be divisible by the requested step size");
    double time = 0;
    for (int step = 0; step < static_cast<int>(std::llround(step_count)); ++step) {
        const double next = time + dt;
        scheduler.advance(time, next);
        time = next;
    }
    if (std::abs(scheduler.position().time - final_time) >= 1.e-14)
        throw std::runtime_error("temporal run did not reach final time");
    return {vector_norm(std::get<PetscFieldBinding>(u.output_field("state", FieldVisibility::Committed).binding()).vector),
            vector_norm(std::get<PetscFieldBinding>(v.output_field("state", FieldVisibility::Committed).binding()).vector)};
}

void require(bool condition, const std::string& message) { if (!condition) throw std::runtime_error(message); }
double convergence_rate(double coarse, double fine) { return std::log(coarse / fine) / std::log(2.); }

void one_step() {
    const auto state = run(SplitScheme::Strang, .1);
    require(std::isfinite(state[0]) && std::isfinite(state[1]), "one-step state is nonfinite");
    require(std::abs(state[0] - 1.) > 1.e-12 && std::abs(state[1] - .2) > 1.e-12,
            "one-step adapter state did not change");
}

void temporal(SplitScheme scheme) {
    const auto exact = homogeneous_reference(.05);
    const std::vector<double> steps{.025, .0125, .00625, .003125};
    std::vector<double> errors, rates;
    for (double dt : steps) {
        const auto state = run(scheme, dt, true);
        errors.push_back(std::hypot(state[0] - exact[0], state[1] - exact[1]));
    }
    for (std::size_t i = 1; i < errors.size(); ++i) rates.push_back(convergence_rate(errors[i - 1], errors[i]));
    std::cout << (scheme == SplitScheme::Lie ? "Lie" : "Strang") << " errors";
    for (double e : errors) std::cout << ' ' << e;
    std::cout << " rates";
    for (double r : rates) std::cout << ' ' << r;
    std::cout << '\n';
    const bool decreasing = errors[0] > errors[1] && errors[1] > errors[2] && errors[2] > errors[3];
    const double minimum_rate = scheme == SplitScheme::Lie ? .85 : 1.8;
    require(decreasing, "temporal errors are not decreasing");
    for (double rate : rates)
        require(std::isfinite(rate) && rate >= minimum_rate, "temporal rate below threshold");
}

bool bitwise_equal(Vec left, Vec right) {
    const PetscScalar *a = nullptr, *b = nullptr; PetscInt n = 0;
    VecGetLocalSize(left, &n); VecGetArrayRead(left, &a); VecGetArrayRead(right, &b);
    bool result = true; for (PetscInt i = 0; i < n; ++i) result = result && a[i] == b[i];
    VecRestoreArrayRead(left, &a); VecRestoreArrayRead(right, &b); return result;
}

class FailingAdapter final : public KARMANativeComponent {
  public:
    FailingAdapter(TransientReactionDiffusionNativeAdapter& wrapped, std::string failure)
        : wrapped_(wrapped), failure_(std::move(failure)) {}
    const ComponentSpec& specification() const noexcept override { return wrapped_.specification(); }
    MPI_Comm communicator() const noexcept override { return wrapped_.communicator(); }
    void initialize() override { wrapped_.initialize(); }
    KARMAField input_field(PortId p) override { return wrapped_.input_field(std::move(p)); }
    KARMAField output_field(PortId p, FieldVisibility v) override {
        auto result = wrapped_.output_field(std::move(p), v);
        int rank = 0; MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
        if (failure_ == "output" && rank == 1) throw std::runtime_error("rank-local output field failure");
        if (failure_ == "output-after-checkpoint" && rank == 1 && ++output_calls_ > 1)
            throw std::runtime_error("rank-local post-checkpoint output field failure");
        return result;
    }
    Mat mass_matrix(PortId p) override {
        int rank = 0; MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
        if (failure_ == "mass" && rank == 1) return nullptr;
        return wrapped_.mass_matrix(std::move(p));
    }
    void checkpoint() override { ++checkpoints; wrapped_.checkpoint(); }
    void bind_trial_input(PortId p, Vec x) override {
        if (failure_ == "exchange" && !failed_) { failed_ = true; throw std::runtime_error("decorator exchange failure"); }
        wrapped_.bind_trial_input(std::move(p), x);
    }
    void advance(const TimeInterval& interval) override {
        if (failure_ == "advance" && !failed_) { failed_ = true; throw std::runtime_error("advance failure"); }
        wrapped_.advance(interval);
    }
    void commit() override {
        if (failure_ == "commit" && !failed_) { failed_ = true; throw std::runtime_error("commit failure"); }
        wrapped_.commit();
    }
    void release_checkpoint() noexcept override { wrapped_.release_checkpoint(); }
    bool restore() noexcept override { return wrapped_.restore(); }
  private:
     TransientReactionDiffusionNativeAdapter& wrapped_; std::string failure_; bool failed_ = false;
     int output_calls_ = 0;
  public:
     int checkpoints = 0;
};

void rollback(const std::string& failure) {
    KARMAMesh u_mesh(2, PETSC_COMM_WORLD), v_mesh(2, PETSC_COMM_WORLD);
    u_mesh.generate_box(2, 2, 0); v_mesh.generate_box(3, 2, 0);
    TransientReactionDiffusionNativeAdapter ur("U", u_mesh, GrayScottSpecies::U, parameters(), 1.0);
    TransientReactionDiffusionNativeAdapter vr("V", v_mesh, GrayScottSpecies::V, parameters(), .2);
    FailingAdapter u(ur, failure == "exchange" ? "exchange" : failure == "commit" ? "commit" : "none");
    FailingAdapter v(vr, failure == "advance" ? "advance" : "none");
    const auto compiled = compile_operator_split_graph(graph(SplitScheme::Strang), PETSC_COMM_WORLD);
    KARMACouplingScheduler scheduler(compiled, {&u, &v});
    Vec ub = std::get<PetscFieldBinding>(ur.output_field("state", FieldVisibility::Committed).binding()).system->create_vector();
    Vec vb = std::get<PetscFieldBinding>(vr.output_field("state", FieldVisibility::Committed).binding()).system->create_vector();
    VecCopy(std::get<PetscFieldBinding>(ur.output_field("state", FieldVisibility::Committed).binding()).vector, ub);
    VecCopy(std::get<PetscFieldBinding>(vr.output_field("state", FieldVisibility::Committed).binding()).vector, vb);
    bool failed = false; try { scheduler.advance(0, .01); } catch (const std::exception&) { failed = true; }
    require(failed, "rollback decorator did not fail the macro step");
    require(scheduler.position().step == 0 && scheduler.position().time == 0, "rollback changed scheduler position");
    require(bitwise_equal(std::get<PetscFieldBinding>(ur.output_field("state", FieldVisibility::Committed).binding()).vector, ub), "U state was not restored");
    require(bitwise_equal(std::get<PetscFieldBinding>(vr.output_field("state", FieldVisibility::Committed).binding()).vector, vb), "V state was not restored");
    VecDestroy(&ub); VecDestroy(&vb);
}

void fixed_point_preflight(const std::string& failure) {
    KARMAMesh u_mesh(2, PETSC_COMM_WORLD), v_mesh(2, PETSC_COMM_WORLD);
    u_mesh.generate_box(2, 2, 0); v_mesh.generate_box(3, 2, 0);
    TransientReactionDiffusionNativeAdapter ur("U", u_mesh, GrayScottSpecies::U, parameters(), 1.0);
    TransientReactionDiffusionNativeAdapter vr("V", v_mesh, GrayScottSpecies::V, parameters(), .2);
    FailingAdapter u(ur, failure);
    FailingAdapter v(vr, "none");
    const auto compiled = compile_operator_split_graph(graph(SplitScheme::Lie), PETSC_COMM_WORLD);
    KARMACouplingScheduler scheduler(compiled, {&u, &v});
    bool failed = false;
    std::string message;
    try { (void)scheduler.advance_fixed_point(0, .01); }
    catch (const std::exception& error) { failed = true; message = error.what(); }
    require(failed && message.find("rank 1") != std::string::npos,
            "asymmetric fixed-point preflight failure was not collective and rank-qualified");
    require(u.checkpoints == 0 && v.checkpoints == 0, "fixed-point preflight failure reached checkpoint");
    require(scheduler.state() == SchedulerState::Ready, "fixed-point preflight failure changed scheduler state");
}

void fixed_point_runtime_failure() {
    KARMAMesh u_mesh(2, PETSC_COMM_WORLD), v_mesh(2, PETSC_COMM_WORLD);
    u_mesh.generate_box(2, 2, 0); v_mesh.generate_box(3, 2, 0);
    TransientReactionDiffusionNativeAdapter ur("U", u_mesh, GrayScottSpecies::U, parameters(), 1.0);
    TransientReactionDiffusionNativeAdapter vr("V", v_mesh, GrayScottSpecies::V, parameters(), .2);
    FailingAdapter u(ur, "output-after-checkpoint");
    FailingAdapter v(vr, "none");
    const auto compiled = compile_operator_split_graph(graph(SplitScheme::Lie), PETSC_COMM_WORLD);
    KARMACouplingScheduler scheduler(compiled, {&u, &v});
    const auto result = scheduler.advance_fixed_point(0, .01);
    require(result.status == FixedPointStatus::SweepFailure,
            "post-checkpoint fixed-point output failure returned the wrong status");
    require(result.message.find("rank 1") != std::string::npos &&
                result.message.find("post-checkpoint output field failure") != std::string::npos,
            "post-checkpoint fixed-point failure was not collective and rank-qualified");
    require(u.checkpoints == 1 && v.checkpoints == 1, "runtime failure did not occur after checkpoint acquisition");
    require(scheduler.state() == SchedulerState::Ready && scheduler.position().step == 0,
            "recoverable fixed-point runtime failure changed scheduler state or position");
}

void initial_condition_failure(const std::string& failure) {
    int rank = 0, ranks = 0; MPI_Comm_rank(PETSC_COMM_WORLD, &rank); MPI_Comm_size(PETSC_COMM_WORLD, &ranks);
    KARMAMesh mesh(2, PETSC_COMM_WORLD); mesh.generate_box(3, 2, 0);
    TransientReactionDiffusionNativeAdapter adapter("U", mesh, GrayScottSpecies::U, parameters(), .375);
    adapter.initialize();
    Vec before = nullptr; VecDuplicate(adapter.committed_solution(), &before); VecCopy(adapter.committed_solution(), before);
    bool failed = false;
    std::string message;
    try {
        std::function<Real(const Point&)> callback = [&](const Point& point) {
            if (failure == "callback" && (rank == 1 || (ranks == 1 && point(0) > .4)))
                throw std::runtime_error("injected initial-condition callback failure");
            if (failure == "nonfinite" && rank == 1) return std::numeric_limits<Real>::quiet_NaN();
            return Real(.5 + point(0));
        };
        if (failure == "empty" && rank == 1) callback = {};
        adapter.set_initial_condition(callback);
    } catch (const std::exception& error) { failed = true; message = error.what(); }
    require(failed, "invalid initial-condition callback was accepted");
    if (ranks > 1)
        require(message.find("rank 1") != std::string::npos, "MPI initial-condition failure was not rank-qualified");
    require(bitwise_equal(adapter.committed_solution(), before), "failed initial condition mutated committed state");
    const auto trial = std::get<PetscFieldBinding>(
        adapter.output_field("state", FieldVisibility::TransactionLocal).binding()).vector;
    require(bitwise_equal(trial, before),
            "failed initial condition mutated trial state");
    VecDestroy(&before);
}

} // namespace

int main(int argc, char** argv) {
    if (PetscInitialize(&argc, &argv, nullptr, nullptr) != PETSC_SUCCESS) return 1;
    int result = 0;
    try {
        const std::string mode = argc > 1 ? argv[1] : "one-step";
        if (mode == "one-step") one_step();
        else if (mode == "lie") temporal(SplitScheme::Lie);
        else if (mode == "strang") temporal(SplitScheme::Strang);
        else if (mode == "rollback-exchange" || mode == "rollback-advance" || mode == "rollback-commit") rollback(mode.substr(9));
        else if (mode == "fixed-point-null-mass") fixed_point_preflight("mass");
        else if (mode == "fixed-point-output-failure") fixed_point_preflight("output");
        else if (mode == "fixed-point-runtime-output-failure") fixed_point_runtime_failure();
        else if (mode == "initial-condition-serial") initial_condition_failure("callback");
        else if (mode == "initial-condition-callback-mpi") initial_condition_failure("callback");
        else if (mode == "initial-condition-nonfinite-mpi") initial_condition_failure("nonfinite");
        else if (mode == "initial-condition-empty-mpi") initial_condition_failure("empty");
        else throw std::invalid_argument("unknown Gray-Scott test mode");
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; result = 1; }
    PetscFinalize(); return result;
}
