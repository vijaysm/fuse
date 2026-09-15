#include "karma_fixed_point.h"

#include "petsc_raii.h"

#include <petsc.h>

#include <cmath>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
using namespace KARMA;

void check(PetscErrorCode error, const char* operation)
{
    if (error != PETSC_SUCCESS) throw std::runtime_error(operation);
}

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

struct ScalarMetric
{
    PetscOwner<Vec, VecDestroy> value;
    PetscOwner<Mat, MatDestroy> mass;

    explicit ScalarMetric(Real initial, Real weight = 2)
    {
        check(VecCreateMPI(PETSC_COMM_WORLD, 1, PETSC_DECIDE, value.put()), "create scalar iterate");
        check(VecSet(value.get(), initial), "set scalar iterate");
        PetscInt begin = 0;
        PetscInt end = 0;
        check(VecGetOwnershipRange(value.get(), &begin, &end), "get scalar ownership");
        PetscInt global_size = 0;
        check(VecGetSize(value.get(), &global_size), "get scalar size");
        check(MatCreateAIJ(PETSC_COMM_WORLD, 1, 1, global_size, global_size, 1, nullptr, 0, nullptr, mass.put()),
              "create scalar mass");
        for (PetscInt row = begin; row < end; ++row) check(MatSetValue(mass.get(), row, row, weight, INSERT_VALUES), "set mass diagonal");
        check(MatAssemblyBegin(mass.get(), MAT_FINAL_ASSEMBLY), "begin mass assembly");
        check(MatAssemblyEnd(mass.get(), MAT_FINAL_ASSEMBLY), "end mass assembly");
    }
};

FixedPointResult measure_constant_residual(Real measure)
{
    ScalarMetric scalar(0, measure);
    FixedPointOptions options{1, 0, 0, 1, false};
    KARMAFixedPointIterator iterator(PETSC_COMM_WORLD, options);
    return iterator.solve(
        {{"state", scalar.value.get(), scalar.mass.get(), {}}}, [] {}, [] { return true; },
        [](const auto&, const auto& mapped) { check(VecSet(mapped[0].values, 1), "set constant residual"); },
        [](const auto&) {});
}

void verify_rms_and_field_tolerances()
{
    const auto unit_measure = measure_constant_residual(1);
    const auto large_measure = measure_constant_residual(100);
    require(unit_measure.fields.size() == 1 && large_measure.fields.size() == 1,
            "RMS test did not report field metrics");
    require(std::abs(unit_measure.fields[0].residual - 1) < 1e-13 &&
                std::abs(large_measure.fields[0].residual - 1) < 1e-13,
            "constant residual RMS depends on domain measure");

    ScalarMetric first(0, 1);
    ScalarMetric second(0, 1);
    FixedPointOptions options{1, 0.1, 0, 1, false};
    KARMAFixedPointIterator iterator(PETSC_COMM_WORLD, options);
    const auto result = iterator.solve(
        {{"loose", first.value.get(), first.mass.get(), {0.6, std::nullopt}},
         {"strict", second.value.get(), second.mass.get(), {0.4, 0.0}}},
        [] {}, [] { return true; },
        [](const auto&, const auto& mapped) {
            check(VecSet(mapped[0].values, 0.5), "set loose field residual");
            check(VecSet(mapped[1].values, 0.5), "set strict field residual");
        },
        [](const auto&) {});
    require(result.status == FixedPointStatus::MaxIterations && result.fields.size() == 2,
            "per-field tolerance test returned the wrong status");
    require(std::abs(result.fields[0].tolerance - 0.6) < 1e-14 &&
                std::abs(result.fields[1].tolerance - 0.4) < 1e-14,
            "per-field tolerance overrides were not applied");
}

void verify_invalid_matrix_precheckpoint()
{
    ScalarMetric scalar(0, 1);
    PetscOwner<Mat, MatDestroy> wrong_mass;
    PetscInt global_size = 0;
    check(VecGetSize(scalar.value.get(), &global_size), "read scalar size for wrong mass");
    check(MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, global_size + 1, global_size + 1,
                       1, nullptr, 0, nullptr,
                       wrong_mass.put()),
          "create wrong mass matrix");
    int checkpoint_count = 0;
    KARMAFixedPointIterator iterator(PETSC_COMM_WORLD, {});
    bool rejected = false;
    try
    {
        (void)iterator.solve({{"state", scalar.value.get(), wrong_mass.get(), {}}},
                             [&] { ++checkpoint_count; }, [] { return true; },
                             [](const auto&, const auto&) {}, [](const auto&) {});
    }
    catch (const std::exception& error)
    {
        rejected = std::string(error.what()).find("mass matrix") != std::string::npos;
    }
    require(rejected && checkpoint_count == 0, "wrong mass dimensions reached checkpoint acquisition");
}

FixedPointResult run_scalar(const FixedPointOptions& options, Real factor, Real offset,
                            bool nonfinite = false, bool asymmetric_throw = false,
                            bool restore_failure = false, int* restores = nullptr,
                            Real* committed = nullptr)
{
    ScalarMetric scalar(0);
    KARMAFixedPointIterator iterator(PETSC_COMM_WORLD, options);
    return iterator.solve(
        {{"state", scalar.value.get(), scalar.mass.get(), {}}}, [] {},
        [&] {
            if (restores != nullptr) ++*restores;
            return !restore_failure;
        },
        [=](const KARMAFixedPointIterator::Iterate& current, const KARMAFixedPointIterator::Iterate& mapped) {
            int rank = 0;
            MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
            if (asymmetric_throw && rank == 1) throw std::runtime_error("rank-local injected sweep failure");
            PetscScalar input = 0;
            const PetscInt index = rank;
            check(VecGetValues(current[0].values, 1, &index, &input), "read current iterate");
            const PetscScalar output = nonfinite ? std::numeric_limits<PetscReal>::quiet_NaN() : factor * input + offset;
            PetscScalar* local_values = nullptr;
            check(VecGetArray(mapped[0].values, &local_values), "borrow mapped iterate");
            local_values[0] = output;
            check(VecRestoreArray(mapped[0].values, &local_values), "restore mapped iterate");
        },
        [&](const KARMAFixedPointIterator::Iterate& result) {
            if (committed != nullptr)
            {
                PetscReal value = 0;
                check(VecNorm(result[0].values, NORM_INFINITY, &value), "read committed iterate");
                *committed = value;
            }
        });
}

void verify_success_and_failures()
{
    verify_rms_and_field_tolerances();
    verify_invalid_matrix_precheckpoint();
    Real committed = 0;
    int restores = 0;
    const auto converged = run_scalar({80, 1e-12, 1e-12, 1, false}, 0.5, 1, false, false, false,
                                      &restores, &committed);
    require(converged.status == FixedPointStatus::Converged && std::abs(committed - 2) < 1e-9,
            "constant-relaxation fixed point did not converge");
    require(restores == static_cast<int>(converged.iterations), "checkpoint was not restored before every sweep");
    require(converged.fields.size() == 1 && converged.fields[0].id == "state" &&
                converged.fields[0].residual <= converged.fields[0].tolerance,
            "per-field mass metric was not reported");

    const auto aitken = run_scalar({20, 1e-12, 1e-12, 0.5, true}, -0.75, 1);
    require(aitken.status == FixedPointStatus::Converged && aitken.iterations < 20,
            "Aitken relaxation did not converge the oscillatory map");

    int nonfinite_restores = 0;
    const auto nonfinite = run_scalar({4, 0, 0, 1, false}, 0, 0, true, false, false,
                                      &nonfinite_restores);
    require(nonfinite.status == FixedPointStatus::Nonfinite && nonfinite_restores == 2,
            "nonfinite iterate did not roll back deterministically");

    const auto maximum = run_scalar({3, 0, 0, 0.25, false}, 1, 1);
    require(maximum.status == FixedPointStatus::MaxIterations && maximum.iterations == 3,
            "maximum-iteration status is wrong");

    FixedPointOptions divergent_options{8, 0, 0, 1, false};
    divergent_options.divergence_factor = 4;
    const auto diverged = run_scalar(divergent_options, 8, 1);
    require(diverged.status == FixedPointStatus::Diverged, "growing residual was not classified as divergence");

    const auto rollback = run_scalar({2, 0, 0, 1, false}, 1, 1, false, false, true);
    require(rollback.status == FixedPointStatus::RollbackFailure && rollback.message.find("restore") != std::string::npos,
            "rollback failure did not become terminal and preserve diagnostics");
}

void verify_asymmetric_failure()
{
    const auto result = run_scalar({4, 0, 0, 1, false}, 0.5, 1, false, true);
    require(result.status == FixedPointStatus::SweepFailure, "asymmetric sweep failure status is wrong");
    require(result.message.find("rank 1") != std::string::npos &&
                result.message.find("rank-local injected sweep failure") != std::string::npos,
            "asymmetric sweep failure diagnostic is not deterministic and rank-qualified");
}

void verify_asymmetric_metadata(bool options_mismatch)
{
    int rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    ScalarMetric scalar(0, 1);
    FixedPointOptions options;
    if (options_mismatch && rank == 1) options.relative_tolerance *= 2;
    std::string message;
    int checkpoints = 0;
    try
    {
        KARMAFixedPointIterator iterator(PETSC_COMM_WORLD, options);
        const std::string id = !options_mismatch && rank == 1 ? "other" : "state";
        (void)iterator.solve({{id, scalar.value.get(), scalar.mass.get(), {}}}, [&] { ++checkpoints; },
                             [] { return true; }, [](const auto&, const auto&) {}, [](const auto&) {});
    }
    catch (const std::exception& error) { message = error.what(); }
    require(!message.empty() && message.find("rank 1") != std::string::npos,
            "asymmetric fixed-point options/metadata were not rejected collectively");
    require(checkpoints == 0, "asymmetric fixed-point metadata reached checkpoint acquisition");
}
}  // namespace

int main(int argc, char** argv)
{
    if (PetscInitialize(&argc, &argv, nullptr, nullptr) != PETSC_SUCCESS) return 1;
    int status = 0;
    try
    {
        if (argc == 2 && std::string(argv[1]) == "--asymmetric-failure") verify_asymmetric_failure();
        else if (argc == 2 && std::string(argv[1]) == "--asymmetric-options") verify_asymmetric_metadata(true);
        else if (argc == 2 && std::string(argv[1]) == "--asymmetric-metadata") verify_asymmetric_metadata(false);
        else
        {
            require(argc == 1,
                    "usage: fixed_point_coupling_test [--asymmetric-failure|--asymmetric-options|--asymmetric-metadata]");
            verify_success_and_failures();
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "fixed-point test failure: " << error.what() << '\n';
        status = 1;
    }
    PetscFinalize();
    return status;
}
