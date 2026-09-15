#include "karma_collective_outcome.h"
#include "karma_fixed_point.h"

#include "petsc_raii.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace KARMA
{
namespace
{
void check(PetscErrorCode error, const char* operation)
{
    if (error != PETSC_SUCCESS) throw std::runtime_error(std::string("PETSc error in ") + operation);
}

/** The agreed failure message for this epoch, or empty when no rank failed.
 *
 * A nonempty local message IS the failure here, which is why the predicate is
 * `!local.empty()`. The reduction rule itself was this file's own fifth copy
 * and is now the shared primitive's.
 */
std::string collective_message(MPI_Comm communicator, const std::string& local)
{
    return reconcile_failure_message(communicator, !local.empty(), local);
}

std::string exception_message(const char* fallback)
{
    try { throw; }
    catch (const std::exception& error) { return error.what(); }
    catch (...) { return fallback; }
}

void require_collective_match(MPI_Comm communicator, const std::string& signature, const char* message)
{
    int size = 0;
    int rank = 0;
    check(MPI_Comm_size(communicator, &size), "MPI_Comm_size(fixed-point signature)");
    check(MPI_Comm_rank(communicator, &rank), "MPI_Comm_rank(fixed-point signature)");
    const int local_size = static_cast<int>(signature.size());
    std::vector<int> sizes(static_cast<std::size_t>(size));
    check(MPI_Allgather(&local_size, 1, MPI_INT, sizes.data(), 1, MPI_INT, communicator),
          "MPI_Allgather(fixed-point signature sizes)");
    std::vector<int> displacements(static_cast<std::size_t>(size));
    int total = 0;
    for (int index = 0; index < size; ++index)
    {
        displacements[static_cast<std::size_t>(index)] = total;
        total += sizes[static_cast<std::size_t>(index)];
    }
    std::string gathered(static_cast<std::size_t>(total), '\0');
    check(MPI_Allgatherv(signature.data(), local_size, MPI_CHAR, gathered.data(), sizes.data(),
                         displacements.data(), MPI_CHAR, communicator),
          "MPI_Allgatherv(fixed-point signatures)");
    int failing_rank = size;
    for (int index = 1; index < size; ++index)
        if (sizes[static_cast<std::size_t>(index)] != sizes[0] ||
            gathered.compare(static_cast<std::size_t>(displacements[static_cast<std::size_t>(index)]),
                             static_cast<std::size_t>(sizes[static_cast<std::size_t>(index)]),
                             gathered, 0, static_cast<std::size_t>(sizes[0])) != 0)
        {
            failing_rank = index;
            break;
        }
    if (failing_rank != size)
        throw std::invalid_argument("rank " + std::to_string(failing_rank) + ": " + message);
}

std::string options_signature(const FixedPointOptions& options)
{
    std::ostringstream stream;
    stream << options.maximum_iterations << '|' << std::hexfloat << options.absolute_tolerance << '|'
           << options.relative_tolerance << '|' << options.initial_relaxation << '|' << options.use_aitken << '|'
           << options.divergence_factor << '|' << options.minimum_relaxation << '|' << options.maximum_relaxation;
    return stream.str();
}

struct OwnedIterate
{
    std::vector<PetscOwner<Vec, VecDestroy>> vectors;
    KARMAFixedPointIterator::Iterate fields;
};

OwnedIterate duplicate(const KARMAFixedPointIterator::Iterate& prototype, bool copy_values)
{
    OwnedIterate result;
    result.vectors.resize(prototype.size());
    result.fields.reserve(prototype.size());
    for (std::size_t index = 0; index < prototype.size(); ++index)
    {
        check(VecDuplicate(prototype[index].values, result.vectors[index].put()), "VecDuplicate(fixed-point iterate)");
        if (copy_values) check(VecCopy(prototype[index].values, result.vectors[index].get()), "VecCopy(fixed-point iterate)");
        result.fields.push_back({prototype[index].id, result.vectors[index].get(), prototype[index].mass,
                                 prototype[index].tolerances});
    }
    return result;
}

Real mass_norm(Vec value, Mat mass, Vec work)
{
    check(MatMult(mass, value, work), "MatMult(fixed-point metric)");
    PetscScalar product = 0;
    check(VecDot(value, work, &product), "VecDot(fixed-point metric)");
    const Real real_product = PetscRealPart(product);
    if (!std::isfinite(real_product) || real_product < -64 * std::numeric_limits<Real>::epsilon())
        return std::numeric_limits<Real>::quiet_NaN();
    return std::sqrt(std::max<Real>(0, real_product));
}

Real mass_rms(Vec value, Mat mass, Vec work, Real measure)
{
    return mass_norm(value, mass, work) / std::sqrt(measure);
}

bool restore_collectively(MPI_Comm communicator, const KARMAFixedPointIterator::Restore& restore,
                          std::string& message)
{
    std::string local;
    try
    {
        if (!restore()) local = "fixed-point checkpoint restore failed";
    }
    catch (...) { local = exception_message("unknown fixed-point restore failure"); }
    message = collective_message(communicator, local);
    return message.empty();
}
}  // namespace

KARMAFixedPointIterator::KARMAFixedPointIterator(MPI_Comm communicator, FixedPointOptions options)
    : communicator_(communicator), options_(options)
{
    int initialized = 0;
    if (MPI_Initialized(&initialized) != MPI_SUCCESS || !initialized || communicator_ == MPI_COMM_NULL)
        throw std::invalid_argument("fixed-point iterator requires initialized MPI and a valid communicator");
    const bool invalid = options_.maximum_iterations == 0 || !std::isfinite(options_.absolute_tolerance) ||
        !std::isfinite(options_.relative_tolerance) || !std::isfinite(options_.initial_relaxation) ||
        !std::isfinite(options_.divergence_factor) || !std::isfinite(options_.minimum_relaxation) ||
        !std::isfinite(options_.maximum_relaxation) || options_.absolute_tolerance < 0 ||
        options_.relative_tolerance < 0 || options_.initial_relaxation <= 0 || options_.divergence_factor <= 1 ||
        options_.minimum_relaxation <= 0 || options_.minimum_relaxation > options_.maximum_relaxation;
    const auto invalid_message = collective_message(communicator_, invalid ? "fixed-point options are invalid" : "");
    if (!invalid_message.empty()) throw std::invalid_argument(invalid_message);
    require_collective_match(communicator_, options_signature(options_), "fixed-point options differ across ranks");
}

FixedPointResult KARMAFixedPointIterator::solve(const Iterate& initial, const Checkpoint& checkpoint,
                                                 const Restore& restore, const Sweep& sweep,
                                                 const Commit& commit) const
{
    if (!checkpoint || !restore || !sweep || !commit) throw std::invalid_argument("fixed-point callbacks are required");
    std::string local_validation;
    std::ostringstream metadata;
    try
    {
        if (initial.empty()) throw std::invalid_argument("fixed-point iterate bundle is empty");
        metadata << initial.size();
        for (std::size_t index = 0; index < initial.size(); ++index)
        {
            const auto& field = initial[index];
            if (field.id.empty() || field.values == nullptr || field.mass == nullptr)
                throw std::invalid_argument("fixed-point fields require an id, values, and mass matrix");
            if (std::find_if(initial.begin(), initial.begin() + static_cast<std::ptrdiff_t>(index),
                             [&](const auto& previous) { return previous.id == field.id; }) !=
                initial.begin() + static_cast<std::ptrdiff_t>(index))
                throw std::invalid_argument("fixed-point field ids must be unique");
            const Real absolute = field.tolerances.absolute.value_or(options_.absolute_tolerance);
            const Real relative = field.tolerances.relative.value_or(options_.relative_tolerance);
            if (!std::isfinite(absolute) || !std::isfinite(relative) || absolute < 0 || relative < 0)
                throw std::invalid_argument("fixed-point field tolerances are invalid");
            MPI_Comm vector_comm = PetscObjectComm(reinterpret_cast<PetscObject>(field.values));
            MPI_Comm matrix_comm = PetscObjectComm(reinterpret_cast<PetscObject>(field.mass));
            int vector_relation = MPI_UNEQUAL;
            int matrix_relation = MPI_UNEQUAL;
            check(MPI_Comm_compare(communicator_, vector_comm, &vector_relation), "MPI_Comm_compare(fixed-point vector)");
            check(MPI_Comm_compare(communicator_, matrix_comm, &matrix_relation), "MPI_Comm_compare(fixed-point matrix)");
            if ((vector_relation != MPI_IDENT && vector_relation != MPI_CONGRUENT) ||
                (matrix_relation != MPI_IDENT && matrix_relation != MPI_CONGRUENT))
                throw std::invalid_argument("fixed-point field communicator differs from iterator communicator");
            PetscInt vector_size = 0, vector_local = 0, rows = 0, columns = 0, local_rows = 0, local_columns = 0;
            check(VecGetSize(field.values, &vector_size), "VecGetSize(fixed-point field)");
            check(VecGetLocalSize(field.values, &vector_local), "VecGetLocalSize(fixed-point field)");
            check(MatGetSize(field.mass, &rows, &columns), "MatGetSize(fixed-point mass matrix)");
            check(MatGetLocalSize(field.mass, &local_rows, &local_columns), "MatGetLocalSize(fixed-point mass matrix)");
            if (rows != vector_size || columns != vector_size || local_rows != vector_local || local_columns != vector_local)
                throw std::invalid_argument("fixed-point mass matrix dimensions or layout differ from field vector");
            metadata << '|' << field.id << '|' << vector_size << '|' << std::hexfloat << absolute << '|' << relative;
        }
    }
    catch (...) { local_validation = exception_message("unknown fixed-point metadata validation failure"); }
    if (const auto message = collective_message(communicator_, local_validation); !message.empty())
        throw std::invalid_argument(message);
    require_collective_match(communicator_, metadata.str(), "fixed-point field metadata differ across ranks");

    FixedPointResult result;
    result.relaxation = options_.initial_relaxation;
    OwnedIterate current;
    OwnedIterate mapped;
    OwnedIterate residual;
    OwnedIterate previous_residual;
    OwnedIterate difference;
    OwnedIterate work;
    OwnedIterate ones;
    std::vector<Real> measures(initial.size());
    local_validation.clear();
    try
    {
        current = duplicate(initial, true);
        mapped = duplicate(initial, false);
        residual = duplicate(initial, false);
        previous_residual = duplicate(initial, false);
        difference = duplicate(initial, false);
        work = duplicate(initial, false);
        ones = duplicate(initial, false);
        for (std::size_t field = 0; field < initial.size(); ++field)
        {
            check(VecSet(ones.fields[field].values, 1), "VecSet(fixed-point measure vector)");
            measures[field] = std::pow(mass_norm(ones.fields[field].values, initial[field].mass,
                                                work.fields[field].values), 2);
            if (!std::isfinite(measures[field]) || measures[field] <= 0)
                throw std::invalid_argument("fixed-point mass matrix has nonpositive or nonfinite domain measure");
        }
    }
    catch (...) { local_validation = exception_message("unknown fixed-point workspace preparation failure"); }
    if (const auto message = collective_message(communicator_, local_validation); !message.empty())
        throw std::invalid_argument(message);
    std::string local;
    try { checkpoint(); }
    catch (...) { local = exception_message("unknown fixed-point checkpoint failure"); }
    if (const auto message = collective_message(communicator_, local); !message.empty())
    {
        result.status = FixedPointStatus::SweepFailure;
        result.message = message;
        return result;
    }

    bool have_previous = false;
    Real previous_norm = 0;

    const auto finish_failure = [&](FixedPointStatus status, std::size_t iteration, Real norm,
                                    std::string message) {
        result.status = status;
        result.iterations = iteration;
        result.residual = norm;
        std::string rollback_message;
        if (!restore_collectively(communicator_, restore, rollback_message))
        {
            result.status = FixedPointStatus::RollbackFailure;
            if (!message.empty()) message += "; additionally ";
            message += rollback_message;
        }
        result.message = std::move(message);
        return result;
    };

    try
    {
      for (std::size_t iteration = 1; iteration <= options_.maximum_iterations; ++iteration)
      {
        std::string restore_message;
        if (!restore_collectively(communicator_, restore, restore_message))
            return finish_failure(FixedPointStatus::RollbackFailure, iteration, result.residual, restore_message);

        local.clear();
        try { sweep(current.fields, mapped.fields); }
        catch (...) { local = exception_message("unknown fixed-point sweep failure"); }
        if (const auto message = collective_message(communicator_, local); !message.empty())
            return finish_failure(FixedPointStatus::SweepFailure, iteration, result.residual, message);

        result.fields.clear();
        Real combined_squared = 0;
        bool nonfinite = false;
        for (std::size_t field = 0; field < initial.size(); ++field)
        {
            check(VecWAXPY(residual.fields[field].values, -1, current.fields[field].values,
                           mapped.fields[field].values), "VecWAXPY(fixed-point residual)");
            const Real field_residual = mass_rms(residual.fields[field].values, initial[field].mass,
                                                 work.fields[field].values, measures[field]);
            const Real field_scale = mass_rms(mapped.fields[field].values, initial[field].mass,
                                              work.fields[field].values, measures[field]);
            const Real absolute = initial[field].tolerances.absolute.value_or(options_.absolute_tolerance);
            const Real relative = initial[field].tolerances.relative.value_or(options_.relative_tolerance);
            const Real tolerance = absolute + relative * field_scale;
            nonfinite = nonfinite || !std::isfinite(field_residual) || !std::isfinite(field_scale) ||
                        !std::isfinite(tolerance);
            result.fields.push_back({initial[field].id, field_residual, field_scale, tolerance});
            combined_squared += field_residual * field_residual;
        }
        const Real norm = std::sqrt(combined_squared);
        result.iterations = iteration;
        result.residual = norm;
        if (nonfinite || !std::isfinite(norm))
            return finish_failure(FixedPointStatus::Nonfinite, iteration, norm, "fixed-point iterate or metric is nonfinite");
        if (iteration > 1 && norm > options_.divergence_factor * previous_norm)
            return finish_failure(FixedPointStatus::Diverged, iteration, norm, "fixed-point residual diverged");
        const bool converged = std::all_of(result.fields.begin(), result.fields.end(),
                                           [](const auto& metric) { return metric.residual <= metric.tolerance; });
        if (converged)
        {
            local.clear();
            try { commit(mapped.fields); }
            catch (...) { local = exception_message("unknown fixed-point commit failure"); }
            if (const auto message = collective_message(communicator_, local); !message.empty())
                return finish_failure(FixedPointStatus::SweepFailure, iteration, norm, message);
            result.status = FixedPointStatus::Converged;
            return result;
        }

        if (options_.use_aitken && have_previous)
        {
            Real numerator = 0;
            Real denominator = 0;
            for (std::size_t field = 0; field < initial.size(); ++field)
            {
                check(VecWAXPY(difference.fields[field].values, -1, previous_residual.fields[field].values,
                               residual.fields[field].values), "VecWAXPY(Aitken difference)");
                check(MatMult(initial[field].mass, difference.fields[field].values, work.fields[field].values),
                      "MatMult(Aitken difference)");
                PetscScalar local_numerator = 0;
                PetscScalar local_denominator = 0;
                check(VecDot(previous_residual.fields[field].values, work.fields[field].values, &local_numerator),
                      "VecDot(Aitken numerator)");
                check(VecDot(difference.fields[field].values, work.fields[field].values, &local_denominator),
                      "VecDot(Aitken denominator)");
                numerator += PetscRealPart(local_numerator);
                denominator += PetscRealPart(local_denominator);
            }
            if (std::isfinite(numerator) && std::isfinite(denominator) && denominator > 0)
                result.relaxation = std::clamp(-result.relaxation * numerator / denominator,
                                               options_.minimum_relaxation, options_.maximum_relaxation);
        }
        for (std::size_t field = 0; field < initial.size(); ++field)
        {
            check(VecAXPY(current.fields[field].values, result.relaxation, residual.fields[field].values),
                  "VecAXPY(fixed-point relaxation)");
            check(VecCopy(residual.fields[field].values, previous_residual.fields[field].values),
                  "VecCopy(previous fixed-point residual)");
        }
        have_previous = true;
        previous_norm = norm;
      }
    }
    catch (...)
    {
        local = exception_message("unknown fixed-point post-checkpoint failure");
        if (const auto message = collective_message(communicator_, local); !message.empty())
            return finish_failure(FixedPointStatus::SweepFailure, result.iterations, result.residual, message);
    }
    return finish_failure(FixedPointStatus::MaxIterations, options_.maximum_iterations, result.residual,
                          "fixed-point iteration limit reached");
}
}  // namespace KARMA
