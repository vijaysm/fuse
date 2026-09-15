#ifndef KARMA_COUPLING_FIXED_POINT_H
#define KARMA_COUPLING_FIXED_POINT_H

#include "karma_types.h"

#include <mpi.h>
#include <petscmat.h>
#include <petscvec.h>

#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace KARMA
{
enum class FixedPointStatus
{
    Converged,
    MaxIterations,
    Diverged,
    Nonfinite,
    SweepFailure,
    RollbackFailure
};

struct FixedPointOptions
{
    std::size_t maximum_iterations = 20;
    Real absolute_tolerance = 1e-10;
    Real relative_tolerance = 1e-8;
    Real initial_relaxation = 1;
    bool use_aitken = true;
    Real divergence_factor = 1e6;
    Real minimum_relaxation = 0.05;
    Real maximum_relaxation = 1.5;
};

/** A distributed iterate field and its consistent finite-element mass matrix. */
struct FixedPointField
{
    struct Tolerances
    {
        std::optional<Real> absolute;
        std::optional<Real> relative;
    };

    std::string id;
    Vec values = nullptr;
    Mat mass = nullptr;
    Tolerances tolerances;
};

struct FixedPointFieldMetric
{
    std::string id;
    Real residual = 0;
    Real scale = 0;
    Real tolerance = 0;
};

struct FixedPointResult
{
    FixedPointStatus status = FixedPointStatus::MaxIterations;
    std::size_t iterations = 0;
    Real residual = 0;
    Real relaxation = 1;
    std::vector<FixedPointFieldMetric> fields;
    std::string message;
};

/**
 * Collective transactional fixed-point driver for distributed PETSc fields.
 * The caller acquires one macro-step checkpoint. The driver restores it before
 * every sweep and publishes only the converged iterate.
 */
class KARMAFixedPointIterator
{
  public:
    using Iterate = std::vector<FixedPointField>;
    using Checkpoint = std::function<void()>;
    using Restore = std::function<bool()>;
    using Sweep = std::function<void(const Iterate&, const Iterate&)>;
    using Commit = std::function<void(const Iterate&)>;

    explicit KARMAFixedPointIterator(MPI_Comm communicator, FixedPointOptions options = {});
    [[nodiscard]] FixedPointResult solve(const Iterate& initial, const Checkpoint& checkpoint,
                                         const Restore& restore, const Sweep& sweep,
                                         const Commit& commit) const;

  private:
    MPI_Comm communicator_ = MPI_COMM_NULL;
    FixedPointOptions options_;
};
}  // namespace KARMA

#endif
