#ifndef KARMA_COUPLING_COMPILED_FEM_CONNECTION_H
#define KARMA_COUPLING_COMPILED_FEM_CONNECTION_H

#include "karma_field.h"

#include <cstddef>
#include <memory>

#include <petscksp.h>

namespace KARMA
{
class KARMASystem;

struct FEMConnectionDiagnostics
{
    KSPConvergedReason solver_reason = KSP_CONVERGED_ITERATING;
    Real residual_norm = 0;
    Real residual_scale = 0;
    Real residual_tolerance = 0;
    std::size_t covered_points = 0;
    std::size_t total_points = 0;
    double setup_seconds = 0;
    double apply_seconds = 0;
};

/**
 * Persistent scalar cG FEM transfer runtime.
 *
 * setup() compiles geometry-dependent data and PETSc solver objects. apply()
 * accepts an updated source field binding but requires the source system,
 * variable layout, target system, basis, mesh, and communicator to remain
 * unchanged. Mesh or layout changes are not detected automatically; callers
 * must invoke setup() again after such changes.
 */
class CompiledFEMConnection
{
  public:
    CompiledFEMConnection(KARMAField source, const KARMASystem& target_system,
                          unsigned target_variable, int quadrature_degree = 5);
    ~CompiledFEMConnection();
    CompiledFEMConnection(const CompiledFEMConnection&) = delete;
    CompiledFEMConnection& operator=(const CompiledFEMConnection&) = delete;
    CompiledFEMConnection(CompiledFEMConnection&&) noexcept;
    CompiledFEMConnection& operator=(CompiledFEMConnection&&) noexcept;

    void setup();
    void apply(const KARMAField& source);

    [[nodiscard]] Vec staging() const noexcept;
    [[nodiscard]] std::size_t setup_count() const noexcept { return setup_count_; }

    // Stable-handle diagnostics used by allocation-regression tests.
    [[nodiscard]] const void* mass_handle() const noexcept;
    [[nodiscard]] const void* rhs_handle() const noexcept;
    [[nodiscard]] const void* solution_handle() const noexcept;
    [[nodiscard]] const void* residual_handle() const noexcept;
    [[nodiscard]] const void* ksp_handle() const noexcept;
    [[nodiscard]] std::size_t located_point_count() const noexcept;
    [[nodiscard]] const void* located_points_handle() const noexcept;
    [[nodiscard]] const FEMConnectionDiagnostics& diagnostics() const noexcept { return diagnostics_; }

  private:
    class Impl;
    KARMAField source_;
    const KARMASystem* target_system_ = nullptr;
    unsigned target_variable_ = 0;
    int quadrature_degree_ = 5;
    std::unique_ptr<Impl> impl_;
    std::size_t setup_count_ = 0;
    FEMConnectionDiagnostics diagnostics_;
};
}  // namespace KARMA

#endif
