#ifndef KARMA_COUPLING_OPERATOR_SPLIT_GRAPH_H
#define KARMA_COUPLING_OPERATOR_SPLIT_GRAPH_H

#include "karma_component.h"
#include "karma_compiled_fem_connection.h"
#include "karma_field.h"
#include "karma_fixed_point.h"

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <mpi.h>
#include <petscvec.h>

namespace KARMA
{
enum class SplitScheme
{
    Lie,
    Strang
};

struct ComponentInstanceSpec
{
    std::string id;
    ComponentSpec component;
    std::string mesh_id = "unknown";
    std::string layout_id = "unknown";
    std::string basis_id = "unknown";
    std::string solver_policy_id = "unknown";
};

struct ConnectionSpec
{
    std::string source_component;
    PortId source_port;
    std::string target_component;
    PortId target_port;
    std::string transfer_id = "persistent scalar cG projection";
    std::string solver_policy_id = "PETSc options database";
};

struct OperatorSplitGraphSpec
{
    GraphId id;
    std::array<ComponentInstanceSpec, 2> components;
    std::array<ConnectionSpec, 2> connections;
    SplitScheme scheme = SplitScheme::Lie;
};

enum class InstructionKind
{
    AcquireCheckpoint,
    Exchange,
    Advance,
    Commit,
    ReleaseCheckpoint
};

enum class AdvanceInterval
{
    Full,
    HalfFirst,
    HalfSecond,
    None
};

struct CompiledInstruction
{
    InstructionKind kind = InstructionKind::AcquireCheckpoint;
    std::string component;
    std::string source_component;
    std::string target_component;
    AdvanceInterval interval = AdvanceInterval::None;
};

struct TimeInterval
{
    Real begin = 0;
    Real end = 0;
};

enum class FieldVisibility
{
    Committed,
    TransactionLocal
};

enum class SchedulerState
{
    Constructed,
    Ready,
    Executing,
    Failed,
    Finalized
};

class KARMANativeComponent
{
  public:
    virtual ~KARMANativeComponent() = default;
    virtual const ComponentSpec& specification() const noexcept = 0;
    virtual MPI_Comm communicator() const noexcept = 0;
    virtual void initialize() = 0;
    virtual KARMAField input_field(PortId) = 0;
    virtual KARMAField output_field(PortId, FieldVisibility) = 0;
    virtual void checkpoint() = 0;
    virtual void bind_trial_input(PortId, Vec) = 0;
    virtual void advance(const TimeInterval&) = 0;
    virtual void commit() = 0;
    virtual void release_checkpoint() noexcept = 0;
    virtual bool restore() noexcept = 0;
    virtual Mat mass_matrix(PortId) { return nullptr; }
};

struct SchedulerPosition
{
    Real time = 0;
    std::size_t step = 0;
};

struct SchedulerOptions
{
    // Test hooks keep scheduler tests independent of PETSc/MOAB field construction.
    using StagingFactory = Vec (*)(const ConnectionSpec&, KARMANativeComponent&, KARMANativeComponent&);
    using ExchangeHook = void (*)(const ConnectionSpec&, KARMANativeComponent&, KARMANativeComponent&, FieldVisibility,
                                  Vec);
    StagingFactory staging_factory = nullptr;
    ExchangeHook exchange_hook = nullptr;
};

struct CompiledGraphMetadata
{
    GraphId id;
    std::array<ComponentInstanceSpec, 2> components;
    std::array<ConnectionSpec, 2> connections;
    SplitScheme scheme = SplitScheme::Lie;
};

struct GraphComponentReport
{
    std::string id;
    std::vector<PortSpec> ports;
    bool supports_fractional_steps = false;
    bool supports_checkpoint_restore = false;
    std::string mesh_id = "unknown";
    std::string layout_id = "unknown";
    std::string basis_id = "unknown";
    std::string solver_policy_id = "unknown";
    /** Hash of this component's canonical SEMANTIC description - what its
     * ports mean - as distinct from the caller-declared mesh, layout, basis
     * and solver identifiers above, which are metadata the graph does not
     * interpret. Two components whose contracts differ in any way the
     * semantic model can express have different fingerprints here, so an
     * introspection report shows a contract change that the declared
     * identifiers would not reveal. */
    std::uint64_t contract_fingerprint = 0;
};

struct GraphConnectionReport
{
    ConnectionSpec connection;
    std::string transfer_id = "unknown";
    std::string solver_policy_id = "unknown";
    std::string setup = "not initialized";
    std::string coverage = "unknown";
};

struct RuntimeConnectionReport
{
    ConnectionSpec connection;
    std::size_t setup_count = 0;
    std::string setup = "unknown";
    std::string coverage = "unknown";
    FEMConnectionDiagnostics diagnostics;
};

struct OperatorSplitRuntimeReport
{
    SchedulerState state = SchedulerState::Constructed;
    SchedulerPosition position;
    std::vector<CompiledInstruction> schedule;
    std::vector<RuntimeConnectionReport> connections;
};

struct CompiledGraphReport
{
    std::string signature;
    SplitScheme scheme = SplitScheme::Lie;
    std::vector<GraphComponentReport> components;
    std::vector<GraphConnectionReport> connections;
    std::vector<CompiledInstruction> instructions;
    std::vector<std::string> capabilities;
    std::vector<std::string> deferred_features;
};

class CompiledOperatorSplitGraph
{
  public:
    CompiledOperatorSplitGraph(CompiledGraphMetadata metadata, std::vector<CompiledInstruction> instructions);

    [[nodiscard]] const CompiledGraphMetadata& metadata() const noexcept { return metadata_; }
    [[nodiscard]] const std::vector<CompiledInstruction>& instructions() const noexcept { return instructions_; }
    [[nodiscard]] std::string signature() const;
    [[nodiscard]] CompiledGraphReport report() const;
    [[nodiscard]] std::vector<std::string> diff(const CompiledOperatorSplitGraph&) const;

  private:
    const CompiledGraphMetadata metadata_;
    const std::vector<CompiledInstruction> instructions_;
};

class KARMACouplingScheduler
{
  public:
    KARMACouplingScheduler(CompiledOperatorSplitGraph,
                           std::array<KARMANativeComponent*, 2>,
                           SchedulerOptions = {});
    KARMACouplingScheduler(CompiledOperatorSplitGraph,
                           std::array<KARMANativeComponent*, 2>, SchedulerPosition,
                           SchedulerOptions = {});
    ~KARMACouplingScheduler();
    KARMACouplingScheduler(const KARMACouplingScheduler&) = delete;
    KARMACouplingScheduler& operator=(const KARMACouplingScheduler&) = delete;

    void advance(Real time_left, Real time_right);
    [[nodiscard]] FixedPointResult advance_fixed_point(Real time_left, Real time_right,
                                                       FixedPointOptions options = {});
    [[nodiscard]] SchedulerPosition position() const noexcept { return position_; }
    [[nodiscard]] SchedulerState state() const noexcept { return state_; }
    [[nodiscard]] std::array<std::size_t, 2> connection_setup_counts() const noexcept;
    [[nodiscard]] std::array<FEMConnectionDiagnostics, 2> connection_diagnostics() const noexcept;
    [[nodiscard]] OperatorSplitRuntimeReport report() const;

  private:
    CompiledOperatorSplitGraph graph_;
    std::array<KARMANativeComponent*, 2> components_;
    SchedulerOptions options_;
    SchedulerPosition position_;
    SchedulerState state_ = SchedulerState::Constructed;
    std::array<Vec, 2> staging_{};
    std::array<std::unique_ptr<CompiledFEMConnection>, 2> connections_;
};

CompiledOperatorSplitGraph compile_operator_split_graph(const OperatorSplitGraphSpec& specification);
CompiledOperatorSplitGraph compile_operator_split_graph(const OperatorSplitGraphSpec& specification,
                                                        MPI_Comm communicator);

}  // namespace KARMA

#endif
