#ifndef KARMA_COUPLING_OPERATOR_SPLIT_CHECKPOINT_H
#define KARMA_COUPLING_OPERATOR_SPLIT_CHECKPOINT_H

#include "karma_operator_split_graph.h"
#include "petsc_raii.h"

#include <array>
#include <string>

namespace KARMA
{
class TransientReactionDiffusionNativeAdapter;

struct OperatorSplitCheckpointBundle
{
    std::string u_checkpoint;
    std::string v_checkpoint;
    std::string metadata;
};

struct OperatorSplitRestartOptions
{
    // Test hook used to inject a rank-local publication failure before mutation.
    using PublicationHook = void (*)(std::size_t component);
    PublicationHook before_publication = nullptr;
};

class PreparedOperatorSplitRestart
{
  public:
    PreparedOperatorSplitRestart(PreparedOperatorSplitRestart&&) noexcept = default;
    PreparedOperatorSplitRestart& operator=(PreparedOperatorSplitRestart&&) noexcept = default;
    PreparedOperatorSplitRestart(const PreparedOperatorSplitRestart&) = delete;
    PreparedOperatorSplitRestart& operator=(const PreparedOperatorSplitRestart&) = delete;

    [[nodiscard]] SchedulerPosition position() const noexcept { return position_; }
    void apply();

  private:
    friend PreparedOperatorSplitRestart prepare_operator_split_restart(
        const OperatorSplitCheckpointBundle&, const CompiledOperatorSplitGraph&,
        std::array<TransientReactionDiffusionNativeAdapter*, 2>, OperatorSplitRestartOptions);

    PreparedOperatorSplitRestart(std::array<TransientReactionDiffusionNativeAdapter*, 2>,
                                 std::array<PetscOwner<Vec, VecDestroy>, 2>, SchedulerPosition,
                                 OperatorSplitRestartOptions);

    std::array<TransientReactionDiffusionNativeAdapter*, 2> components_{};
    std::array<PetscOwner<Vec, VecDestroy>, 2> staged_;
    SchedulerPosition position_;
    OperatorSplitRestartOptions options_;
    bool applied_ = false;
};

void write_operator_split_checkpoint_bundle(
    const OperatorSplitCheckpointBundle&, const CompiledOperatorSplitGraph&,
    std::array<TransientReactionDiffusionNativeAdapter*, 2>, SchedulerPosition);

[[nodiscard]] PreparedOperatorSplitRestart prepare_operator_split_restart(
    const OperatorSplitCheckpointBundle&, const CompiledOperatorSplitGraph&,
    std::array<TransientReactionDiffusionNativeAdapter*, 2>, OperatorSplitRestartOptions = {});
}  // namespace KARMA

#endif
