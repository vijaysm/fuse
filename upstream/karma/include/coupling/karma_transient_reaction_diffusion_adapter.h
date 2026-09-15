#ifndef KARMA_COUPLING_TRANSIENT_REACTION_DIFFUSION_ADAPTER_H
#define KARMA_COUPLING_TRANSIENT_REACTION_DIFFUSION_ADAPTER_H

#include "karma_operator_split_graph.h"
#include "gray_scott_model.h"
#include "transient_reaction_diffusion_component.h"

#include <string>
#include <string_view>
#include <functional>

namespace KARMA
{
class TransientReactionDiffusionNativeAdapter final : public KARMANativeComponent
{
  public:
    TransientReactionDiffusionNativeAdapter(std::string id, KARMAMesh&, GrayScottSpecies,
                                            GrayScottParameters parameters = {}, Real initial_state = 0);

    [[nodiscard]] const ComponentSpec& specification() const noexcept override { return specification_; }
    [[nodiscard]] MPI_Comm communicator() const noexcept override { return communicator_; }
    void initialize() override;
    KARMAField input_field(PortId) override;
    KARMAField output_field(PortId, FieldVisibility) override;
    void checkpoint() override;
    void bind_trial_input(PortId, Vec) override;
    void advance(const TimeInterval&) override;
    void commit() override;
    void release_checkpoint() noexcept override;
    [[nodiscard]] bool restore() noexcept override;
    [[nodiscard]] Mat mass_matrix(PortId) override;
    [[nodiscard]] bool is_initialized() const noexcept { return initialized_; }
    [[nodiscard]] Real committed_time() const noexcept;
    [[nodiscard]] KARMASystem& checkpoint_system();
    [[nodiscard]] Vec committed_solution() const;
    [[nodiscard]] Mat mass() const;
    [[nodiscard]] const TransientSolveDiagnostics& diagnostics() const noexcept;
    void set_initial_condition(const std::function<Real(const Point&)>&);
    void write_checkpoint(std::string_view filename);
    void restore_committed_solution(Vec staged, Real time);

  private:
    static TransientReactionDiffusionModel make_model(GrayScottSpecies, GrayScottModel);
    [[nodiscard]] KARMAField field(Vec);
    void require_port(const PortId&, PortDirection) const;

    std::string id_;
    Real initial_state_;
    ComponentSpec specification_;
    TransientReactionDiffusionComponent component_;
    MPI_Comm communicator_ = MPI_COMM_NULL;
    bool initialized_ = false;
};
}  // namespace KARMA

#endif
