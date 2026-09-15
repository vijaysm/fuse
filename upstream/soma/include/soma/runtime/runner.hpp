#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "soma/field/registry.hpp"
#include "soma/mesh/mesh_view.hpp"
#include "soma/runtime/component.hpp"
#include "soma/schedule/plan.hpp"
#include "soma/transport/transport.hpp"

namespace soma {

struct RunnerConfig {
  std::string transport = "copy";
  bool        check_conservation = true;
  double      conservation_tolerance = 1e-12;
  bool        abort_on_violation = false;
  /// Spec §5.3. An iteration region that reaches `max_iterations` without
  /// meeting its convergence criterion has NOT converged, and the values it
  /// leaves behind are the last iterate, not a converged solution.
  ///
  /// `false` (the default) counts the window in
  /// `RunStats::windows_not_converged` and continues, which is what a
  /// production run that must not lose a job to one stiff window wants; the
  /// counter is the only record, so a run that ignores it is reporting an
  /// unconverged answer as a converged one. `true` fails the period with
  /// `E-RT-NOCONVERGE`.
  bool        abort_on_nonconvergence = false;

  /// `MPI_Comm_c2f()` of a communicator whose group contains every rank of
  /// every component in the graph -- `MPI_COMM_WORLD` under spec §6 mode 1,
  /// which is the only integration mode v1 implements. Set it to make
  /// `Transport` actually redistribute; leave it unset and `Transport` moves
  /// whole arcs, which is what `copy` does and what every existing caller
  /// relies on.
  ///
  /// WHY A BASE COMMUNICATOR AND NOT A PER-COMPONENT CALLBACK.
  /// `ComponentDecl::ranks` (include/soma/graph/component.hpp:13) already IS
  /// the graph's statement of who belongs to a component, and
  /// src/ir/rules_channel.cpp already asks it whether a Transport is needed at
  /// all. Splitting each component's communicator out of this one, by that
  /// same field, keeps the lowering and the runtime reading ONE source of
  /// truth. A `std::function<int(ComponentId)>` would be a second one, free to
  /// disagree with the first, and would add C ABI surface -- whereas
  /// `soma_set_communicator` (src/capi/soma_c.cpp:659-667) already stores
  /// exactly this handle and need only forward it.
  ///
  /// LIMITATION, WRITTEN DOWN RATHER THAN DISCOVERED. `RankSet` is a
  /// contiguous range, so the graph cannot currently express arbitrary partial
  /// overlap. `CommGraph` does not care -- it takes a communicator per side
  /// precisely so that every overlap shape is uniform data. When the graph
  /// gains non-contiguous rank sets, only the `MPI_Comm_split` colour changes.
  ///
  /// std::optional because a Fortran communicator handle has no portable
  /// "unset" value to use as a sentinel: 0 is a legal handle.
  std::optional<int> base_communicator;

  /// Task 8.5's gap-closer. `Transport`'s virtual interface (prepare/move/
  /// name/is_local) has no hook for a backend's one-time, backend-specific
  /// setup -- `MpiTransport::set_communicator`+`build_plan` or
  /// `LowFiveTransport::set_location`+`build_plan` -- because that setup takes
  /// a concrete type (a Fortran communicator handle, a file path), not
  /// anything `Transport&` can express. Without this, `RunnerConfig{.transport
  /// = "lowfive"}` (or "mpi") builds an unconfigured transport that fails at
  /// the first Transport instruction.
  ///
  /// If set, `Runner::initialize()` calls it exactly once, immediately after
  /// the registry constructs the named transport and before any instruction
  /// is resolved, with a mutable reference to that transport. A caller that
  /// needs the concrete type `dynamic_cast`s to it. Left empty (the default)
  /// for `"copy"` and any transport that needs no such setup; empty is a
  /// silent no-op, not an error, so `CopyTransport` and every existing
  /// `RunnerConfig{}` user is unaffected.
  std::function<Result<void>(Transport&)> transport_setup;
};

struct RunStats {
  std::int64_t periods = 0;
  std::int64_t instructions_executed = 0;
  std::int64_t conservation_checks = 0;
  std::int64_t conservation_violations = 0;
  double       worst_conservation_error = 0.0;
  /// Sweeps of an iteration region body, summed over every region and every
  /// period. A period with no region adds nothing.
  std::int64_t total_iterations = 0;
  /// Region-periods that hit `max_iterations` without satisfying the
  /// criterion. Non-zero means some answer in this run is not converged.
  std::int64_t windows_not_converged = 0;
  /// Spec §5.4. How many conservative remap matrices `initialize()` built for
  /// this plan -- one per distinct `RemapPlan`, built at setup and never in
  /// the step loop.
  ///
  /// WHY THIS IS A COUNTER AND NOT A BOOLEAN. `worst_conservation_error` reads
  /// exactly 0 both when a remap was checked and was perfect and when no remap
  /// ran at all, so no error magnitude can distinguish "the weights were used"
  /// from "the weights were never built". A test that wants to assert a REAL
  /// remap happened -- rather than the coincident-mesh identity the runtime
  /// still uses when it can prove the two regions are the same cells in the
  /// same order -- asserts on this.
  std::int64_t remap_weight_builds = 0;
  /// Non-zeros summed over those matrices. `n` for an exact permutation of `n`
  /// cells, more when cells genuinely overlap several partners; 0 when every
  /// `Remap` took the proved-coincident identity path.
  std::int64_t remap_weight_nonzeros = 0;
  /// Times the runtime compared a component's registered `Provides` storage
  /// against the checkpoint it took, after that component's `restore_state()`
  /// (spec §5.3: SOMA verifies that the component saved what the derived set
  /// said it must, rather than trusting `supports_rollback()`). Exposed so the
  /// verification is observable, and therefore testable, rather than asserted
  /// in a comment.
  ///
  /// R16: "registered" is load-bearing, not decorative. Each count is a
  /// `memcmp` against one arc's REGISTERED storage; a component's own
  /// private state that was never bound via `register_fields()` is not
  /// counted here and is not covered by any of these verifications, even
  /// though `save_state()`/`restore_state()` are contractually required to
  /// handle it too (see the caveat on `Component::save_state()`,
  /// include/soma/runtime/component.hpp).
  std::int64_t checkpoint_verifications = 0;
};

/// Per-region outcome of the most recent period (spec §5.3).
struct IterationStats {
  int    iterations = 0;
  bool   converged = false;
  /// One entry per sweep: the 2-norm of the whole concatenated coupling
  /// residual `x_new - x_old`. Capacity is reserved at setup, so appending to
  /// it inside the sweep loop allocates nothing.
  std::vector<double> residual_history;
};

/// Spec §2 layer 3: interprets the flat instruction stream. Every ArcId is
/// resolved to a raw pointer once at initialize(); the loop allocates nothing.
class Runner {
 public:
  Runner(const CouplingGraph& graph, const OperatorDAG& dag, ExecutionPlan plan,
         const MeshBackend& backend);
  ~Runner();

  Runner(Runner&&) noexcept;
  Runner& operator=(Runner&&) noexcept;

  Result<void> initialize(ComponentSet& components, FieldRegistry& registry,
                          const RunnerConfig& config = {});

  Result<void> run_period(ComponentSet& components);
  Result<void> run(ComponentSet& components, std::int64_t periods);

  [[nodiscard]] const RunStats& stats() const noexcept;
  /// One entry per iteration region of the plan, in DAG region order,
  /// describing the most recent period. Empty for a plan with no region.
  [[nodiscard]] const std::vector<IterationStats>& iteration_stats()
      const noexcept;
  /// The derived checkpoint set's explanation for every region, in DAG region
  /// order (spec §5.3: the user never wrote the set down, so the only way they
  /// can disagree with it is if it is printed). Available after initialize().
  [[nodiscard]] const std::vector<std::string>& checkpoint_explanations()
      const noexcept;
  [[nodiscard]] Result<const double*> buffer(ArcId arc) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace soma
