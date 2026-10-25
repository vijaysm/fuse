#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <iterator>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "fuse/base/rational.hpp"
#include "fuse/base/error.hpp"
#include "fuse/base/result.hpp"
#include "fuse/field/registry.hpp"
#include "fuse/model/capabilities.hpp"
#include "fuse/model/coupling_graph.hpp"
#include "fuse/model/multiscale.hpp"
#include "fuse/model/restart.hpp"

namespace fuse {

/// The half-open time interval `[t_begin, t_end)` a single
/// `Component::advance()` call must cover, plus where the call sits in the
/// schedule.
///
/// Times are `Rational`, never `double` (spec Section 4.3): 1/3 and 1/7 subcycling
/// must decompose into slabs with no drift over 10^6 steps, which
/// floating-point accumulation destroys silently.
///
/// `step` counts coupling steps from the start of the run. `iteration` is the
/// 0-based index of the current sweep inside an iteration region (spec Section 5.3)
/// and is 0 everywhere outside one; a component that is re-advanced over the
/// *same* interval with `iteration > 0` is being iterated and must have had
/// `restore_state()` called first.
struct StepContext {
  Rational     t_begin{0};
  Rational     t_end{0};
  std::int64_t step = 0;
  int          iteration = 0;  // 0 outside an iteration region

  /// Exact interval length. Zero for a degenerate step.
  [[nodiscard]] constexpr Rational dt() const { return t_end - t_begin; }

  /// Structural sanity of a context produced by the scheduler.
  ///
  /// The driver that builds a `StepContext` is responsible for checking this
  /// before handing it to `advance()`: `Component` cannot enforce it, because
  /// `advance()` is the implementer's own override and nothing in the base
  /// runs first. A context with `t_end < t_begin` fed to a component that
  /// integrates over `dt()` produces a negative time step, which is not a
  /// crash -- it is a plausible-looking wrong answer.
  [[nodiscard]] constexpr bool valid() const noexcept {
    return t_begin <= t_end && step >= 0 && iteration >= 0;
  }
};

/// The residual/linearisation seam: a component's own approximate
/// linearisation of its `ImplicitComponent::evaluate_residual()` about the
/// point of its most recent call.
///
/// Acts on this component's OWN block of the region's stacked interface
/// vector only -- `cols()`-long in, `rows()`-long out (and the reverse for
/// the transpose) -- never the whole region's vector. The Newton coupler
/// is responsible for scattering a global Krylov iterate into each
/// member's own block before calling `apply_jacobian()`, and gathering the
/// per-block outputs back, exactly the convention
/// `fuse::linalg::MatrixFreeOperator` already uses for a single operator
/// over the whole vector (fuse/linalg/operator.hpp): this token is that
/// same duck-typed shape, one per region member.
class LinearizationToken {
 public:
  virtual ~LinearizationToken() = default;

  /// y = J x, J this component's own Jacobian of `evaluate_residual()` at
  /// the point this token was returned for. `x.size() == cols()`,
  /// `y.size() == rows()`.
  virtual Result<void> apply_jacobian(std::span<const double> x,
                                      std::span<double> y) = 0;

  /// y = J^T x. `x.size() == rows()`, `y.size() == cols()`. Needed by a
  /// normal-equations or adjoint formulation; a component whose block is
  /// non-square or not meaningfully transposable may return an error
  /// rather than approximate.
  virtual Result<void> apply_jacobian_transpose(std::span<const double> x,
                                                std::span<double> y) = 0;

  /// z = M^-1 r, this component's own physics-based preconditioner for its
  /// diagonal block: an approximate inverse cheaper than factorising
  /// `apply_jacobian()` (a lagged operator, a multigrid V-cycle, a
  /// lower-fidelity model). `r.size() == rows()`, `z.size() == cols()`. A
  /// component with no better choice may implement this as the identity
  /// (copy `r` into `z`) rather than refuse: the Newton coupler's
  /// block-Jacobi preconditioner composes whatever every member
  /// returns, and the identity is always a legal, if weak, one.
  virtual Result<void> apply_preconditioner(std::span<const double> r,
                                            std::span<double> z) = 0;

  /// Output/input lengths of `apply_jacobian()`. A Newton region checks
  /// these against the block sizes its own interface-vector assembly
  /// derived for this component and refuses (E-NEWTON-SHAPE,
  /// `fuse::verify_linearization_shape()` below) on a disagreement, rather
  /// than running a Krylov iteration over a silently truncated or
  /// out-of-bounds block.
  [[nodiscard]] virtual std::size_t rows() const = 0;
  [[nodiscard]] virtual std::size_t cols() const = 0;

  /// Changes whenever the linearisation point moves (a new
  /// `evaluate_residual()` call), so the Newton loop can tell a retained
  /// token is stale without re-comparing the interface vector itself. No
  /// ordering or uniqueness guarantee beyond "equal signature, same
  /// point" -- a call counter, a hash of the input, or (for a component
  /// that only ever linearises once) a fixed constant are all legal
  /// implementations.
  [[nodiscard]] virtual std::uint64_t signature() const = 0;
};

/// The residual/linearisation seam: an interface a `Component` may
/// ADDITIONALLY implement, queried through `Component::implicit()` rather
/// than through a second base class -- see that method's own comment for
/// why.
///
/// A component implementing this must also set `kCapabilityResidual` (and,
/// if `linearization()` can return non-null, `kCapabilityLinearization`)
/// in `capabilities()` (fuse/model/capabilities.hpp): a Newton-solved
/// iteration region checks the BIT, not this interface, at validation time
/// (`fuse::verify_newton_capability_support()`), the same checked-not-
/// trusted pattern `kCapabilityRollback` already follows for
/// `supports_rollback()`.
class ImplicitComponent {
 public:
  virtual ~ImplicitComponent() = default;

  /// This component's own block of the fixed-point map, G_i(x): the values
  /// it would `Provide` at time level `ctx.t_end` given the interface
  /// values it `Requires` inside the region (already remapped and
  /// time-interpolated by FUSE, exactly as `advance()` would see them at
  /// the SAME `ctx`) in `interface_in`, in the stacking order
  /// `fuse::FixedPointResidual` (fuse/runtime/fixed_point_residual.hpp)
  /// assembles a region's blocks in. `residual_out` is G_i ALONE -- the
  /// Runner forms the region's residual F(x) = x - G(x) itself, so a
  /// component never subtracts `interface_in` here (dn_heat's
  /// DirichletSlab/NeumannSlab and the Brusselator's species components
  /// are the two implementations, and both return G_i).
  ///
  /// MUST NOT mutate any state `advance()` would mutate -- registered
  /// field storage, private state, anything `save_state()` would need to
  /// capture -- and must leave it bit-identical to what it was on entry: a
  /// test (test/runtime/implicit_component_test.cpp) calls `save_state()`,
  /// this, then `save_state()` again and compares the two checkpoints.
  /// Called possibly many times at the SAME `ctx` and the SAME
  /// `interface_in` (a Krylov iteration can revisit a point, and a
  /// matrix-free Jacobian action perturbs around one), and must return the
  /// same `residual_out`, bit for bit, every time: no RNG, no unordered
  /// reduction, no hidden counter -- the same purity `advance()` itself
  /// must have inside an iteration region (`Component::advance()`'s own
  /// doc comment).
  ///
  /// `interface_in.size()` is this component's own slice of the region's
  /// stacked interface vector: the sum of the lengths of every `Requires`
  /// field this component reads inside the region. `residual_out.size()`
  /// is the sum of the lengths of every `Provides` field it writes inside
  /// the region. The two need not be equal.
  ///
  /// Returning an error aborts the Newton step the same way an error from
  /// `advance()` aborts an explicit one; the coupler does not retry.
  ///
  /// WHAT "A FROZEN TIME LEVEL" MEANS FOR A WINDOW-SCOPED MEMBER (window
  /// Newton, newton-krylov.md Sec. 9). The paragraph above
  /// describes an ORDINARY (single-clock) region, where `ctx.t_end` is
  /// one instant and `interface_in`/`residual_out` are each one field's
  /// worth of plain values. `Runner::Impl::evaluate_window_region_
  /// residual()` (src/runtime/iteration.cpp) -- the window-scoped
  /// seam -- never calls this method at all; it reaches a window
  /// member's own `G` through `advance_window()`'s own substep loop
  /// directly, the same way the ORDINARY seam's `evaluate_region_
  /// residual()` reaches an ordinary member's own `G` through `advance()`
  /// directly rather than through this interface (see that function's
  /// own class-level comment on `DirichletSlab`, examples/dn_heat/
  /// dn_component.hpp: this method exists to satisfy the capability gate
  /// HONESTLY, not to be the Newton solver's own call path). For a
  /// window-scoped member that chooses to implement this method anyway
  /// (as `examples/brusselator/species_component.hpp`'s `UComponent`/
  /// `VComponent` do, to make `kCapabilityResidual` true for a real
  /// reason), "a frozen time level" generalises to a FROZEN WINDOW:
  /// `ctx.t_begin`/`ctx.t_end` span the WHOLE window `[T0, T1)`, not one
  /// instant; `interface_in` is this member's own stacked, per-element,
  /// per-sample import over that window (the SAME `(element, sample)`
  /// layout `RegionBlock::window_export_capacity` documents, sample 0
  /// the window-start value); and `residual_out` is this member's own
  /// stacked EXPORT over the SAME window, in the identical layout --
  /// i.e. this component's own `G` for the window, computed by running
  /// its own substeps across `[T0, T1)` exactly as `advance_window()`
  /// would, without advancing (mutating) anything -- never `x - G`, the
  /// genuine residual, which only the Runner itself assembles, one level
  /// up, by comparing this output against the interface vector's own
  /// matching block (`evaluate_window_region_residual()`'s own `f_out =
  /// x - g`). The purity contract above is UNCHANGED and, if anything,
  /// harder to satisfy for a window member: every one of its own
  /// substeps must run on a SAVED/RESTORED copy of its state, not the
  /// live one `advance_window()` mutates.
  virtual Result<void> evaluate_residual(const StepContext& ctx,
                                         std::span<const double> interface_in,
                                         std::span<double> residual_out) = 0;

  /// The linearisation of `evaluate_residual()` about the point of its
  /// most recent call, or `nullptr` if this component cannot supply one --
  /// the Newton loop then falls back to a purely matrix-free directional
  /// derivative for this component's own block. Ownership: the returned
  /// pointer is borrowed, valid until this component's next
  /// `evaluate_residual()` or `advance()` call, whichever comes first; a
  /// caller must not retain it across either.
  [[nodiscard]] virtual LinearizationToken* linearization() { return nullptr; }
};

/// The E-NEWTON-SHAPE check: whether `token`'s own `rows()`/`cols()`
/// agree with the block sizes a region's interface-vector assembly derived
/// for the component that returned it. A pure function of already-known
/// sizes, so it needs no graph, no `ComponentSet` and no running region --
/// test/model/newton_capability_test.cpp calls it directly against a
/// hand-built token of the wrong shape.
[[nodiscard]] inline Diagnostics verify_linearization_shape(
    const std::string& component_name, const LinearizationToken& token,
    std::size_t expected_rows, std::size_t expected_cols) {
  Diagnostics diags;
  if (token.rows() != expected_rows || token.cols() != expected_cols) {
    diags.add(
        Severity::Error, "E-NEWTON-SHAPE",
        "component '" + component_name +
            "' returned a LinearizationToken with rows()=" +
            std::to_string(token.rows()) + ", cols()=" +
            std::to_string(token.cols()) +
            ", but the region's own interface-vector assembly expects this "
            "component's block to be " + std::to_string(expected_rows) +
            " residual row(s) by " + std::to_string(expected_cols) +
            " interface column(s). A Krylov iteration run over a token of "
            "the wrong shape either runs past the end of the block or never "
            "reads part of it, both silently",
        "component '" + component_name + "'");
  }
  return diags;
}

/// A participating physics code, integrated in-process (spec Section 6 mode 1).
///
/// Deliberately minimal: name yourself, bind your storage, advance in time.
/// Everything else FUSE needs -- remap, accumulation, scheduling -- it derives
/// from the `CouplingGraph`, not from the component.
///
/// OWNERSHIP MODEL (spec Section 4.2). The component owns its field arrays. FUSE
/// never allocates, copies, or frees them; it wraps them in non-owning
/// `FieldRef`s. FUSE owns only its own intermediates (accumulators, waveform
/// buffers, remap targets, previous iterates). A `Component` held in a
/// `ComponentSet` is owned by that set and destroyed with it, through this
/// base pointer -- hence the virtual destructor.
///
/// THREADING. All methods are called from the component's own rank, from a
/// single thread, and never concurrently with each other. A component may use
/// threads or a device queue internally, but every method must have completed
/// its externally visible effects before it returns.
class Component {
 public:
  Component() = default;
  virtual ~Component() = default;

  // Copying or slicing a component would duplicate the storage identity the
  // registry's FieldRefs point at. Derived classes remain free to define
  // their own copy semantics if they need them.
  Component(const Component&) = delete;
  Component& operator=(const Component&) = delete;
  Component(Component&&) = delete;
  Component& operator=(Component&&) = delete;

  /// This component's name, which must equal the `ComponentDecl::name` of the
  /// graph node it implements -- that string is the only link between the
  /// authored graph and the running object.
  ///
  /// Contract for the implementer: must be non-empty, must be the same value
  /// on every call for the object's whole lifetime, and must not depend on
  /// anything mutated by `advance()`. `ComponentSet` uses it as a key and
  /// rejects an empty or duplicate name at insertion time; a name that
  /// *changes* afterwards silently breaks every lookup and cannot be caught
  /// here.
  ///
  /// Returns by value; call it at setup, not in the step loop.
  [[nodiscard]] virtual std::string name() const = 0;

  /// Bind this component's own storage into `registry`, once, at setup
  /// (spec Section 4.2).
  ///
  /// The implementer must, for each `FieldId` in
  /// `graph.component(id).fields`, call `registry.bind(id, FieldRef{...})`
  /// with a pointer to storage it owns and a `Layout` describing how to walk
  /// it. `bind()` rejects a null pointer, an invalid `FieldId`, a layout whose
  /// strides alias or overflow, and a second bind of the same field; propagate
  /// the returned error rather than ignoring it.
  ///
  /// LIFETIME. `FieldRef` is non-owning. The registered address must stay
  /// valid and *fixed* for the whole run: FUSE resolves once here and
  /// dereferences in the hot loop. Registering `vec.data()` and later growing,
  /// reassigning, or reallocating `vec` dangles every view. This includes
  /// `restore_state()` -- restore a checkpoint element-wise, do not
  /// reassign the container.
  ///
  /// `registry` outlives this call and keeps copies of the views. `graph` is
  /// borrowed for the duration of the call only; do not retain a pointer or
  /// reference to it or to anything it owns.
  ///
  /// Called exactly once, before the first `advance()`. On error the caller
  /// aborts setup; the registry is left with whatever succeeded so far, so
  /// nothing may be assumed about partial state after a failure.
  virtual Result<void> register_fields(FieldRegistry& registry,
                                       const CouplingGraph& graph) = 0;

  /// Integrate this component's own state from `ctx.t_begin` to `ctx.t_end`.
  ///
  /// On entry the implementer may assume: `register_fields()` succeeded; every
  /// field this component declared `Intent::Requires` holds values valid at
  /// `ctx.t_begin`, already remapped and time-interpolated by FUSE; and
  /// `ctx.valid()` (the driver checks it -- see `StepContext::valid()`).
  ///
  /// On return the implementer must guarantee: every field it declared
  /// `Intent::Provides` holds values valid at `ctx.t_end`, written *in place*
  /// through the storage registered in `register_fields()`; and no registered
  /// pointer has moved. Writing to a `Requires` field is undefined from FUSE's
  /// point of view -- FUSE may overwrite it at any later exchange.
  ///
  /// `ctx` is borrowed for the duration of the call; copy it if you need it
  /// afterwards.
  ///
  /// Inside an iteration region `advance()` is called repeatedly with the same
  /// `t_begin`/`t_end` and increasing `ctx.iteration`, each call preceded by a
  /// `restore_state()`. It must therefore be a pure function of (restored
  /// state, inputs) -- accumulating a side effect per call, such as appending
  /// to an output file or incrementing an internal step counter, produces
  /// results that depend on the iteration count.
  ///
  /// Returning an error aborts the step; FUSE does not retry.
  virtual Result<void> advance(const StepContext& ctx) = 0;

  /// Checkpoint the state `advance()` mutates, so the same interval can be
  /// re-advanced (spec Section 5.3).
  ///
  /// Spec Section 5.3: FUSE derives the checkpoint set from dataflow over the SSA
  /// form, tells the component which of its state crosses the region
  /// boundary, and *verifies at runtime that it was saved* -- unlike preCICE,
  /// which pushes `requiresWritingCheckpoint()` onto the user. A component that
  /// appears in an iteration region with `supports_rollback() == false` is
  /// refused.
  ///
  /// CAVEAT, because the sentence above reads as a stronger guarantee
  /// than it is: the runtime verification is a `memcmp` against REGISTERED
  /// field storage only (the arcs `register_fields()` bound; see
  /// `Runner::Impl::run_region`, src/runtime/iteration.cpp), one per
  /// component the derived checkpoint set names. It cannot see, and
  /// therefore cannot verify, any state private to the component that was
  /// never registered -- the paragraph below asks the implementer to save
  /// that too, but nothing in FUSE checks that they did. "FUSE verifies the
  /// rollback" means "FUSE verifies the registered part of the rollback."
  ///
  /// The default is a successful no-op, correct only for a component that is
  /// never iterated. An implementer that overrides these must save everything
  /// `advance()` writes, including state private to the component that FUSE
  /// cannot see, and must not move any registered field storage while doing
  /// so. Only one checkpoint need be retained; FUSE saves at most once per
  /// iteration region entry.
  ///
  /// Outside an iteration region FUSE may call `save_state()` and never call
  /// `restore_state()`.
  virtual Result<void> save_state() { return Result<void>::success(); }

  /// Restore the state captured by the most recent successful `save_state()`.
  ///
  /// Called before every re-sweep of an iteration region. Restoring must not
  /// reallocate or otherwise move registered field storage (see
  /// `register_fields()`); copy element-wise into the existing arrays.
  ///
  /// Calling this without a preceding successful `save_state()` is a caller
  /// error; an implementer that can detect it should return an error rather
  /// than restore garbage.
  virtual Result<void> restore_state() { return Result<void>::success(); }

  /// Restart (docs/dev/restart.md), Form A: put every array this component
  /// owns into `writer`, keyed by
  /// THIS COMPONENT'S OWN global ids for the array's index space -- a
  /// component that lives on one rank numbers its own entries `0..n-1`,
  /// trivially satisfying the "re-partitionable" requirement (there is
  /// nothing to re-partition when the whole array is already on one
  /// rank); a mesh-backed component uses `MeshView::global_id()`
  /// (fuse/backend/mesh_view.hpp) instead, so a later restart at a
  /// DIFFERENT rank count can find each row by the mesh entity it
  /// belongs to, not by this run's own local position.
  ///
  /// Deliberately a SEPARATE virtual from `save_state()`/`restore_state()`
  /// above, which this method shares no caller with: those are the
  /// in-memory, single-slot ROLLBACK checkpoint `run_region()` uses to
  /// re-advance one iteration-region sweep (the caveat on that pair
  /// applies to restart identically and is restated here rather than
  /// assumed: FUSE can verify that every id this call named was written
  /// by SOME rank and that no id repeats, `RestartWriter::finish_point()`'s
  /// own contract, but it cannot verify that `values` is a faithful
  /// serialization of this component's full state).
  ///
  /// Called at a period boundary only (docs/dev/restart.md Section 1.1:
  /// an externally-driven component's suspended coroutine frame has no
  /// serializable representation, so a restart point is never taken
  /// mid-period). The default is a successful no-op, correct only for a
  /// component that declares neither `kCapabilityRestartFormA` nor
  /// `kCapabilityRestartFormB` -- `write_restart_point()` (fuse/runtime/
  /// restart.hpp) never calls this for such a component at all.
  ///
  /// PURITY. An override must not mutate component state: `Runner::
  /// restore_restart_fields()` calls this a SECOND time on a restore
  /// (through `fuse::IdCollectingRestartWriter`, fuse/model/restart.hpp),
  /// purely to learn this component's own current global ids before
  /// `read_restart()` is called for real, with no intent to change
  /// anything -- a `write_restart()` with a side effect would run it
  /// twice. This mirrors `advance()`'s own purity requirement inside an
  /// iteration region's repeated sweeps (this header's own comment on
  /// that method), for the identical reason: a method FUSE may call more
  /// than once for the same logical point must be a pure function of
  /// the state it reports.
  virtual Result<void> write_restart(RestartWriter& /*writer*/) {
    return Result<void>::success();
  }

  /// The read-back counterpart of `write_restart()`. Called before the
  /// FIRST period after a restart, in the SAME per-component order
  /// `write_restart()` was called in when the point was written (the
  /// order itself carries no meaning Form A's own id-keyed reads depend
  /// on, but `restore_restart_point()`, fuse/runtime/restart.hpp,
  /// fixes one anyway, for a reproducible error if a read ever fails
  /// partway through).
  virtual Result<void> read_restart(RestartReader& /*reader*/) {
    return Result<void>::success();
  }

  /// Form B: an opaque, versioned per-rank byte
  /// record, same-decomposition only -- `memcpy`-cheap for a component
  /// with no mesh behind it to implement, at the cost that a restart at
  /// a DIFFERENT rank count refuses this component BY NAME (a graph
  /// refuses a rank-count change collectively if ANY
  /// component in it sets `kCapabilityRestartFormB` without also setting
  /// `kCapabilityRestartFormA`). The default returns an empty blob,
  /// correct only for a component that does not set
  /// `kCapabilityRestartFormB`.
  virtual Result<std::vector<std::byte>> save_restart_blob() {
    return std::vector<std::byte>{};
  }

  /// `written_by_version` is this component's own `restart_blob_version()`
  /// AT WRITE TIME, so an implementer can refuse a version it no longer
  /// understands by name rather than misread old bytes under a new
  /// layout -- the same posture `kGraphVersion` (fuse/model/
  /// descriptor.hpp) already takes for the coupling graph.
  virtual Result<void> load_restart_blob(std::span<const std::byte> /*bytes*/,
                                         std::uint32_t /*written_by_version*/) {
    return Result<void>::success();
  }

  /// This component's own serialization version for `save_restart_blob()`/
  /// `load_restart_blob()`. Must be constant for the object's lifetime,
  /// the same contract `capabilities()` already carries. The default, 0,
  /// is the version a component that never overrides either blob method
  /// writes (an empty blob at version 0, read back by a default
  /// `load_restart_blob()` that ignores it) -- consistent, never
  /// meaningful on its own.
  [[nodiscard]] virtual std::uint32_t restart_blob_version() const { return 0; }

  /// Whether `kCapabilityRestartFormA` is set. Non-virtual for the same
  /// reason `supports_rollback()` is (component.hpp's own comment on that
  /// method): one place states everything a derived class knows about
  /// itself (`capabilities()`), so an old override of a `bool` method
  /// from before this bit existed cannot silently shadow it.
  [[nodiscard]] bool supports_restart_form_a() const {
    return (capabilities() & kCapabilityRestartFormA) != 0;
  }

  /// Whether `kCapabilityRestartFormB` is set.
  [[nodiscard]] bool supports_restart_form_b() const {
    return (capabilities() & kCapabilityRestartFormB) != 0;
  }

  /// This component's capability bits. ONE virtual;
  /// `supports_rollback()`, `determinism_level()` and `carries_hidden_state()`
  /// below are non-virtual readers of it, not separate overridable questions
  /// -- a derived class states everything it knows about itself in one call,
  /// and cannot, say, claim rollback while leaving determinism at whatever a
  /// stale override happened to return.
  ///
  /// The bits are the SAME 64-bit space `fuse::ComponentSemantics::capabilities`
  /// and `fuse::FieldRequirement::required_capabilities` (fuse/model/
  /// semantics.hpp) carry; the assignments are named in fuse/model/
  /// capabilities.hpp, which also has the full rationale. In short: bit 0 is
  /// rollback, bits 1-2 encode `fuse::DeterminismLevel`
  /// (fuse/model/multiscale.hpp), bit 3 is hidden state, the rest reserved.
  ///
  /// Must be constant for the object's lifetime. The default, 0, decodes to
  /// {no rollback, determinism UNDECLARED, no hidden state}.
  /// `kCapabilityRollback` unset is the honest answer for the default no-op
  /// `save_state()`/`restore_state()` (a component that silently claimed
  /// rollback support while doing nothing would make an iteration region
  /// converge on stale state). The determinism field is likewise not
  /// assumed: `Bitwise` is the strongest claim the enum has, and a component
  /// that said nothing has not made it, so silence decodes to
  /// `std::nullopt` and an iteration region WARNS
  /// (W-ITER-DETERMINISM-UNDECLARED) until the component declares a level.
  /// Say what is true: `Bitwise` for a hand-written numerical component with
  /// no RNG and no unordered reduction, `RunToRun` if a library's reduction
  /// order is fixed per configuration only, `Nondeterministic` for an RNG or
  /// a stochastic surrogate (refused inside an iteration region). Hidden
  /// state: any state `advance()` mutates outside what `register_fields()`
  /// bound. See fuse/model/capabilities.hpp for the bits and
  /// `fuse::make_capabilities()` to set all three at once.
  [[nodiscard]] virtual std::uint64_t capabilities() const { return 0; }

  /// Whether `kCapabilityRollback` is set -- i.e. whether this component
  /// implements a real `save_state()`/`restore_state()` pair. Non-virtual on
  /// purpose: `capabilities()` is the one virtual a derived class can
  /// override, so a stale `bool supports_rollback() const override` from
  /// before this bit existed no longer overrides anything and is a compile
  /// error (test/runtime/component_iface_test.cpp's `compile_fail` case
  /// proves it), rather than a silently-ignored shadow of a base member.
  [[nodiscard]] bool supports_rollback() const {
    return (capabilities() & kCapabilityRollback) != 0;
  }

  /// This component's declared `DeterminismLevel`, decoded from
  /// `capabilities()`'s bits 1-2. See `decode_capability_determinism()`
  /// (fuse/model/capabilities.hpp) for how an out-of-range field is read.
  /// `std::nullopt` when the component has not declared one: the default
  /// `capabilities()` says nothing about determinism, and silence is not a
  /// `Bitwise` claim (capabilities.hpp). An iteration region warns on an
  /// undeclared level and refuses an explicit `Nondeterministic` one.
  [[nodiscard]] std::optional<DeterminismLevel> determinism_level() const {
    return decode_capability_determinism(capabilities());
  }

  /// Whether `kCapabilityHiddenState` is set -- i.e. whether `advance()`
  /// mutates state this component never bound through `register_fields()`,
  /// which FUSE's byte-for-byte rollback verification therefore cannot see.
  [[nodiscard]] bool carries_hidden_state() const {
    return (capabilities() & kCapabilityHiddenState) != 0;
  }

  /// Whether this component is driven by a caller OUTSIDE fuse::Runner --
  /// i.e. through the participant ABI's `fuse_begin_step()`/
  /// `fuse_end_step()` (src/abi/participant.cpp) rather than by
  /// Runner::Impl's own instruction-stream walk.
  ///
  /// Deliberately a SEPARATE virtual from `capabilities()`, not a new bit
  /// in that bitmask: `capabilities()` is a property a component author
  /// states about their own physics (rollback, determinism, hidden
  /// state), serialized into the model layer's own capability space
  /// (fuse/model/capabilities.hpp); "who drives my advance() calls" is a
  /// runtime wiring fact private to src/abi/, never declared by a real
  /// component and never meaningful outside this one check.
  ///
  /// Checked at exactly the points `Runner::Impl::advance()` (runner.cpp)
  /// and the two direct `comp.advance(ctx)` calls inside
  /// `advance_window()` (iteration.cpp) are about to call this
  /// component's own `advance()`: true suspends the coroutine there
  /// instead (see src/runtime/resume.hpp), handing the caller the exact
  /// `StepContext` and the checkpoint/restore request before running the
  /// (no-op, for the one override below) `advance()` itself on resume.
  ///
  /// Default false: every Component this header's own tests, and every
  /// real physics component, construct is driven in-process. The one
  /// override is `fuse::abi_detail::ExternalStepComponent`
  /// (src/abi/external_step_component.hpp).
  [[nodiscard]] virtual bool is_externally_driven() const noexcept {
    return false;
  }

  /// The residual/linearisation seam: this component's own
  /// `ImplicitComponent` interface, or `nullptr` if it does not implement
  /// one.
  ///
  /// A SEPARATE QUERY, NOT A SECOND BASE CLASS -- the same choice
  /// `is_externally_driven()` makes immediately above, and for the same
  /// reason: most components never implement `ImplicitComponent`, and
  /// multiply inheriting from it would force every one of them to either
  /// override its two pure virtuals with do-nothing bodies or leave
  /// `Component` itself unable to be instantiated. A nullable virtual
  /// query costs one vtable slot and one branch -- exactly what
  /// `is_externally_driven()` already costs -- and needs no RTTI
  /// (`dynamic_cast`) to ask the question, which a second base would.
  ///
  /// Must agree with `capabilities()`: a component that sets
  /// `kCapabilityResidual` but returns `nullptr` here, or vice versa, is
  /// an authoring defect this method cannot itself catch (it is called
  /// only where a real `ComponentSet` already exists; the capability bit
  /// is what a Newton region's validation checks at `initialize()` time,
  /// per `fuse::verify_newton_capability_support()`'s own comment on why
  /// it reads the bit and not this pointer).
  ///
  /// Default `nullptr`: every `Component` this header's own tests
  /// construct, and every component written before this seam existed,
  /// does not implement `ImplicitComponent`.
  [[nodiscard]] virtual ImplicitComponent* implicit() { return nullptr; }
};

namespace detail {

/// Forward iterator over `ComponentSet` that yields `Component&` rather than
/// the `std::unique_ptr<Component>&` the set stores.
///
/// Hiding the owning pointer is not cosmetic: handing out a mutable
/// `unique_ptr&` would let a caller `reset()` or move out an element and
/// break the "every stored component is non-null and uniquely named"
/// invariant that `ComponentSet::add()` enforces and that `find()` relies on.
///
/// Standard container semantics apply: an iterator is invalidated by any
/// `add()`, which may reallocate the underlying vector.
template <class UnderlyingIt, class Ref>
class ComponentIterator {
 public:
  using iterator_category = std::forward_iterator_tag;
  using difference_type   = std::ptrdiff_t;
  using value_type        = std::remove_cv_t<std::remove_reference_t<Ref>>;
  using reference         = Ref;
  using pointer           = std::remove_reference_t<Ref>*;

  ComponentIterator() = default;
  explicit ComponentIterator(UnderlyingIt it) : it_(it) {}

  [[nodiscard]] reference operator*() const { return **it_; }
  [[nodiscard]] pointer operator->() const { return it_->get(); }

  ComponentIterator& operator++() {
    ++it_;
    return *this;
  }
  ComponentIterator operator++(int) {
    ComponentIterator tmp = *this;
    ++it_;
    return tmp;
  }

  [[nodiscard]] friend bool operator==(const ComponentIterator& a,
                                       const ComponentIterator& b) noexcept {
    return a.it_ == b.it_;
  }

 private:
  UnderlyingIt it_{};
};

}  // namespace detail

/// The set of in-process components taking part in a run, keyed by name.
///
/// Owns its components: they are destroyed with the set, through
/// `Component`'s virtual destructor. Move-only, and moving the set moves only
/// the owning pointers -- the components themselves never move, which is what
/// keeps every `FieldRef` bound during `register_fields()` valid.
///
/// Insertion order is the iteration order and is the order `at()` indexes.
/// It is *not* an execution order; the schedule comes from the graph.
class ComponentSet {
 public:
  using iterator = detail::ComponentIterator<
      std::vector<std::unique_ptr<Component>>::iterator, Component&>;
  using const_iterator = detail::ComponentIterator<
      std::vector<std::unique_ptr<Component>>::const_iterator,
      const Component&>;

  /// Takes ownership of `c`.
  ///
  /// Throws `std::invalid_argument` if `c` is null, if `c->name()` is empty,
  /// or if a component with that name is already present. All three are
  /// programmer errors in the harness that assembles the run, so they follow
  /// the same convention as `Rational`'s and `FieldRef`'s guards: an
  /// unconditional throw, not an `assert()` that vanishes under `-DNDEBUG`.
  ///
  /// Duplicate names in particular must not be tolerated: `find()` would
  /// return the first match and the second component would never be scheduled
  /// or exchanged, which is a wrong-physics failure with no symptom.
  ///
  /// On a throw the set is unchanged and `c` is destroyed as the exception
  /// unwinds (it was moved from at the call boundary).
  void add(std::unique_ptr<Component> c) {
    if (c == nullptr) {
      FUSE_THROW(InvalidArgument, "{}", "ComponentSet::add: null component");
    }
    const std::string n = c->name();
    if (n.empty()) {
      FUSE_THROW(InvalidArgument, "{}", 
          "ComponentSet::add: component name is empty; it must match a "
          "ComponentDecl name in the coupling graph");
    }
    if (find(n) != nullptr) {
      FUSE_THROW(InvalidArgument, "{}", 
          "ComponentSet::add: a component named '" + n + "' is already present");
    }
    components_.push_back(std::move(c));
  }

  /// The component named `name`, or `nullptr` if there is none. Never returns
  /// a dangling pointer, but the pointee is owned by this set: it dies with
  /// the set and must not be deleted by the caller.
  [[nodiscard]] Component* find(std::string_view name) {
    for (const auto& c : components_) {
      if (c->name() == name) return c.get();
    }
    return nullptr;
  }
  [[nodiscard]] const Component* find(std::string_view name) const {
    for (const auto& c : components_) {
      if (c->name() == name) return c.get();
    }
    return nullptr;
  }

  [[nodiscard]] std::size_t size() const noexcept { return components_.size(); }
  [[nodiscard]] bool empty() const noexcept { return components_.empty(); }

  /// Bounds-checked in every build configuration; throws `std::out_of_range`.
  [[nodiscard]] Component& at(std::size_t i) { return *components_.at(i); }
  [[nodiscard]] const Component& at(std::size_t i) const {
    return *components_.at(i);
  }

  // Iteration yields Component& / const Component&; see ComponentIterator.
  [[nodiscard]] iterator begin() { return iterator{components_.begin()}; }
  [[nodiscard]] iterator end() { return iterator{components_.end()}; }
  [[nodiscard]] const_iterator begin() const {
    return const_iterator{components_.begin()};
  }
  [[nodiscard]] const_iterator end() const {
    return const_iterator{components_.end()};
  }
  [[nodiscard]] const_iterator cbegin() const { return begin(); }
  [[nodiscard]] const_iterator cend() const { return end(); }

 private:
  // Node-based ownership on purpose: push_back reallocates this vector, but
  // it moves unique_ptrs, never the Components, so a Component* handed out by
  // find() or at() -- and every FieldRef bound into it -- survives a later
  // add(). Never hold the *iterators* across an add().
  std::vector<std::unique_ptr<Component>> components_;
};

}  // namespace fuse
