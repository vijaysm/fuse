#pragma once

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "soma/base/rational.hpp"
#include "soma/base/result.hpp"
#include "soma/field/registry.hpp"
#include "soma/graph/coupling_graph.hpp"

namespace soma {

/// The half-open time interval `[t_begin, t_end)` a single
/// `Component::advance()` call must cover, plus where the call sits in the
/// schedule.
///
/// Times are `Rational`, never `double` (spec §4.3): 1/3 and 1/7 subcycling
/// must decompose into slabs with no drift over 10^6 steps, which
/// floating-point accumulation destroys silently.
///
/// `step` counts coupling steps from the start of the run. `iteration` is the
/// 0-based index of the current sweep inside an iteration region (spec §5.3)
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
  /// crash — it is a plausible-looking wrong answer.
  [[nodiscard]] constexpr bool valid() const noexcept {
    return t_begin <= t_end && step >= 0 && iteration >= 0;
  }
};

/// A participating physics code, integrated in-process (spec §6 mode 1).
///
/// Deliberately minimal: name yourself, bind your storage, advance in time.
/// Everything else SOMA needs — remap, accumulation, scheduling — it derives
/// from the `CouplingGraph`, not from the component.
///
/// OWNERSHIP MODEL (spec §4.2). The component owns its field arrays. SOMA
/// never allocates, copies, or frees them; it wraps them in non-owning
/// `FieldRef`s. SOMA owns only its own intermediates (accumulators, waveform
/// buffers, remap targets, previous iterates). A `Component` held in a
/// `ComponentSet` is owned by that set and destroyed with it, through this
/// base pointer — hence the virtual destructor.
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
  /// graph node it implements — that string is the only link between the
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
  /// (spec §4.2).
  ///
  /// The implementer must, for each `FieldId` in
  /// `graph.component(id).fields`, call `registry.bind(id, FieldRef{...})`
  /// with a pointer to storage it owns and a `Layout` describing how to walk
  /// it. `bind()` rejects a null pointer, an invalid `FieldId`, a layout whose
  /// strides alias or overflow, and a second bind of the same field; propagate
  /// the returned error rather than ignoring it.
  ///
  /// LIFETIME. `FieldRef` is non-owning. The registered address must stay
  /// valid and *fixed* for the whole run: SOMA resolves once here and
  /// dereferences in the hot loop. Registering `vec.data()` and later growing,
  /// reassigning, or reallocating `vec` dangles every view. This includes
  /// `restore_state()` — restore a checkpoint element-wise, do not
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
  /// `ctx.t_begin`, already remapped and time-interpolated by SOMA; and
  /// `ctx.valid()` (the driver checks it — see `StepContext::valid()`).
  ///
  /// On return the implementer must guarantee: every field it declared
  /// `Intent::Provides` holds values valid at `ctx.t_end`, written *in place*
  /// through the storage registered in `register_fields()`; and no registered
  /// pointer has moved. Writing to a `Requires` field is undefined from SOMA's
  /// point of view — SOMA may overwrite it at any later exchange.
  ///
  /// `ctx` is borrowed for the duration of the call; copy it if you need it
  /// afterwards.
  ///
  /// Inside an iteration region `advance()` is called repeatedly with the same
  /// `t_begin`/`t_end` and increasing `ctx.iteration`, each call preceded by a
  /// `restore_state()`. It must therefore be a pure function of (restored
  /// state, inputs) — accumulating a side effect per call, such as appending
  /// to an output file or incrementing an internal step counter, produces
  /// results that depend on the iteration count.
  ///
  /// Returning an error aborts the step; SOMA does not retry.
  virtual Result<void> advance(const StepContext& ctx) = 0;

  /// Checkpoint the state `advance()` mutates, so the same interval can be
  /// re-advanced (spec §5.3).
  ///
  /// Spec §5.3: SOMA derives the checkpoint set from dataflow over the SSA
  /// form, tells the component which of its state crosses the region
  /// boundary, and *verifies at runtime that it was saved* — unlike preCICE,
  /// which pushes `requiresWritingCheckpoint()` onto the user. Phase 7
  /// Task 7.2 raises an error for a component that appears in an iteration
  /// region with `supports_rollback() == false`; the hooks are declared now so
  /// the interface does not change under existing implementers later.
  ///
  /// R16 CAVEAT, because the sentence above reads as a stronger guarantee
  /// than it is: the runtime verification is a `memcmp` against REGISTERED
  /// field storage only (the arcs `register_fields()` bound; see
  /// `Runner::Impl::run_region`, src/runtime/iteration.cpp), one per
  /// component the derived checkpoint set names. It cannot see, and
  /// therefore cannot verify, any state private to the component that was
  /// never registered — the paragraph below asks the implementer to save
  /// that too, but nothing in SOMA checks that they did. "SOMA verifies the
  /// rollback" means "SOMA verifies the registered part of the rollback."
  ///
  /// The default is a successful no-op, correct only for a component that is
  /// never iterated. An implementer that overrides these must save everything
  /// `advance()` writes, including state private to the component that SOMA
  /// cannot see, and must not move any registered field storage while doing
  /// so. Only one checkpoint need be retained; SOMA saves at most once per
  /// iteration region entry.
  ///
  /// Outside an iteration region SOMA may call `save_state()` and never call
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

  /// Whether this component implements a real `save_state()`/`restore_state()`
  /// pair.
  ///
  /// Must be constant for the object's lifetime and must be `true` if and only
  /// if both hooks are genuinely implemented. Defaults to `false`, which is
  /// the honest answer for the default no-op hooks: a component that silently
  /// claimed rollback support while doing nothing would make an iteration
  /// region converge on stale state.
  [[nodiscard]] virtual bool supports_rollback() const { return false; }
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
/// the owning pointers — the components themselves never move, which is what
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
      throw std::invalid_argument("ComponentSet::add: null component");
    }
    const std::string n = c->name();
    if (n.empty()) {
      throw std::invalid_argument(
          "ComponentSet::add: component name is empty; it must match a "
          "ComponentDecl name in the coupling graph");
    }
    if (find(n) != nullptr) {
      throw std::invalid_argument(
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
  // find() or at() — and every FieldRef bound into it — survives a later
  // add(). Never hold the *iterators* across an add().
  std::vector<std::unique_ptr<Component>> components_;
};

}  // namespace soma
