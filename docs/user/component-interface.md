# Writing a component

The contract a C++ class must honour to take part in a FUSE run. [](quickstart.md) shows one
component inside a complete program; this page is the reference for every method of
`fuse::Component`, the capability bits, rollback, restart hooks and the Newton interface. Read it
when you are wrapping your own solver, or when `Runner::initialize()` refuses a component you wrote.

```cpp
#include <fuse/runtime/component.hpp>   // Component, StepContext, ImplicitComponent, ComponentSet
```

## The smallest component

A component names itself, binds its arrays once, and advances in time. Everything else FUSE
derives from the `CouplingGraph`.

```cpp
class Source final : public fuse::Component {
 public:
  explicit Source(std::size_t n) : u_(n, 0.0) {}

  std::string name() const override { return "source"; }

  fuse::Result<void> register_fields(fuse::FieldRegistry& reg,
                                     const fuse::CouplingGraph& g) override {
    const fuse::ComponentId me = *g.find_component(name());
    for (const fuse::FieldId f : g.component(me).fields) {   // exactly one: "u"
      if (auto r = reg.bind(f, fuse::FieldRef{u_.data(),
                                              fuse::Layout{.stride = 1, .count = u_.size()},
                                              fuse::MemorySpace::Host});
          !r.ok()) {
        return r;
      }
    }
    return fuse::Result<void>::success();
  }

  fuse::Result<void> advance(const fuse::StepContext& ctx) override {
    const double t = ctx.t_end.to_double();        // "u" must be valid at t_end
    for (std::size_t i = 0; i < u_.size(); ++i) u_[i] = t * (1.0 + double(i));
    return fuse::Result<void>::success();
  }

 private:
  std::vector<double> u_;   // never resized after register_fields()
};
```

The matching consumer binds a `Requires` field the same way and only reads it:

```cpp
class Sink final : public fuse::Component {
 public:
  explicit Sink(std::size_t n) : u_in_(n, 0.0) {}
  std::string name() const override { return "sink"; }

  fuse::Result<void> register_fields(fuse::FieldRegistry& reg,
                                     const fuse::CouplingGraph& g) override {
    const fuse::ComponentId me = *g.find_component(name());
    for (const fuse::FieldId f : g.component(me).fields) {   // exactly one: "u_in"
      if (auto r = reg.bind(f, fuse::FieldRef{u_in_.data(),
                                              fuse::Layout{.stride = 1, .count = u_in_.size()},
                                              fuse::MemorySpace::Host});
          !r.ok()) {
        return r;
      }
    }
    return fuse::Result<void>::success();
  }

  fuse::Result<void> advance(const fuse::StepContext& ctx) override {
    // u_in_ is valid at ctx.t_begin on entry; FUSE wrote it, this class only reads it.
    double sum = 0.0;
    for (double v : u_in_) sum += v;
    std::cout << "sink step " << ctx.step << " [" << ctx.t_begin.to_string() << ", "
              << ctx.t_end.to_string() << ")  sum(u_in) = " << sum << "\n";
    return fuse::Result<void>::success();
  }

 private:
  std::vector<double> u_in_;
};
```

Coupled over a four-cell `generated:line:4` with `source` stepping at 1/2 and `sink` at 1, channel
`u_link` with `.accumulate(fuse::AccumOp::Mean)`, and run for two periods with the harness from
[](quickstart.md), the program prints:

```text
sink step 0 [0, 1)  sum(u_in) = 0
sink step 1 [1, 2)  sum(u_in) = 7.5
periods run: 2
```

Expected output (exact digits can vary by platform). Step 0 reads zeros because an explicit channel
delivers the previous period's value; step 1 reads the mean of `source`'s two sub-steps at
t = 1/2 and t = 1, which is 0.75 · (1 + 2 + 3 + 4).

## `StepContext`

Every `advance()` call receives the interval it must cover and where that call sits in the
schedule:

| Field | Type | Meaning |
|---|---|---|
| `t_begin`, `t_end` | `fuse::Rational` | The half-open interval `[t_begin, t_end)` to integrate over. Always exact fractions; convert with `to_double()` at the edge of your own physics |
| `step` | `std::int64_t` | Coupling steps since the start of the run, counted in this component's own steps |
| `iteration` | `int` | 0 outside an iteration region; the 0-based sweep index inside one |

`ctx.dt()` returns `t_end - t_begin`. `ctx.valid()` (`t_begin <= t_end`, `step >= 0`,
`iteration >= 0`) is checked by the Runner before every call, so a component never sees a negative
step; a context that fails the check is refused with `E-RT-STEP`. `ctx` is borrowed for the
duration of the call: copy it if you need it afterwards.

## The virtual methods

`Component` cannot be copied or moved. The three pure-virtual methods are the whole mandatory
surface; everything below them has a default.

### `std::string name() const`

Must equal the `ComponentDecl::name` of the graph node this object implements, be non-empty, and
return the same value for the object's whole lifetime. The name is the only link between the
authored graph and the running object: `ComponentSet::add()` rejects an empty or duplicate name,
and `Runner::initialize()` refuses with `E-RT-COMPONENT` when a graph component has no object of
that name or an object has no graph component. Returns by value, so call it at setup, not in the
step loop.

### `Result<void> register_fields(FieldRegistry&, const CouplingGraph&)`

Called exactly once, before the first `advance()`. For each `FieldId` in
`graph.component(id).fields`, call `registry.bind(id, FieldRef{pointer, Layout, MemorySpace})` with
storage the component owns and propagate any error. `bind()` refuses:

| Code | Cause |
|---|---|
| `E-FIELD-ID` | an invalid `FieldId` |
| `E-FIELD-NULL` | a null pointer with `count > 0` (null with `count == 0` is allowed, for a rank that owns nothing) |
| `E-FIELD-LAYOUT` | strides that alias or overflow |
| `E-FIELD-REBIND` | a second bind of the same field |

On error the registry is unchanged, and the Runner aborts setup. `graph` is borrowed for the call
only; do not keep a pointer into it.

:::{important}
**Register once, write in place.** `FieldRef` is non-owning and FUSE resolves the address here,
once, then dereferences it in the hot loop. Growing, reassigning, swapping or reallocating the
container afterwards dangles every view with no error. This includes `restore_state()`: copy a
checkpoint back element-wise (`std::copy`), never `state_ = saved_;`.
:::

### `Result<void> advance(const StepContext& ctx)`

Integrate this component's own state from `ctx.t_begin` to `ctx.t_end`.

- **On entry** every `Requires` field holds values valid at `t_begin`, already remapped, converted
  and time-interpolated by FUSE.
- **On return** every `Provides` field must hold values valid at `t_end`, written in place through
  the registered storage, and no registered pointer may have moved.
- Writing into a `Requires` field is undefined from FUSE's point of view: it is overwritten at the
  next exchange.
- Returning an error aborts the step. FUSE does not retry.
- Inside an iteration region `advance()` is called repeatedly over the same interval, with
  increasing `ctx.iteration` and a `restore_state()` before each call. It must therefore be a pure
  function of (restored state, inputs): no appending to an output file, no step counter that is not
  part of the checkpoint.

All methods are called from the component's own rank, from one thread, never concurrently. A
component may use threads or a device queue internally, but every externally visible effect must
be complete when the method returns.

### `save_state()` / `restore_state()`

`Result<void> save_state()` and `Result<void> restore_state()` default to successful no-ops, which
is correct only for a component that is never iterated. FUSE keeps at most one checkpoint per
iteration region entry, so one slot is enough. See [Rollback](#rollback-and-iteration-regions).

### Restart hooks

`write_restart(RestartWriter&)` / `read_restart(RestartReader&)` (Form A, keyed by global ids) and
`save_restart_blob()` / `load_restart_blob(bytes, written_by_version)` /
`restart_blob_version()` (Form B, an opaque versioned blob) default to no-ops. They are called at
period boundaries only. `write_restart()` must not mutate state, because a restore calls it a
second time to learn the component's current ids. [](restart.md) covers both forms.

### `std::uint64_t capabilities() const`

One bitmask stating everything the component claims about itself; default 0. See
[Capabilities](#capabilities).

### `ImplicitComponent* implicit()`

Returns the component's `ImplicitComponent` interface, or `nullptr` (the default). See
[Joining a Newton region](#joining-a-newton-region).

### `bool is_externally_driven() const noexcept`

Default `false`. The only override lives inside the participant ABI, where an external code owns
the time loop ([](c-abi.md)). A component you write in C++ never overrides it.

### Non-virtual readers

`supports_rollback()`, `determinism_level()` (an `std::optional<DeterminismLevel>`),
`carries_hidden_state()`, `supports_restart_form_a()` and `supports_restart_form_b()` decode
`capabilities()`. They are deliberately non-virtual, so a legacy
`bool supports_rollback() const override` no longer compiles instead of silently shadowing the bit.

## Fields: which array goes where

**`Provides`** points at the state array itself: the values the component computes, read by others.
**`Requires`** points at a separate input buffer that FUSE fills before each step; never alias it
with state. **`Bidirectional`** is both: the component writes it and a channel may also write into
it, so it is bound once and used both ways.

### Multi-component (vector) fields

`FieldSpec::components` (1 scalar, 3 vector, 9 tensor) sets the number of values per entity.
`Layout` describes how the array is walked, in units of `double`:

```cpp
fuse::Layout{.stride = 3, .component_stride = 1, .count = n, .components = 3}  // interleaved xyzxyz
```

`component_stride == 0` means interleaved (1). A blocked layout (`stride = 1`,
`component_stride = count`) is a valid `FieldRef`, but the Runner's interpreter walks every buffer
as a flat, component-fastest array and refuses a layout that is not (`E-RT-LAYOUT`). Register
interleaved storage. `Runner::initialize()` also checks that `count` equals the region size the
plan computed and `components` equals the field's declaration (`E-RT-SIZE`), and that every
component-owned field is bound (`E-RT-UNBOUND`).

### Nodal and cell fields, and what `count` must be

How many entities a field has is decided from the field's discretisation and order, never guessed:

| `FieldSpec` | Values per | Size |
|---|---|---|
| `Discretization::FV` (default), or `DiscontinuousGLL` with `order = 0` | cell | the region's entity count |
| `ContinuousGLL` with `order = 1` | vertex | the whole mesh's vertex count |
| anything else | — | refused, `E-EXTENT-DISC` |

A nodal field on a named entity subset is refused with `E-EXTENT-REGION`, and one region carrying
both a cell field and a nodal field is refused with `E-EXTENT-MIXED`. On a distributed mesh the
count includes ghost entities (`MeshView::entity_count`), and a conservation sum skips them with
`MeshView::is_owned`; see [](meshes.md). Transfers from or to a nodal field use consistent
interpolation and its transpose ([](interface-transfer.md)).

### Typed fields and extents

The registry binds `double` storage only. Restart datasets are the one place other element types
appear (`RestartWriter::put_field<T>`, [](restart.md)). To read part of a field without copying,
wrap the `FieldRef` in a lazy `FieldSpan` and slice it with half-open extents:

```cpp
auto first_component = fuse::FieldSpan<>::from(ref)
                           .owned(n_owned)                       // leading owned elements
                           .slice({2, fuse::end}, fuse::all)     // elements 2.. , all components
                           .component(0);
```

Every stage is a strided view over the same memory; a reversed or out-of-range extent is refused
(`E-FIELD-EXTENT-REVERSED`, `E-FIELD-EXTENT-RANGE`), never clamped. [](field-pipeline.md) has the
full grammar and the lifetime rules.

### Device fields

Bind device memory by passing `fuse::MemorySpace::Device` with a pointer that lives in Kokkos'
default execution space's memory space:

```cpp
// data_ is a Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space>
reg.bind(f, fuse::FieldRef{data_.data(), fuse::Layout{.stride = 1, .count = data_.extent(0)},
                           fuse::MemorySpace::Device});
```

Two conditions apply. The Runner must run in device mode,
`RunnerConfig::execution_space = RunnerConfig::RunnerExecutionSpace::Device` (or
`FUSE_EXECUTION_SPACE=device`); a device-registered field under a host Runner is refused with
`E-RT-SPACE`. And the build's Kokkos must have a device backend; asking for `Device` without one is
refused with `E-RT-DEVICE`. There is no device capability bit: placement is a property of each
field. See [](power-users.md) for execution spaces.

## Rollback and iteration regions

A component inside an iteration region ([](concepts.md#iteration)) is re-advanced over the same
interval on every sweep. FUSE derives, from dataflow, which components must be restorable, and
checks the claim twice:

1. At `Runner::initialize()`, every component in the derived checkpoint set must set
   `kCapabilityRollback`; otherwise `E-ITER-NOROLLBACK` names it.
2. After every `restore_state()`, the registered `Provides` storage is compared byte for byte
   with what it held at `save_state()`; a mismatch is `E-ITER-ROLLBACK`. The count is in
   `RunStats::checkpoint_verifications`.

That verification covers registered storage only. State that `advance()` mutates but never
registered (a time stamp, a solver history, a counter) is **hidden state**: declare it with the
hidden-state bit and carry it in the checkpoint yourself, because nothing in FUSE can see it.

```cpp
class Relax final : public fuse::Component {
 public:
  Relax(std::string name, std::size_t n)
      : name_(std::move(name)), out_(n, 0.0), in_(n, 0.0), saved_out_(n, 0.0) {}

  std::string name() const override { return name_; }

  fuse::Result<void> register_fields(fuse::FieldRegistry& reg,
                                     const fuse::CouplingGraph& g) override {
    const fuse::ComponentId me = *g.find_component(name_);
    for (const fuse::FieldId f : g.component(me).fields) {
      // Provides -> the state array; Requires -> a separate buffer FUSE overwrites.
      std::vector<double>& storage =
          g.field(f).intent == fuse::Intent::Provides ? out_ : in_;
      if (auto r = reg.bind(f, fuse::FieldRef{storage.data(),
                                              fuse::Layout{.stride = 1, .count = storage.size()},
                                              fuse::MemorySpace::Host});
          !r.ok()) {
        return r;
      }
    }
    return fuse::Result<void>::success();
  }

  fuse::Result<void> advance(const fuse::StepContext& ctx) override {
    for (std::size_t i = 0; i < out_.size(); ++i) out_[i] = 0.5 * in_[i] + 1.0;
    ++advance_calls_;   // hidden state: mutated, never registered, so it is saved below
    ++total_calls_;     // diagnostic only, deliberately NOT part of the checkpoint
    (void)ctx;
    return fuse::Result<void>::success();
  }

  fuse::Result<void> save_state() override {
    std::copy(out_.begin(), out_.end(), saved_out_.begin());   // element-wise
    saved_calls_ = advance_calls_;
    return fuse::Result<void>::success();
  }

  fuse::Result<void> restore_state() override {
    std::copy(saved_out_.begin(), saved_out_.end(), out_.begin());  // never `out_ = saved_out_`
    advance_calls_ = saved_calls_;
    return fuse::Result<void>::success();
  }

  std::uint64_t capabilities() const override {
    return fuse::make_capabilities(/*rollback=*/true, fuse::DeterminismLevel::Bitwise,
                                   /*hidden_state=*/true);
  }

  double out0() const { return out_[0]; }
  long advance_calls() const { return advance_calls_; }
  long total_calls() const { return total_calls_; }

 private:
  std::string name_;
  std::vector<double> out_, in_, saved_out_;
  long advance_calls_ = 0, saved_calls_ = 0, total_calls_ = 0;
};
```

Two instances, `A` providing `a` and requiring `b_in`, `B` providing `b` and requiring `a_in`,
form the fixed-point problem a = ½b + 1, b = ½a + 1 whose solution is a = b = 2. Both channels go
into one iteration region (one of them lagged, as every cycle must be), measured with
`Converged::Relative(1e-10, "a")` and `"b"`, Picard iteration (`AccelKind::None`), 100 sweeps at
most. One period prints:

```text
converged: yes after 34 sweeps
a[0] = 2  (fixed point 2)
A::advance() calls made: 34, recorded in the restored state: 1
checkpoint verifications: 66
```

Expected output (exact digits can vary by platform). The counter that was part of the checkpoint
reads 1 at the end: every sweep restored it and advanced once, which is exactly the state a single
converged step should leave behind. The counter that was left out of the checkpoint reads 34, and a
component that wrote such a counter into its output would produce a different answer for every
`max_iterations`.

Exhausting `max_iterations` is counted in `RunStats::windows_not_converged`;
`RunnerConfig::abort_on_nonconvergence = true` turns it into `E-RT-NOCONVERGE`. Rollback is
in-memory and lives within one process; restart to disk is a separate mechanism ([](restart.md)).

## Capabilities

`capabilities()` returns one 64-bit mask. The bits, from `fuse/model/capabilities.hpp`:

| Bit | Constant | Claim |
|---|---|---|
| 0 | `kCapabilityRollback` | `save_state()` / `restore_state()` are a real pair |
| 1–2 | determinism field (`kCapabilityDeterminismMask`) | 0 undeclared, 1 `Bitwise`, 2 `RunToRun`, 3 `Nondeterministic` |
| 3 | `kCapabilityHiddenState` | `advance()` mutates state that was never registered |
| 4 | `kCapabilityResidual` | `implicit()` returns an `ImplicitComponent` whose `evaluate_residual()` works |
| 5 | `kCapabilityLinearization` | the `LinearizationToken` from `linearization()` is usable |
| 6 | `kCapabilityRestartFormA` | `write_restart()` / `read_restart()` are implemented |
| 7 | `kCapabilityRestartFormB` | the restart blob methods are implemented |
| 8–63 | `kCapabilityReservedMask` | reserved, must be zero |

`fuse::make_capabilities(bool rollback, DeterminismLevel, bool hidden_state)` builds the first three
fields; OR the remaining constants onto it:

```cpp
std::uint64_t capabilities() const override {
  return fuse::make_capabilities(true, fuse::DeterminismLevel::Bitwise, true) |
         fuse::kCapabilityResidual | fuse::kCapabilityRestartFormA;
}
```

`encode_capability_determinism(level)` and `decode_capability_determinism(mask)` convert the
determinism field on their own. The value must be constant for the object's lifetime.

### Determinism levels

`fuse::DeterminismLevel` has three values, and silence is a fourth state:

| Declared | Meaning | Inside an iteration region |
|---|---|---|
| (nothing) | the component has made no claim | allowed, with `W-ITER-DETERMINISM-UNDECLARED` |
| `Bitwise` | bit-identical for identical input, on identical hardware and across rank counts: no RNG, no unordered reduction | allowed |
| `RunToRun` | reproducible run to run on one configuration only | allowed |
| `Nondeterministic` | stochastic by construction (an RNG, a library with unspecified reduction order) | refused, `E-ITER-NONDETERMINISTIC` |

A nondeterministic component is fine in an explicit coupling. Inside a region it is refused
because the convergence test compares successive iterates and cannot tell noise from a residual.
The default mask decodes to *undeclared*, not to `Bitwise`: a component that says nothing has not
made the strongest claim in the enum.

### Restart bits

A component with neither restart bit is skipped by `Runner::write_restart_fields()`. Form A allows
restart at a different rank count; Form B alone restores only at the same decomposition, and a
rank-count change is refused collectively with `E-RESTART-FORMB-ONLY-RANK-CHANGE` if any component
offers Form B without Form A. [](restart.md) has the writer and reader API.

## Joining a Newton region

A region solved with `.newton(cfg)` (or `.accelerate()` with `AccelKind::Newton`) needs each
member's own block of the fixed-point map at a frozen time level, without advancing it. That is
the `fuse::ImplicitComponent` interface, reached through `Component::implicit()` rather than
through a second base class that every ordinary component would have to stub out.

```cpp
class ImplicitComponent {
 public:
  virtual Result<void> evaluate_residual(const StepContext& ctx,
                                         std::span<const double> interface_in,
                                         std::span<double> residual_out) = 0;
  virtual LinearizationToken* linearization() { return nullptr; }
};
```

`evaluate_residual()` writes G_i(x): the values this component would `Provide` at `ctx.t_end` given
the `Requires` values in `interface_in`. It returns G_i alone; the Runner forms the residual
F(x) = x − G(x) itself. `interface_in` is this component's own slice of the region's stacked
interface vector (the lengths of its `Requires` fields in the region, concatenated) and
`residual_out` the lengths of its `Provides` fields; the two need not be equal. The call must not
mutate any state `save_state()` would capture, and must return bit-identical output for identical
input, because a Krylov iteration revisits points and a matrix-free Jacobian action perturbs around
them. An error aborts the Newton step; there is no retry. For a window-scoped member, `ctx` spans the
whole window and both spans use the (element, sample) layout ([](coupling-windows.md)).

`linearization()` returns a borrowed `LinearizationToken*` for the point of the most recent
`evaluate_residual()` call, valid until the next `evaluate_residual()` or `advance()`, or
`nullptr`, in which case Newton falls back to a matrix-free directional derivative for this block.
A token implements, on this component's own block only:

| Method | Contract |
|---|---|
| `apply_jacobian(x, y)` | y = J x, `x.size() == cols()`, `y.size() == rows()` |
| `apply_jacobian_transpose(x, y)` | y = Jᵀ x; may return an error if the block has no meaningful transpose |
| `apply_preconditioner(r, z)` | z ≈ M⁻¹ r for the diagonal block; the identity is always legal |
| `rows()`, `cols()` | checked against the region's block sizes, `E-NEWTON-SHAPE` on disagreement |
| `signature()` | changes whenever the linearisation point moves |

A sketch that compiles against the headers:

```cpp
class DiagonalToken final : public fuse::LinearizationToken {
 public:
  explicit DiagonalToken(std::size_t n) : diag_(n, 0.0) {}
  void set_diagonal(const std::vector<double>& d) { diag_ = d; ++signature_; }

  fuse::Result<void> apply_jacobian(std::span<const double> x, std::span<double> y) override {
    for (std::size_t i = 0; i < diag_.size(); ++i) y[i] = diag_[i] * x[i];
    return fuse::Result<void>::success();
  }
  fuse::Result<void> apply_jacobian_transpose(std::span<const double> x,
                                              std::span<double> y) override {
    return apply_jacobian(x, y);              // symmetric block
  }
  fuse::Result<void> apply_preconditioner(std::span<const double> r,
                                          std::span<double> z) override {
    for (std::size_t i = 0; i < diag_.size(); ++i) z[i] = r[i] / (1.0 - diag_[i]);
    return fuse::Result<void>::success();
  }
  std::size_t rows() const override { return diag_.size(); }
  std::size_t cols() const override { return diag_.size(); }
  std::uint64_t signature() const override { return signature_; }

 private:
  std::vector<double> diag_;
  std::uint64_t signature_ = 0;
};

class ImplicitRelax final : public fuse::Component, public fuse::ImplicitComponent {
 public:
  // name(), register_fields(), advance(), save_state(), restore_state() as in Relax above.

  // The same G_i as advance(), on a caller-supplied input, touching no state at all.
  fuse::Result<void> evaluate_residual(const fuse::StepContext&,
                                       std::span<const double> interface_in,
                                       std::span<double> residual_out) override {
    if (interface_in.size() != out_.size() || residual_out.size() != out_.size()) {
      return fuse::Failure{"E-MYCODE-SHAPE", "relax: unexpected block size"};
    }
    std::vector<double> diag(out_.size());
    for (std::size_t i = 0; i < out_.size(); ++i) {
      residual_out[i] = g_i(interface_in[i]);
      diag[i] = dg_i(interface_in[i]);
    }
    token_.set_diagonal(diag);                // linearisation point moved
    return fuse::Result<void>::success();
  }
  fuse::LinearizationToken* linearization() override { return &token_; }
  fuse::ImplicitComponent* implicit() override { return this; }

  std::uint64_t capabilities() const override {
    return fuse::make_capabilities(true, fuse::DeterminismLevel::Bitwise, false) |
           fuse::kCapabilityResidual | fuse::kCapabilityLinearization;
  }

 private:
  static double g_i(double x) { return 0.5 * x + 1.0; }
  static double dg_i(double) { return 0.5; }
  std::vector<double> out_, in_, saved_out_;
  DiagonalToken token_;
};
```

What the region checks at `initialize()`, over the same checkpoint set as rollback:

- `.newton()` sets the region's `requires_residual_capability`; a member without
  `kCapabilityResidual` is refused with `E-NEWTON-CAPABILITY`.
- `NewtonConfig::use_block_jacobi_preconditioner = true` additionally sets
  `requires_linearization_capability`; a member without `kCapabilityLinearization` is refused with
  the same code. Without it the token is optional and the preconditioner is the identity.
- The bit is what is checked, not the pointer: setting `kCapabilityResidual` while `implicit()`
  returns `nullptr` is an authoring defect the check cannot catch.
- Rollback is still required: Newton re-advances members exactly where Picard would.

`NewtonConfig::residual_path` selects how F is evaluated: `Advance` (the default) calls `advance()`
under save/restore; `Direct` calls `evaluate_residual()` on every member instead, with no
checkpointing, for ordinary (non-window) regions. [](linear-algebra.md) describes the solver these
calls feed, and [](petsc-reaction-diffusion.md) is a complete implicit component whose token wraps
an assembled PETSc matrix.

## `ComponentSet`

`fuse::ComponentSet` owns the component objects for a run and hands them to the `Runner`:

```cpp
fuse::ComponentSet components;
components.add(std::make_unique<Source>(n));     // throws std::invalid_argument on null,
components.add(std::make_unique<Sink>(n));       // empty or duplicate name
fuse::Component* s = components.find("sink");    // nullptr if absent; owned by the set
for (fuse::Component& c : components) { /* insertion order, not execution order */ }
components.size(); components.empty(); components.at(0);   // at() is bounds-checked
```

Moving the set moves only the owning pointers; the components themselves never move, which is what
keeps every `FieldRef` bound in `register_fields()` valid.

## Error handling

- Return `fuse::Result<void>`; build an error with `fuse::Failure{"E-YOURCODE-...", message}`.
  Never `assert` a physical condition: in a Release build the assertion vanishes and an unstable
  scheme produces growing numbers without complaint.
- An error from `register_fields()` aborts setup. An error from `advance()`,
  `save_state()`, `restore_state()` or `evaluate_residual()` aborts the step: the code and message
  come back unchanged from `Runner::run_period()` / `run()`. Nothing is retried.
- In a multi-rank run, the other ranks learn of the failure at the next rank-agreement point
  (`Epoch`) and finish the period with `E-RT-EPOCH-PEER` instead of blocking in a later collective.
  The failing rank reports the cause; read that one.
- Programmer errors in the harness (`ComponentSet::add` with a bad name, `at()` out of range)
  throw, in every build type.
- `FUSE_ON_ERROR=debugger` stops in a debugger on the failing rank
  ([](troubleshooting.md#debugging-a-failure)).

## Checklist

- [ ] `name()` matches `ComponentDecl::name` exactly and never changes.
- [ ] Every `FieldId` in `graph.component(id).fields` is bound; `Provides` to state, `Requires` to a
      separate buffer; interleaved layout with the region's count.
- [ ] No registered container is resized, reassigned or swapped after `register_fields()`.
- [ ] `advance()` writes `Provides` fields valid at `t_end`, in place, and returns `Result`.
- [ ] Inside an iteration region: `save_state()` / `restore_state()` copy element-wise and cover
      hidden state; `kCapabilityRollback` is set; `advance()` has no per-call side effect.
- [ ] `capabilities()` declares a determinism level and the hidden-state bit truthfully.
- [ ] For a Newton region: `implicit()` returns `this`, `evaluate_residual()` is pure and returns
      G_i, `kCapabilityResidual` (and `kCapabilityLinearization` when a token is offered) is set.
- [ ] For restart: the matching Form A/B bit is set and `write_restart()` is pure.

## See also

- [](quickstart.md): the harness around a component.
- [](concepts.md): the model the contract serves.
- [](field-pipeline.md): reading field data lazily.
- [](restart.md): Form A and Form B.
- [](convergence.md) and [](dn-heat.md): iteration regions and Newton in a worked example.
- [](c-abi.md), [](fortran.md), [](providers.md): components written outside C++.
