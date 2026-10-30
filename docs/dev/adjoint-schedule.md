# The adjoint schedule (reverse mode through the coupler)

Stage U6 of `plans/2026-10-22-universal-number-types.md`. `Runner::initialize_adjoint()` and
`Runner::run_period_adjoint()` run one period of a coupling backwards: the adjoint of every coupling
operator is applied by FUSE, and the adjoint of every component step by the component. The user
guide section is [](../user/field-pipeline.md) ("Reverse mode"); this note is the contract and the
evidence.

## 1. What is differentiated, and by whom

A forward period is a sequence of events: component steps (`Component::advance`) and coupling
instructions (the compiled operators of the plan: Source, Sample, Accumulate, Remap, Merge, Scale,
Sink, ...). The reverse sweep visits the same events in the opposite order.

- **A coupling instruction** is linear in the field values, with weights fixed at setup (remap
  matrices, scale factors, sample counts). Its adjoint is the transpose, `xbar += A^T ybar`, and
  needs no value from the forward run. The remap applies the transposed weights as a CSR matrix
  built at `initialize_adjoint()` with the construction and the summation order of the forward
  run's own matrix, so each entry is the sum `WeightsOperator::apply_transpose` makes, the
  transpose the derivative gate verifies ([](derivative-gate.md)); every other rule is a copy, a
  scalar multiple or a fan-out (section 3).
- **A component step** is reversed by the component: `AdjointComponent::advance_adjoint(ctx)`
  (`fuse/runtime/component.hpp`), reached through `Component::adjoint()`, with the `StepContext` the
  forward step had. The component brings whatever differentiation it likes (a hand-derived
  transpose, an operator-overloading tool, source transformation) and whatever forward values it
  needs (a tape or a checkpoint of its own). FUSE stores no forward value and passes no tape: an
  operator-overloading tool records process-local memory, which cannot cross MPI or the C ABI, so
  the only things that cross the interface are binary64 arrays of adjoint variables.

## 2. Interface

```cpp
class AdjointComponent {
 public:
  virtual Result<void> register_adjoint_fields(FieldRegistry& registry, const CouplingGraph& graph) = 0;
  virtual Result<void> advance_adjoint(const StepContext& ctx) = 0;
};
// Component::adjoint() returns an AdjointComponent* (default nullptr)

Result<void> Runner::initialize_adjoint(ComponentSet&, FieldRegistry& adjoint_registry);
Result<void> Runner::run_period_adjoint(ComponentSet&);
```

`register_adjoint_fields` binds, for every field of the component, an array of adjoint variables
under the field's own `FieldId`, with the layout of the field's own binding. The Runner aliases
these arrays exactly as it aliases the fields themselves (a Source or Sink arc is the component's
own array); the arcs the coupler owns get adjoint arrays of its own, zero at the start.

**Conventions for the adjoint array `Fbar` of a field F:**

| field | on entry to `advance_adjoint` | on return |
|---|---|---|
| Provides | the adjoint of F as the step left it, with the adjoint of every later read of F already added by the coupler | the adjoint of F as it was before the step (the forward step overwrote F, so the reverse step overwrites `Fbar`) |
| Requires | whatever the later steps added | unchanged except that the component ADDS the adjoint of every value it read in this step (a value read by several steps between two deliveries gets one contribution from each; the coupler zeroes `Fbar` when it reverses the delivery) |

**Sweeps.** The caller seeds the adjoint arrays with the adjoint of the final state (the Provides
arrays: the adjoint of each field after the last step; the Requires arrays: the adjoint of the value
delivered at the end of the run) and calls `run_period_adjoint()` once per forward period. The first
call after N forward periods reverses period N-1, the next N-2, and so on; periods are numbered as
the forward run numbered them, so `ctx.step`, `ctx.t_begin` and `ctx.t_end` of a reversed step equal
those of the forward step. After the last call the Provides arrays hold the adjoint of the initial
state. Running a forward period between two calls, calling more times than there are forward
periods, and calling before any forward period are refused (`E-ADJ-ORDER`); a fresh sweep starts
after the forward run continues.

## 3. The rule for each instruction

`g` is the adjoint of the instruction's output. An instruction that overwrites its output is
reversed as an assignment: `g` is taken from the output's adjoint array, which is set to zero,
because the output was dead before the write; then `g` is added to the adjoint of each input.

| instruction | forward | reverse |
|---|---|---|
| Source | the arc is the provider's own array | nothing: readers added to the provider's adjoint array |
| Sink, Epoch, Transport (one process), Constrain, Diagnose, Interpolate (no time history), Extrapolate (order 0) | `out = in` | `in_bar += g` |
| Remap | `out = W in` (identity when the regions coincide) | `in_bar += W^T g`, a gather over the transposed matrix |
| Scale | `out = f in` | `in_bar += f g` |
| Merge first | `out = in_0` | `in_0_bar += g` |
| Merge sum | `out = sum in_k` | `in_k_bar += g` for every k |
| Merge mean | `out = (sum in_k) / n` | `in_k_bar += g / n` |
| Sample | `acc = in` for the first sample of a window or for instant accumulation, else `acc = acc + in` | `in_bar += acc_bar`; `acc_bar` is set to zero after the first sample of the window (the adjoint of the accumulator before it was written), or after every sample for instant accumulation |
| Accumulate | `out = acc`; mean divides by the count, integral multiplies by the sample interval | `acc_bar += c g`, `c = 1/count`, the interval, or 1 |

The forward run counts the samples of a window at run time (`sample_count`). The reverse needs the
same numbers and takes them from a replay of the period's event order (`AdjointState::window_index`),
which is possible because the stream is static. A window that is still open at the end of the
period would make the counts depend on the previous period; that is refused (`E-ADJ-WINDOW`),
though no configuration reaching it has been found.

**Execution space.** The rules run as kernels on the execution space the forward run uses
(`RunnerConfig::execution_space`): the host space, or the device space of a build with a CUDA, HIP
or SYCL backend. The adjoint array of an arc is in the memory of the arc itself: the arcs the
coupler owns get an array in device memory when the forward run keeps the arc there, and a field
a component registers gets the memory its own adjoint array is registered in, which must be the
space the field is in (`E-ADJ-SPACE` otherwise). A field in host memory reversed on the device is
staged through a scratch array on the device: read in when the rule reads it, written back when
the rule changes it, as the forward interpreter stages the arcs it keeps on the host. Every rule
except the remap is one of three loops over an adjoint array, in `fuse/num/adjoint_kernels.hpp`
(add a multiple of one array to another, take an array and set it to zero, fill); each is a named
functor, whose element results do not depend on how the range is split, and the multiply and add
of the first are not fused (`unfused_multiply_add`), so the host and the device round the same
way. The transpose of a remap is `remap_apply` over the transposed CSR matrix, a gather per
output row and therefore without atomic additions, and the order of the sum in each row is the
order of the triplets, as on the host. Each instruction ends in a fence on the device, and a
fence precedes every call to `advance_adjoint` and ends the sweep, so a component, whose own
kernels must have finished when it returns, sees complete adjoint arrays and the next instruction
sees its result.

**Order.** The sweep reproduces the merged walk of `Runner::Impl::step_period()` -- ticks before the
instructions stamped later, `AfterTicks` instructions after the ticks at their own time -- by
replaying its condition on the static lists of ticks and instructions, then reads the list
backwards. The test `TheReverseSweepVisitsEveryStepInTheReverseOfTheForwardOrder` compares the
calls of a multirate loop with the forward calls; a change to the walk in `step_period()` must be
repeated in `init_adjoint()`.

## 4. Refusals

Everything below is refused by name at `initialize_adjoint()` (or at the first sweep for the order
codes), so a graph that asks for more is stopped before anything is differentiated.

| code | when |
|---|---|
| `E-ADJ-INIT` | `initialize_adjoint()` before `initialize()`, or `run_period_adjoint()` before `initialize_adjoint()` |
| `E-ADJ-ORDER` | a sweep call with no forward period to reverse, after every period was reversed, or after a forward period ran in the middle of the sweep |
| `E-ADJ-COMPONENT` | a component that advances has no `AdjointComponent` (`Component::adjoint()` is null), or is driven from outside the Runner |
| `E-ADJ-UNBOUND` | a component-owned arc has no adjoint array in the registry |
| `E-ADJ-LAYOUT` | an adjoint array is not a flat, component-fastest array with the field's count and components |
| `E-ADJ-ALIAS` | an adjoint array is the field's own array |
| `E-ADJ-REGION` | the plan has an iteration region (the adjoint of a fixed-point iteration is implicit differentiation, or the unrolled sweeps: not implemented) |
| `E-ADJ-ACCUM` | a minimum or maximum accumulation (the adjoint selects a sample and needs the forward values) |
| `E-ADJ-SPACETIME` | a space-time Sample or Interpolate (a time history) |
| `E-ADJ-TRANSPORT` | a Transport between processes (the adjoint of an MPI exchange) |
| `E-ADJ-NODAL-DIST` | a distributed nodal Remap |
| `E-ADJ-FIELD-TYPE` | a field of an Arithmetic (dd, qd) or Storage (binary32, bfloat16, binary16, posit) value type |
| `E-ADJ-SPACE` | an adjoint array in a different memory space than its field (a field in device memory is refused by `initialize()` itself under host execution, `E-RT-SPACE`) |
| `E-ADJ-OP` | an instruction with no adjoint (Relax, Residual, ExternalDelivery, Accelerate, WindowAdvance, ...) or an Extrapolate above order 0 (the forward run applies order 0 whatever order is asked; an adjoint written for that would go wrong silently when it does not) |
| `E-ADJ-WINDOW` | an accumulation window open at the end of a period |

## 5. Verification

**The adjoint identity for the composed coupling**
(`test/runtime/adjoint_runner_test.cpp`, ctest `runtime.adjoint_runner`). The composed map `G` takes
the inputs of a run (the initial state of each component that provides a field and the value each
producer is driven with at each step) to the state after N periods (each component's provided and
delivered fields). The components are linear (`out <- a out + c in + drive`), so `G` is the same for
every run, `G x` is the Runner's forward run from a fresh network, `G^T w` is the reverse sweep
seeded with `w` and read from the adjoint arrays, and the derivative gate's own functions check
`|<G x, w> - <x, G^T w>| <= c * eps * ||G||_est * ||x|| * ||w||` with `c = kAdjointGateConstant = 50`
and `||G||_est` by `estimate_operator_norm` (power iteration through the same two maps). Nothing in
the test knows the remap weights.

The sixteen cases (`identity_cases()`), with the periods each runs:

| case | what it exercises | periods |
|---|---|---|
| `instant_remap_8_to_5` | an instantaneous channel through a real 8 to 5 cell remap | 1 |
| `instant_remap_5_to_8` | the remap the other way | 1 |
| `three_components_per_cell` | three values per cell | 1 |
| `mean_of_four_steps` | a producer four times faster than its consumer, mean accumulation | 2 |
| `sum_of_four_steps` | the same, sum | 2 |
| `integral_of_four_steps` | the same, integral (units of value times time) | 2 |
| `instant_of_four_steps` | the same, instantaneous: the value the producer holds when the chain runs | 2 |
| `conservative_remap` | a conservative remap with the conservation check after it (Constrain, Diagnose) | 2 |
| `loop_one_region_lag_only` | a loop on one region, so the lagged channel is a bare Interpolate | 4 |
| `extrapolate_order_0` | an Extrapolate of order 0 on the channel | 1 |
| `unit_conversion` | a Scale by 1000 ahead of the remap | 1 |
| `merge_sum`, `merge_mean`, `merge_first` | two producers into one field | 1 |
| `loop_lagged_5_periods` | state in both components, a loop closed by a lagged channel | 5 |
| `loop_mean_multirate_3_periods` | the loop with a mean over four steps of the faster component | 3 |

Between them the cases reach every instruction kind the adjoint has a rule for (the test
`TheCasesReachTheInstructionKindsTheDocumentationClaims` fails if one stops being reached): Source,
Sink, Remap, Sample, Accumulate, Merge, Scale, Extrapolate, Interpolate, Transport, Epoch, Constrain
and Diagnose (BuildRemapHalo, which carries no value, is in every plan).

The test prints each case's worst ratio `defect / (eps ||G||_est ||x|| ||w||)` over three draws;
the largest over the cases, measured on the Mac (AppleClang 21, Debug, tier 0), was **0.351**
against `c = 50`, the same order as the 0.235 the gate measured for its own operators.
`AWrongComponentAdjointIsDetected` is the control: the loop with a component whose adjoint leaves
out the factor of its state fails the same gate by more than 100 times `c`.

**On the device.** The test binary is registered a second time as `runtime.adjoint_runner.device`
(`FUSE_EXECUTION_SPACE=device`) in a build with a device backend, so every test above also runs
with the sweep as device kernels: the fields the test components register are in host memory and
are staged. On higgs (CUDA 12.5, sm_61) the worst ratio is the same 0.351. Four tests exist only
in such a build (`RuntimeAdjointDevice.*`): the identity for the same sixteen cases with the
fields and the adjoint arrays of the components in device memory, with the components' own steps
and adjoints as kernels (worst ratio 0.480); the adjoint of each case under the host and under the
device execution space for the same seed, which agrees to 2.2e-16 of the largest entry at worst
(the test allows 4 eps); and the two refusals of an adjoint array in the other memory space than
its field.

**The rule of each instruction, by exact value.** A mean of four samples gives each sample exactly
1/4 of the adjoint, a mean of eight 1/8, an integral over eight samples at `dt = 1/4` gives 1/4
(not 1/8: this separates the two), a sum gives each the whole, a unit conversion multiplies by
1000, a merge sum copies to both producers, a merge mean halves, a merge first feeds only the
first (the second gets exactly zero), and a remap's adjoint equals the transpose of the matrix
recovered from the forward run (a unit vector in, a column out).

**The order of the sweep** against the forward order, and **every refusal** above except
`E-ADJ-TRANSPORT`, `E-ADJ-NODAL-DIST` and `E-ADJ-WINDOW`, which need an MPI plan or a configuration
not known to exist.

**Mutation coverage.** `build-evidence/mutations/runtime-adjoint.toml`, area `runtime_adjoint`:
51 rows, against `src/runtime/runner_adjoint.cpp` and `include/fuse/num/adjoint_kernels.hpp`: the
transposed matrix of a remap and the accumulation and the take of an adjoint (the three kernels),
the rule of each reduction (sample, accumulate, merge, scale), the window bookkeeping, the order of
the sweep and the step contexts, each copy-type instruction, each refusal, and the device path
(the memory an arc's adjoint array is in, the dispatch to the device space, the staging of a host
operand in and out, the mirror of the transposed matrix). Six rows exist only for the device path
and are marked `requires = "device"`: the tests that kill them are compiled only against a device
backend, and the Mac tier-0 build reports them NOT-ASSESSED; the other 45 are killed on the Mac.

Three candidate rows were dropped as equivalent mutants, not as tested. Replacing the accumulation of
a remap's input adjoint by an assignment changes nothing, because each arc is read by one
instruction and the instruction that wrote it was reversed first and took its adjoint, so the array
is zero when the remap is reversed. The instantaneous branch of the Sample rule is reached only by
a hand-built plan: lowering emits no Sample for an instantaneous channel. That branch mirrors the
forward kernel and has no test. Removing the fence that ends each instruction on the device
changes nothing the sweep can observe: Kokkos runs the kernels and the copies of one execution
space in order, and the fence before each `advance_adjoint` call, which is kept, is the one a
component depends on.

## 6. Not done

- **MPI.** The adjoint of a Transport between processes is the same exchange with send and receive
  exchanged (the CommGraph reversed); a distributed nodal Remap needs `apply_distributed_HT` in
  the roles of `H` and `H^T`; `Epoch` carries a collective agreement that the reverse sweep skips
  (it is a copy here, single process).
- **Iteration regions and time windows.** Reverse mode through a fixed-point iteration is implicit
  differentiation (a linear solve with the transposed interface Jacobian, for which the
  `ImplicitComponent` linearization already offers `apply_jacobian_transpose`) or the unrolled
  sweeps (which need the number the forward run reached and every sweep's forward values).
- **Time histories** (space-time Sample and Interpolate), **minimum and maximum accumulation**,
  and **Arithmetic and Storage field types**: refused above. The last two are mechanical (the
  adjoint of a widening is a rounding, of an Arithmetic remap the transpose in the type).
- **The C ABI, Fortran and Python.** A component through the C ABI has no `advance_adjoint` entry
  and no adjoint array registration yet; `Component::adjoint()` is a C++ interface only.
- **Gradients with respect to the mesh** (U8).
