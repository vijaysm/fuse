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
Result<std::optional<ExternalAdvanceRequest>> Runner::resume_period_adjoint(ComponentSet&);
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

**Components driven from outside.** The participant ABI (`fuse_begin_step()`/`fuse_end_step()`) wraps
each component its caller drives in an `ExternalStepComponent`, whose `advance()` is a no-op and whose
forward walk suspends at its step (`Runner::resume_period()`). The reverse sweep does the same, so that
the caller reverses its own step in its own adjoint arrays: `resume_period_adjoint()` reverses the open
period (the newest one not yet reversed, when none is open) until the step of the next such component
and returns it, as an `ExternalAdvanceRequest` with the `StepContext` of the forward step; calling it
again carries on below that step, and `std::nullopt` means the period is reversed completely. The sweep
is a cursor into the period's list of events (`AdjointState::cursor`), so a suspension is a return and
nothing is kept on a stack. Everything the sweep reverses before returning is what the forward run did
after that step, so on return the adjoint arrays of the component's Provides fields hold what
`advance_adjoint()` would be given. `run_period_adjoint()` is the same loop without suspension and
refuses a run with such a component (`E-ADJ-EXTERNAL`). The adjoint arrays of such a component are the
caller's, registered with `fuse_register_adjoint_field()` and bound under the field's `FieldId` by
`ExternalStepComponent::register_adjoint_fields()`; a component with none at all is refused
(`E-ADJ-COMPONENT`), one with some missing is refused at the field (`E-ADJ-UNBOUND`). A component
driven from outside inside an iteration region is refused (`E-ADJ-REGION`): the adjoint iteration
repeats the steps of the region, once per pass, and cannot ask the caller for each. A failure on a rank
is carried to the end of the period as before; the failing call is the `end_adjoint_step()` that
reaches it. The entry points built on this are in [](../user/c-abi.md#reverse-mode) (participant
direction) and [](../user/providers.md#adjoint-of-a-plugin) (provider direction, where
`ProviderAdapter` is the plugin's `AdjointComponent`).

## 3. The rule for each instruction

`g` is the adjoint of the instruction's output. An instruction that overwrites its output is
reversed as an assignment: `g` is taken from the output's adjoint array, which is set to zero,
because the output was dead before the write; then `g` is added to the adjoint of each input.

| instruction | forward | reverse |
|---|---|---|
| Source | the arc is the provider's own array | nothing: readers added to the provider's adjoint array |
| Sink, Transport (one process), Constrain, Diagnose, Interpolate (no time history), Extrapolate (order 0) | `out = in` | `in_bar += g` |
| Epoch | the ranks of the exchange agree whether all succeeded, then `out = in` | the same agreement (`reconcile_epoch`), then `in_bar += g` |
| Transport between processes | `Transport::move` copies the entries of `in` it moves into entries of `out`, same-rank matches directly, the rest by an exchange; the other entries of `out` are left alone | `g_w` is the adjoint of the entries written, taken from `out`'s adjoint array and set to zero there; `in_bar += P^T g_w`, the exchange run backwards (`Transport::move_adjoint`); the entries not written keep their adjoint |
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

**Iteration regions.** A region is reversed as the fixed point its forward sweeps converged to, not
sweep by sweep. At the fixed point `y* = F(y*, x)`, with `x` everything the region reads from
outside (the checkpointed states, the drives, the values delivered before it), the derivative of
the converged values is `(I - F_y)^-1 F_x` whatever way the iteration reached `y*` (plain,
relaxed, accelerated), so the adjoint is `lambda = ybar + F_y^T lambda`, `xbar += F_x^T lambda`.
One pass of the reverse of the region's interval of the period (the component steps the region
re-advances, the instructions between them and the body, which ends at the value `G(y)` that the
delivery after the region copies into the components' inputs), seeded with `lambda` on the
values of the body, leaves `F_y^T lambda` on the delivered values and `F_x^T lambda` in the
adjoint arrays of everything else. The sweep therefore repeats that pass to a fixed point
(Christianson's reverse accumulation, 1994): before each pass every adjoint array the interval
can change is restored from a copy taken before the first, so the contribution to `xbar` is that
of the last pass only; the new iterate is the seed plus what the pass added to the delivered
values; and the iteration is held to the convergence criterion of the region (the same
measures, applied to the change of the iterate). The initial iterate does not influence
the fixed point, so what the last pass added to the delivered values is not passed on. The
`Residual` of the body, which only feeds the convergence test, has no adjoint to apply.

A component the region steps is therefore reversed once per pass: `advance_adjoint(ctx)` is called
J times for the same `ctx`, with the adjoint arrays as they were found except for the delivered
values, and must give the same result each time (it may not consume what it recorded). The
forward run advanced the step several times for the same `ctx.step` (after `restore_state()`,
with `ctx.iteration` counting the sweeps); the component's adjoint is the adjoint of the last
advance, and `ctx.iteration` is 0 in the reverse calls. The plain iteration converges when the
forward iteration does without acceleration; a region whose forward iteration needed an accelerator
to converge fails with `E-ADJ-REGION-NOCONVERGE` rather than return a derivative. The adjoint is
that of the iteration stopped at the criterion: the last pass is seeded with the iterate before
the last, so the error is of the order of the criterion times the contraction factor over one
minus it.

**Between ranks.** The reverse of a Transport between processes is the same exchange run
backwards, by the backend that ran it forward. `Transport::move_adjoint(dst_bar, src_bar, n)`
(`fuse/transport/transport.hpp`) overwrites `src_bar` with `P^T dst_bar`, where `dst = P src` is the
0/1 matrix `move()` applies to the entries it writes; `has_adjoint()` says whether a backend has
it, and a backend that has not is refused at `initialize_adjoint()` (`E-ADJ-TRANSPORT`).
`CopyTransport` copies the other way. `MpiTransport` packs `dst_bar` in the order of its receive
buffer, runs the `MPI_Alltoallv` of `move()` with the send and receive counts and displacements
exchanged, and adds what arrives to the entries of `src_bar` that `move()` read: a source entry
sent to several ranks (a halo) receives the sum of their adjoints, in ascending peer rank, and the
same-rank matches of a partially overlapping plan are added directly. The Runner reverses a
Transport on host buffers, as the forward run does, so a destination arc in device memory is copied
to the host and back, and the adjoint handed to the source arc is copied in from the host; the
device-direct exchange of the forward run (`move_packed`) is not used. The exchange and the
`Epoch` that follows it are collective, so the sweep is carried like a period of the forward run
(`step_period`): a rank on which a component's adjoint step or an instruction fails records the
failure and goes on to execute only the collective instructions (`is_collective`), so that the
ranks it exchanges with are not left waiting for it, skips the local work, and returns its failure
at the end of the period; the reverse of an `Epoch` takes part in the agreement, so that a rank
whose partner failed earlier in the sweep fails too (`E-RT-EPOCH-PEER`) rather than pass on an
adjoint derived from an exchange that did not complete. The reverse order puts a component's step
after the exchanges that follow it in the forward order, so a failure at a step has no `Epoch` behind
it in the sweep (measured on higgs: with the failing step reversed last, the rank that does not hold
the component returned success and its partners were left waiting in the next period); the end of
the period is therefore an agreement point of its own (`agree_on_adjoint_period`), the reduction of
`reconcile_epoch` over the ranks that take part in any `Epoch` of the plan, and a rank that did not
fail records `E-RT-EPOCH-PEER` when another did. Following the rule of KNOWN_ISSUES.md D97 (the
original error and the rank it came from are known on every rank), the lowest-numbered rank that
failed sends its own code and message to the others, and the failure a healthy rank records is the
agreed refusal of `fuse::make_agreed_failure()` with `origin_rank` and `origin_code` set; a rank that
failed itself keeps its own failure. (The `Epoch` agreement of the forward run, which the reverse of
an `Epoch` also takes part in, does not carry the origin's message; it was not changed.) A period
that failed did not happen: the next `run_period_adjoint()` reverses it again, on every rank.

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
| `E-ADJ-COMPONENT` | a component that advances has no `AdjointComponent` (`Component::adjoint()` is null): a native component without one, a plugin that does not offer the adjoint, or a component driven from outside the Runner whose caller registered no adjoint array at all |
| `E-ADJ-EXTERNAL` | `run_period_adjoint()` on a run with a component driven from outside the Runner (use `resume_period_adjoint()`) |
| `E-ADJ-UNBOUND` | a component-owned arc has no adjoint array in the registry |
| `E-ADJ-LAYOUT` | an adjoint array is not a flat, component-fastest array with the field's count and components |
| `E-ADJ-ALIAS` | an adjoint array is the field's own array |
| `E-ADJ-REGION` | an iteration region that is solved by Newton's method, is scoped to a time window, spans several ranks, overlaps another region in the order of a period, or whose interval holds the step of a component the region does not step |
| `E-ADJ-REGION-NOCONVERGE` | the adjoint iteration of a region did not meet the region's convergence criterion within its `max_iterations` passes (the forward iteration was not a contraction, or converged only through its accelerator) |
| `E-ADJ-REGION-DIVERGED` | the adjoint iteration of a region left the finite numbers |
| `E-ADJ-ACCUM` | a minimum or maximum accumulation (the adjoint selects a sample and needs the forward values) |
| `E-ADJ-SPACETIME` | a space-time Sample or Interpolate (a time history) |
| `E-ADJ-TRANSPORT` | a Transport between processes whose backend has no `move_adjoint` (LowFive), or a Transport that resolved to no plan |
| `E-ADJ-NODAL-DIST` | a distributed nodal Remap (its adjoint would be `apply_distributed_HT`, and `D H` for the consistent transpose; not verifiable here, see section 6) |
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

**Mutation coverage.** Three areas in `build-evidence/mutations/`: `runtime_adjoint` (83 rows) and
`runtime_adjoint_agreement` (5 rows), both in `runtime-adjoint.toml`, and `abi_adjoint` (31 rows,
`abi-adjoint.toml`). The first runs against `src/runtime/runner_adjoint.cpp`,
`include/fuse/num/adjoint_kernels.hpp` and the two transports: the transposed matrix of a remap and
the accumulation and the take of an adjoint (the three kernels), the rule of each reduction (sample,
accumulate, merge, scale), the window bookkeeping, the order of the sweep and the step contexts, each
copy-type instruction, each refusal, the device path (the memory an arc's adjoint array is in, the
dispatch to the device space, the staging of a host operand in and out, the mirror of the transposed
matrix), the exchange between ranks (the index maps of the send and the receive, the zeroing of what
was passed back, adding against overwriting, the local matches, the size and null checks) and the
region sweep (the number of passes, the convergence test, the seed, the restore of the arrays, the
feedback of the iterate). The second holds the agreement of the end of a reverse period (the
reduction, the recording of the peer's failure, the origin's rank and message, the lowest rank as
origin). `abi_adjoint` holds the provider adapter (10 rows), the table-size check of the registry (2), the participant entries (16), the
external step component (1) and the Runner's suspension at an external step (2). Six rows of the
first area are marked `requires = "device"` and 22 `requires = "mpi"`: the tests that kill them are
compiled only against a device backend or run only under a launcher, and the Mac tier-0 build
reports them NOT-ASSESSED (the tool runs the MPI rows only with `FUSE_MUTATE_MPI=1`).

Results. On the Mac (tier 0, before the rebase onto the main that carries the field transaction):
the 55 host rows of `runtime_adjoint` and all 31 rows of `abi_adjoint` are killed. On higgs (CUDA, OpenMP and Open MPI; the rebased stack, with
`FUSE_MUTATE_MPI=1`): `runtime_adjoint_agreement` 5 killed of 5; `runtime_adjoint` 80 killed, 1
declared equivalent and 2 not killed of 83; `abi_adjoint` 29 killed and 2 without a baseline of 31
(`abi.conformance`, the test of the two rows on the table size, fails on higgs at the base, with
`MPI_ERR_COMM` from the handles it fakes; both rows are killed on the Mac). The equivalent row,
`runtime-adjoint-device-arc-in-host-memory`: an arc the forward run keeps in device memory whose
adjoint array is in host memory is staged through the lanes of `reverse_on`, which gives the same
result as the adjoint array in device memory. The two not killed, `epoch-reverse-skips-the-agreement`
and `sweep-skips-the-exchanges-after-a-failure`, both concern a failure that is followed by a
collective instruction in the sweep: the failing step of the tests' plan is reversed after the last
exchange of the period (traced on higgs), so the agreement at the end of the period carries the
failure and nothing collective follows it. They need a test with a failing instruction (an exchange
that errors) ahead of an exchange, or a plan that steps a component after an exchange; none is
built here, and the rows stay in the table, with that note, as the open gap.

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

- **Distributed nodal Remap.** Its adjoint is `apply_distributed_HT` where the forward run applied
  `H`, and `apply_distributed_H` followed by the diagonal `D` where it applied `H^T D` (the
  consistent transpose), each with the exchange inside it run the other way. It is not written
  because it cannot be verified here: the forward test of the same path,
  `parallel.nodal_remap_runner.{1..5}` (and `.1` to `.3` on the device), fails on higgs at 8820aed2
  (a main commit before any change for number types) and at the head of the number-types branch,
  with `E-RT-CONSERVATION: relative conservation error 0.909091 (integral 11.000000 -> 1.000000)`.
  An adjoint of a forward map that does not pass its own test would be checked against nothing.
  KNOWN_ISSUES.md D102 has the record.
- **Transports other than copy and MPI.** A LowFive transport has no `move_adjoint` and is refused;
  the device-direct exchange (`move_packed`, GPU-aware MPI) is not used in the reverse sweep, which
  exchanges through host buffers. The same-rank matches of a plan that is not peer-wise
  (`TransportPlan::local_src`) are reversed by code that mirrors the forward loop and that no test is
  known to reach: the Runner plans a partially overlapping placement peer-wise, and the tests'
  placements with same-rank matches are partially overlapping.
- **Regions solved by Newton's method, regions scoped to a time window, regions across ranks.**
  A Newton region is differentiated by a linear solve with the transposed interface Jacobian
  (`ImplicitComponent::apply_jacobian_transpose` already exists); a window region has several
  passes per period and a waveform iteration; an adjoint iteration across ranks would need every
  rank to agree on the criterion after each pass, as the forward run does with
  `reduce_region_residuals`. Each is refused by name (`E-ADJ-REGION`). A forward iteration that
  converges only through its accelerator has no plain adjoint iteration
  (`E-ADJ-REGION-NOCONVERGE`); reversing the accelerator's own scheme is not implemented.
- **Time histories** (space-time Sample and Interpolate), **minimum and maximum accumulation**,
  and **Arithmetic and Storage field types**: refused above. The last two are mechanical (the
  adjoint of a widening is a rounding, of an Arithmetic remap the transpose in the type).
- **Python, and what the C and Fortran entry points leave out.** The participant ABI has the six
  reverse mode entries and a provider plugin has `bind_adjoint`/`advance_adjoint`; Python has none.
  Not done through them: a provider plugin's field in device memory (`E-ADJ-SPACE`) and a plugin or
  participant field of a storage or arithmetic element type (`E-ADJ-FIELD-TYPE`,
  `E-ABI-ADJOINT-REGISTER-ELEMENT`); a participant's step inside an iteration region (`E-ADJ-REGION`);
  and no test drives the participant entries on more than one process, so the collective agreement of
  `initialize_adjoint`, `run_adjoint`, `begin_adjoint_step` and `end_adjoint_step` (the same
  `reconcile_collective_status` as the forward entries) is wired and not exercised under MPI. There
  is no Fortran plugin example of an adjoint (the generated `ffuse_provider` module has the setters,
  `ffuse_provider_set_bind_adjoint` and `ffuse_provider_set_advance_adjoint`).
- **Gradients with respect to the mesh** (U8). Forward mode exists for the moving-least-squares
  weights ([](mls-shape-gradient.md)). Reverse mode does not: this schedule treats every remap weight
  as a constant of the problem, so a reverse sweep returns gradients with respect to fields and never
  with respect to the positions of points or the vertices of a mesh, and the weights of the supermesh
  and TempestRemap builders have no gradient at all.
