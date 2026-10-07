# The Conformance track

S6 phase 3. The question this track answers: when an external code couples
through the participant ABI instead of `fuse::Runner` driving it in-process,
does it see the same answer? "Conformance" here means specifically:

- **Bit-for-bit** against an in-process `fuse::Runner` reference driving the
  identical `CouplingGraph`, over the example's full span (every period, not
  a prefix).
- **Identical sweep counts** at every window — the convergence decision
  agrees, not just the converged state.
- **Both waveform iterations** (Jacobi and Gauss-Seidel) on the real
  example parameters, not a reduced fixture.
- **`--check` thresholds unchanged** — `brusselator_check`'s own
  `kMaxWindowBalanceThreshold`/`kGlobalBalanceThreshold`/
  `kMaxWindowExchangeThreshold` are not loosened to make a path pass.

A defect that breaks any one of these is a conformance defect, whether or
not it also breaks something more obviously visible (a crash, a hang). The
four properties this track was opened to demonstrate, and the Runner
defects it found doing so, are in `KNOWN_ISSUES.md` A4; this page is the
record a future external code — a different participant, a different
transport — should be held to: what broke conformance once, and what proves
it is fixed.

## Unit 1 (Time Windows track): four `fuse::Runner` defects

Found and fixed building `test/verification/brusselator_runner_test.cpp`
(see that file's own header comment for the narrative). All four are in the
window-scoped region path; none are visible on a graph with no window
regions, which is why the Time Windows track found them and nothing
earlier did.

1. **Window instance detection compared a period-relative step against no
   period counter.** `Runner::Impl::advance_window()`'s "is this the first
   sweep of a new window instance" test, and its T0/T1 computation, read
   `Instruction::dst_step` directly. `dst_step` resets to 0 on every
   `run_period()` call (the compiled `TimedDag` spans one period, replayed
   verbatim), so a window-scoped region with one instance per period saw
   `dst_step == 0` on every call, not just the first, and never advanced
   past window 0 after the first period.
   Fix: `window_absolute_instance(cfg, ins.dst_step)` — commit `7939643`
   ("Track a time window's absolute instance across repeated periods",
   2026-10-02). Mutation row: `window-runner-new-instance-uses-period-
   relative-dst-step` (`build-evidence/mutations/window-runner.toml`).

2. **`ClockSet` fed a window member's own fine interval into the period
   LCM instead of its region's window length.** On the Brusselator's own
   `dt_u=0.002`/`dt_v=0.003`/`window=0.02`, this compiled a period of
   `lcm(0.002, 0.003) = 0.006`, which `0.02` does not divide evenly, and
   `schedule()` failed `E-SCHED-PERIOD` on a graph that should succeed with
   period `0.02`.
   Fix: commit `c083ea6` ("Give a time-window member the window length as
   its clock, not its substep", 2026-10-02). Mutation row:
   `clockset-window-member-reverts-to-own-interval`
   (`build-evidence/mutations/ir-schedule.toml`).

3. **Window-start "Sample 0" re-stamp read the live export, not the value
   captured at the window's own first sweep.** `advance_window()` re-read
   `cfg.provided[e]` on every sweep instead of a value cached once;
   `Component::restore_state()` is only contractually required to restore
   the component's own hidden state, not to resynchronize a `Provides`
   field that is a function of it, so from the second sweep on this
   mislabeled the previous (rejected) sweep's own last `advance()` output
   as the window-start value. Measured effect: `max_window_balance_error`
   `1.686e-03` before the fix, `1.703e-13` after.
   Fix: `Runner::Impl::window_start_value`
   (`src/runtime/runner_impl.hpp`/`src/runtime/iteration.cpp`) — commit
   `60e2d97` ("Cache a window member's start-of-window export instead of
   re-reading it each sweep", 2026-10-02). Mutation row:
   `window-runner-sample0-reads-live-provided-every-sweep`
   (`build-evidence/mutations/window-runner.toml`).

4. **Per-block residual norm summed the window-start sample into the
   convergence measure.** `run_region()`'s named-measure residual summed
   the window-scoped value arc's full capacity-wide buffer — window-start
   sample included — instead of exactly the consumer's own substep count.
   The window-start sample never changes within one window instance, so
   this only inflates the reference norm, never the residual, loosening a
   `MeasureKind::Relative` tolerance enough to converge one sweep earlier
   than the driver. Measured divergence: window 20, sweep k=1, hexfloat
   `0x1.6be7e489b2a8cp+5` vs `0x1.5b510b3c3ad89p+5`; confirmed by
   `PerWindowStateMatchesDriverWithStoppingRuleRemoved` that all 150
   windows match bit-for-bit once the stopping rule is removed — only the
   convergence *decision* disagreed, not the underlying state.
   Fix: `RegionBlock::window_export_capacity`/
   `window_export_valid_substeps` (`src/runtime/runner_impl.hpp`) and the
   stripped-gather norm in `run_region()` (`src/runtime/iteration.cpp`) —
   commit `640e95e` ("Exclude the window-start sample from a window
   block's convergence norms", 2026-10-02). Mutation row:
   `window-runner-named-measure-includes-window-start-sample`
   (`build-evidence/mutations/window-runner.toml`).

## Unit 2b (coupling-window multi-process path, `KNOWN_ISSUES.md` D19):
four defects

Found making the window-scoped path run conformant on more than one
process (a coupler placed as its own rank set). Each is tagged in its own
mutation row's `area` field as `Unit 2b step (<letter>), D19`.

5. **Step (a) — hosting.** `advance_window()` advanced a window member on
   every process in its region, not only the processes that host it;
   a coupler-placed rank with no window member of its own still ran the
   advance.
   Fix: commit `0eb576a` ("Advance a time-window member only on its own
   processor set", 2026-10-03). Mutation row: `window-runner-hosting-
   ignored` (`build-evidence/mutations/window-runner.toml`).

6. **Step (b) — priming the wrong address space.** `resolve_regions()`'s
   cold-start prime read a window member's own provider export directly,
   which is only valid when priming and priming-from live in the same
   address space; across a process boundary this primed from memory the
   providing rank never populated.
   Fix: commit `2df7052` ("Prime a window member's import through its
   field transfer", 2026-10-03). Mutation row: `window-runner-priming-
   through-transfer-disabled`
   (`build-evidence/mutations/window-runner.toml`).

7. **Step (c) — convergence not agreed.** `run_region()`'s window
   convergence decision was taken locally per process instead of agreed
   across the window's participants; two ranks could reach different
   sweep counts for the same window.
   Fix: commit `5dddd68` ("Agree a time window's convergence across its
   participants", 2026-10-03). Mutation row: `window-runner-convergence-
   agreement-disabled` (`build-evidence/mutations/window-runner.toml`).

8. **Step (d) — placeholder ranks never ran.** `fuse_run()`'s
   `E-ABI-RUN-EXTERNAL` refusal was unconditional on the graph declaring
   any externally driven component anywhere, not on this process hosting
   one (`Runner::hosts_component()`). A process hosting only a
   `ComponentKind::Coupler` placement — no `advance()` to drive at all —
   was refused `fuse_run()` and so called neither it nor
   `fuse_begin_step()`/`fuse_end_step()` between `fuse_initialize()` and
   `fuse_destroy()`: `coupler_main.c`'s and `u_main.c`'s own placeholder
   ranks.
   Fix: commit `f7f4d03` ("Let run execute a period on a process that
   hosts no externally driven component", 2026-10-03). Mutation row:
   `run-external-refuses-unconditionally-not-by-hosting`
   (`build-evidence/mutations/abi.toml`).

Covered by `parallel.window_runner_mpi.{2,3}` (`test/parallel/
window_runner_mpi_test.cpp`) — built tier-1, not run in this sandbox (MPI
cannot start here); see `KNOWN_ISSUES.md` D19 for the exact host-run
status.

## Unit 3a (LowFive track, `KNOWN_ISSUES.md` D14): two defects

Found bringing an `ExternalDelivery` channel's own `fuse_step_report`
protocol to agreement with the in-process Runner. Both fixed in the same
commit, `f768fb3` ("Hand control to the application at an external
delivery", 2026-10-06), which also grew `fuse_step_report` 72→88
bytes (`step_opened`/`period`, see `docs/user/c-abi.md`'s growth-history
table).

9. **`prime_next_request()` treated a legitimate rollover as the
   unreachable error.** A `resume_period()` call that finds nothing left
   in the current period is an ordinary, legitimate period-end reached by
   resuming past a cached LowFive delivery proxy — not the genuine
   invariant violation `E-ABI-STEP-ORDER` is reserved for (a *second*
   consecutive empty period). Before the fix, the first such call was
   misreported as that error.
   Mutation row: `abi-prime-next-request-rollover-removed`
   (`build-evidence/mutations/abi.toml`).

10. **`window_import_delivery_arc`'s chain-root walk was missing.** An
    `ExternalDelivery` channel inside a window-scoped region needs its
    import resolved back to the region's own chain root, not just its
    immediate predecessor, to land on the correct
    `fuse_step_report::step_opened`/`period` pair at a matched rollover.
    Added in the same commit as defect 9 (`src/runtime/runner_impl.hpp`/
    `src/runtime/iteration.cpp`); no mutation row in the current catalog
    isolates this half of the commit separately from defect 9's — noted
    here rather than invented, since inventing one would misrepresent
    what the mutation catalog actually proves.

Covered by `BrusselatorLowFiveHarness.ExternalDeliveryMatchesInProcessRunner`
(bit-for-bit against the in-process Runner, identical sweep counts, both
waveform iterations) and, at the ABI level, by
`AbiParticipant.EndTimeBoundedRunRefusesPastItsOwnDeclaredEnd` and
`AbiParticipant.MismatchedRolloverNamesTheRealPeriodStartComponent` in
`test/abi/participant_test.cpp`, plus the golden-IR fixture
`GoldenIr.PairLowFiveBound`.

## What this page does not cover

Unit 3b/3c's channel-compilation and multi-channel defects
(`delivery_pending`/`delivery_field`, the E-LAG gap in `channel_window()`)
are this round's own item 2 work, already in `KNOWN_ISSUES.md` D14's
"Item 2 (this round)" account with their own commits
(`aedc3b6`/`dfcd9d1`/`43fe58d`/`513d263`) — not duplicated here, since
they are not Runner or window-path defects but channel-lowering and
ABI-report ones.
