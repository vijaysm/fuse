# Coupling windows: internals

User-facing semantics are in [](../user/coupling-windows.md). This page is
for whoever next touches `fuse/schedule/window.hpp`,
`fuse/runtime/window_coupler.hpp`, or the tests around them.

## Why this is two files, not one field

The natural spelling of the opt-in is a member on `IterationRegionDecl`
(`fuse/model/coupling_graph.hpp`). `IterationRegionDecl` is left alone, and `CouplingWindowScope`
(`fuse/schedule/window.hpp`) carries the opt-in as a sibling parameter
instead — passed alongside the graph, not inside it — and
`validate_coupling_window()` is the legality gate. `schedule()` itself calls
it, and acts on the scope it names, for every window-scoped region in the
graph (`src/schedule/scheduler.cpp`'s `schedule()`, which runs the gate
before `expand()`). `mark_regions()`'s contiguous-run rule (same file) is
NOT unchanged: it carries its own window multi-pass bound -- a window-scoped
region's own clock may tick several times per period, so the rule stops a
run the moment every member has appeared once rather than greedily
consuming every later pass into it. A two-clock region without the opt-in
still refuses exactly as `test/schedule/multirate_region_test.cpp` pins;
that part never changed.

`WindowCoupler` (`fuse/runtime/window_coupler.hpp`) is the runtime driver a
model author can use directly today, against real `Component`/`Waveform`/
`Accelerator` — it does not go through `CouplingGraph`/`FieldRegistry` at
all. `WindowMember::provided`/`required` are raw pointers the caller binds.

## The one-checkpoint constraint, and why the fixup replays the whole window

`Component::save_state()`/`restore_state()` retain **one** checkpoint
(`component.hpp`: "Only one checkpoint need be retained; FUSE saves at most
once per iteration region entry"). A `Fixed` member's straddling step needs
its own rollback point a whole window later than the window-start
checkpoint the sweep loop is using for its own rollbacks. A second,
independent checkpoint at the straddle's own start does not fit that
contract — nothing revokes the window-start checkpoint mid-loop without
breaking the very rollback it exists for.

The resolution: `resolve_pending_straddles()` never tries to checkpoint the
straddle's own start. Instead, at the start of the window AFTER a straddle,
it `restore_state()`s back to the checkpoint the previous window's sweep
loop already took (still valid — nothing overwrote it since that loop's
last `restore_state()` call) and **replays every substep of the previous
window from there**, now reading that window's converged data for the part
inside `[window_start, window_end)` and an extrapolation for whatever ran
past it. This is more work per straddle than replaying only the overshoot,
and it is the only shape that stays inside the documented contract.

## Why a component receives a mean, not two endpoint values

An earlier draft of this algorithm (kept only in the design notes, never
committed) fed `advance()` both endpoint values of the partner's data over a
substep, so the component itself could do exact variation-of-parameters
against a piecewise-linear forcing. That does not match the real
`Component::advance()` contract: one snapshot, valid at `ctx.t_begin`. The
committed design feeds the substep's **mean** (`Waveform::integrate(...) /
dt`) instead — see the class comment on `WindowCoupler` for the full
argument, and `docs/user/coupling-windows.md` for the user-facing version.
This is why `LinearOdeComponent` in `test/runtime/window_coupler_test.cpp`
is exact given a piecewise-CONSTANT forcing per substep, not
piecewise-linear.

## Refusal codes

`E-SCHED-WINDOW-*` (schedule-time, `validate_coupling_window`): `EMPTY`,
`AMBIGUOUS`, `COMPONENT`, `INTERVAL`, `POLICY`, `ORDER`, `SHORTEN`,
`CAPACITY`, `GAUSSSEIDEL-NOLAG`. `E-SCHED-WINDOW-OVERLAP` (schedule-time,
`schedule()` itself, before any per-region gate runs): a component or
channel named by more than one window-scoped region. `E-RT-WINDOW-*`
(`WindowCoupler::create`/`run_window`): `EMPTY`, `NULL`, `INTERVAL`,
`ORDER`, `CHANNEL`, `ROLLBACK`, `NOCONVERGE`. Every one has a test that
makes it fire (`test/schedule/window_scope_test.cpp`,
`test/schedule/plan_test.cpp`, `test/runtime/window_coupler_test.cpp`'s
`WindowCouplerRefusals` suite) — per [](testing.md), an unexercised refusal
is not verified.

## Two bugs found in the straddle replay, and one gap found but not fixed

Two defects made a `Fixed` member's error stop shrinking with refinement
(O(1), not the order-2 the design targets) in the asymptotic regime
(small `k*dt`); both are fixed, and the fix is verified directly in
`test/runtime/window_coupler_test.cpp`'s
`ErrorConvergesAtOrderTwoFixedFixedIncommensurate` and its
Adjustable/Fixed-mixed siblings, at every refinement step, to
[1.9, 2.1].

**Root cause (the dominant one — O(1), not a smaller-order defect):**
`sweep_window()`'s commit code set `straddle_window_start = T0` — the
window's own nominal start — but `resolve_pending_straddles()`'s
`restore_state()` returns to whatever `save_state()` actually
checkpointed at the TOP of that same `sweep_window()` call, which is
`start_clock[i]`, not `T0`. For a member that straddles only once
(its first straddle ever), the two are equal and nothing shows. For a
member straddling EVERY window (the 7:13 case), `start_clock[i] > T0`
by the previous window's own overshoot, so the next window's replay ran
its substeps as if it began a whole component's worth of elapsed time
short of where `restore_state()` actually put it — a mislabelling whose
size is a fixed fraction of the window rather than a fraction that
shrinks with it, which is exactly why the resulting error was O(1)
rather than O(h^2). Fix: record `start_clock[i]`, not `T0`.

**Second defect (smaller order, visible only once the first was fixed):**
a straddling substep's boundary value at T1 was computed as a single
`advance()` call over the WHOLE straddling substep, fed one BLENDED mean
covering both the in-window and tail portions, with the T1 value read
off by linear interpolation between the substep's two endpoints. That
interpolant does not recover x(T1) to the substep's own order — an
ordinary (non-straddling) substep gets its endpoint from an `advance()`
call whose own mean is exact for that shorter interval, but the
interpolated value here is not on the same trajectory a direct call
with the true in-window mean would produce. Fix: split the call at T1 —
one `advance()` for `[sub.t0, T1)` with the in-window mean, a second for
`[T1, sub.t1)` with the boundary/extrapolated mean — in both
`sweep_window()` and `resolve_pending_straddles()`'s replay of the same
substep, matching how an Adjustable member's own last substep already
ends exactly at T1 for free.

**Measured** (this build, tier 0): with both fixes, `d = 35*m` for
`m` in {2,4,8,16,32} (`Tc = 13/d`, `hB = 7/d`, `n_windows = 7*m` — a
multiple of the 7:13 straddle's own phase cycle, since gcd(7,13)=1, so
T_total and the final window's phase stay fixed across refinements),
Fixed/Fixed order is 2.09, 2.05, 2.02, 2.01 between successive steps;
Adjustable/Fixed and Fixed/Adjustable land in the same [2.0, 2.1] band.
Refining by naive doubling instead of this phase-aligned scheme (same
coefficients, same Tc/hB ratio) gives a non-monotone, sometimes
order-0-looking sequence purely from measuring different runs' final
windows at different points in that 7-window cycle — not a remaining
bug, but a reason to align refinement to the cycle rather than read
order off an arbitrary fixed wall-clock time.

**The supported envelope, enforced at `WindowCoupler::create`.** The
straddle replay handles one straddling member per window. A member straddles
only if it is Fixed and its step does not divide the window, so:
- two such members are refused (`E-RT-WINDOW-STRADDLE`), because each would
  replay against the other's unresolved trajectory;
- a Fixed step longer than the window is refused (`E-RT-WINDOW-LONGSTEP`),
  because it spans a window end it never samples.
Both were run before they were refused. Fixed/Fixed threw from
`Waveform::integrate`, and the Fixed long-step mixed case returned a
non-converging answer with no diagnostic, which is why they are refusals and
not merely documented limits. Inside the envelope, a window that neither step
divides still converges at order 2
(`AWindowNeitherStepDividesConvergesAtOrderTwo`). Supporting simultaneous
straddles needs a joint replay over a consistent snapshot of every member,
and supporting long Fixed steps needs a commit path that can record "no new
data this window".

## The window-to-window seed

`WindowCoupler::sweep_window()` seeds `y_fed` (sweep 0's "what was fed") from
`fuse::extrapolate_boundary_value()`: a polynomial fit of degree
`min(max(order, 2), history.size()) - 1` through the last `max(order, 2)`
converged window-end values, evaluated at the new window's own sample times.
With a single history point (a member's first-ever window) this reduces to
persistence (degree 0); from the second window on it is a genuine forward
extrapolation. An earlier attempt to seed `y_fed` by shifting the previous
window's own fine-grained waveform back one window length -- persistence,
not extrapolation -- was measured and rejected: it needed MORE
sweeps to converge (`gray_scott_multirate`'s pinned totals rose 9/13/12 ->
12/16/15).

The Runner compiles a window-scoped region's own window-to-window seed as
TWO separate quantities, not one:

1. `Runner::Impl::advance_window()`'s own import trajectory, for a lagged
   member's substep computation this sweep (`src/runtime/iteration.cpp`,
   the `use_predictor` branch) -- applies the SAME `extrapolate_boundary_
   value()` predictor, over the consuming member's own
   `WindowMemberState::history`, and is what `test/runtime/window_runner_
   straddle_test.cpp`'s own bit-for-bit parity cases against `WindowCoupler`
   check.
2. `Runner::Impl::run_region()`'s own `R.x_old` pre-loop seed -- the ONLY
   value the region's generic Picard/accelerator machinery (the residual
   norm, `R.accel->update()`) ever sees as "the previous iterate". A raw read of the live, already-delivered buffer
   for every window instance past a member's first-ever one would be the SAME
   zeroth-order/persistence predictor the paragraph above rejects
   for `y_fed`. `AccelKind::None`'s `update()` ignores
   `x_old` entirely, so no test built against it could ever have noticed;
   `AccelKind::Aitken`/`SteffensenAccel`/`QuasiNewtonAccel` all read
   `x_old` directly (the relaxed output itself, and the residual
   difference a history-dependent relaxation weight is computed from),
   which is where the two drivers were measured to disagree, starting
   exactly at the window-0/window-1 boundary. The fix is
   applying the identical predictor to `R.x_old`'s own seed, gated the
   same way `advance_window()`'s own `use_predictor` branch is
   (`cfg.import_lagged`), reusing the SAME `window_member_config`/
   `window_member_state` lookup the pre-existing first-window bootstrap
   already performed narrowly (`history.size() == 1` only). Verified
   bit-for-bit against `y_fed`, element for element, for the two-member
   `LinearOdeComponent` fixture `test/runtime/window_runner_straddle_
   test.cpp` already uses, and against a new `AccelKind::Aitken` parity
   case in that same file.

**Aitken relaxation factor.** Applying the seed above to the
Brusselator's own `examples/brusselator` configuration (Jacobi,
`AccelKind::Aitken`, `max_iterations = 100`) made one window (window 3 of
6, in `test/verification/brusselator_restart_test.cpp`'s own
`SeparateLayoutJacobiAitken`/`GrowingLayoutJacobiAitken`) run the FULL
100-sweep cap without converging, where it converged in 25 sweeps without the seed -- confirmed genuine non-convergence (not merely slower
convergence) by raising the cap to 2000 sweeps and measuring the SAME
window still exhaust it. `AccelKind::IQN_ILS` on the identical
configuration showed no such regression. The mechanism:
`AitkenAccel::update()` (`src/num/accel_relaxation.cpp`) accepted any
finite, nonzero relaxation factor `omega`, including a negative one; at the
Brusselator's own window 3, `omega` goes negative at sweep 3 and then
random-walks toward zero, and the residual plateaus rather than
contracting. A stationary scalar relaxation only contracts for `omega` in
`(0, 2(1 - Re(lambda)) / |1 - lambda|^2)` -- never for a non-positive
value -- which is the rationale for the rule below, not a convergence proof for
Aitken's own dynamic, history-dependent `omega` recursion. `AitkenAccel::
update()` now rejects a non-positive candidate and restarts to `omega0_`
(the configured initial relaxation) instead of accepting it; of three
measured rejection strategies (keep the last accepted positive
value; this restart; a noise-perturbed restart), this is the only one that
converges the Brusselator's own windows 3, 6 AND 9 without breaking the
two bit-identical round-trip tests above, because it carries no
epsilon-scaled threshold to be round-off-sensitive about. `--driver
window --accel aitken` and `--driver runner`, same configuration, now both
report `--check: OK` with identical `final_mean_u`/`final_mean_v` and no
cap hit on either side.

## What is NOT built here

- `Runner` resolving `WindowMember` fields from a `CouplingGraph`/
  `FieldRegistry` the way a single-clock `RegionBlock` does, against a
  standalone `WindowCoupler` rather than through `schedule()`'s own compiled
  plan (`schedule()` itself DOES compile a window-scoped `IterationRegionDecl`
  — see above).
- Extending `IterationRegionDecl` itself with the opt-in.

`src/CMakeLists.txt` already registers `schedule/window.cpp` and
`runtime/window_coupler.cpp` into the `fuse::schedule`/`fuse::runtime`
library targets.
