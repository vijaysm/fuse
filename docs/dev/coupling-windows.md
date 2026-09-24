# Coupling windows: internals

User-facing semantics are in [](../user/coupling-windows.md). This page is
for whoever next touches `fuse/schedule/window.hpp`,
`fuse/runtime/window_coupler.hpp`, or the tests around them.

## Why this is two files, not one field

The natural spelling of the opt-in is a member on `IterationRegionDecl`
(`fuse/model/coupling_graph.hpp`). That file belongs to a different track of
this port and is not touched here. `CouplingWindowScope`
(`fuse/schedule/window.hpp`) carries the opt-in as a sibling parameter
instead — passed alongside the graph, not inside it — and
`validate_coupling_window()` is the legality gate. Wiring `schedule()`
itself to accept and act on a `CouplingWindowScope` is designed but not
built; today's contiguous-run rule (`mark_regions`,
`src/schedule/scheduler.cpp`) is unchanged, and a two-clock region without
the opt-in keeps refusing exactly as `test/schedule/multirate_region_test.cpp`
pins.

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
`CAPACITY`. `E-RT-WINDOW-*` (`WindowCoupler::create`/`run_window`): `EMPTY`,
`NULL`, `INTERVAL`, `ORDER`, `CHANNEL`, `ROLLBACK`, `NOCONVERGE`. Every one
has a test that makes it fire (`test/schedule/window_scope_test.cpp`,
`test/runtime/window_coupler_test.cpp`'s `WindowCouplerRefusals` suite) —
per [](testing.md), an unexercised refusal is not verified.

## What is NOT built here

- `schedule()`/`Runner` resolving `WindowMember` fields from a
  `CouplingGraph`/`FieldRegistry` the way a single-clock `RegionBlock` does.
- Extending `IterationRegionDecl` itself with the opt-in (belongs to the
  track that owns `fuse/model/coupling_graph.hpp`).
- `src/CMakeLists.txt` registering `schedule/window.cpp` and
  `runtime/window_coupler.cpp` into the `fuse::schedule`/`fuse::runtime`
  library targets — outside this track's file ownership; see the
  hand-off note this change's commit carries.
