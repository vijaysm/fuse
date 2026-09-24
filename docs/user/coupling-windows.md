# Coupling two clocks over a window, not a tick

An iteration region's default form couples components on ONE clock: every
member advances by the same interval, and the region's sweep loop rolls
everyone back to the same instant. That default is unchanged and is what you
get unless you opt in to what this page describes.

Two components rarely share a clock in practice. A structural solver
stepping every 13 s and a thermal solver stepping every 7 s do not
coincide again until 91 s — and refusing to couple them until then is not
an option. **A coupling window is the fix: couple every 13 s (the slower
component's own step), and remap the faster component's steps onto that
window exactly, never by waiting for a common multiple.**

## Declaring a window

A window's length comes from either a NAMED component's own interval, or an
explicit value:

```cpp
fuse::CouplingWindowScope scope;
scope.window_component = "structural";      // couple every time it steps
// scope.explicit_interval = fuse::Rational{13};  // ...or state the window directly
scope.order = 2;                             // interpolation/extrapolation floor
scope.policies = {
    {"structural", fuse::DtPolicy::Fixed},
    {"thermal",    fuse::DtPolicy::Adjustable},
};
```

Every component that owns either end of a channel in the region must appear
in `policies` exactly once. Validate it before use — this is a schedule-time
check, independent of the runtime driver:

```cpp
const auto ok = fuse::validate_coupling_window(graph, region, scope);
```

Without this opt-in, a two-clock region is refused today exactly as it
always has been (`E-SCHED-REGION`) — see
`test/schedule/multirate_region_test.cpp`.

## Fixed or Adjustable, per component

Each component in a window declares how its own step relates to the window
boundary:

**Adjustable** shortens its last substep in the window so it lands exactly on
the boundary — a 7 s step inside a 13 s window runs 7, then 6. No
straddling, no extrapolation, ever, for this component.

**Fixed** keeps its own step regardless. When that step would cross the
window end, it runs anyway (the *straddling* step) using the partner's
trajectory **extrapolated** past the window end. That step is provisional:
at the start of the *next* window, it is rolled back and recomputed using
the partner's now-converged data for the part inside the window that just
finished, and a fresh extrapolation for the part beyond it.

A window may mix the two — one component Fixed, the other Adjustable — and
that pairing is exercised directly
(`test/runtime/window_coupler_test.cpp`'s
`IncommensurateSevenThirteenAllPolicyPairs`).

## What a partner actually receives

`fuse::Component::advance()`'s own contract hands the implementer one
snapshot per call, valid at the interval's start — not a trajectory. To get
second-order accuracy out of that, the driver feeds each substep the
sender's **mean** over exactly that substep's interval, not a point sample —
the same convention an ordinary `AccumOp::Mean` channel already uses. The
sender's own outgoing data is represented with a linear-in-time waveform,
which is exact for `integrate()` over any sub-interval of its support, so
the substep means handed to different consumers over the same sender window
always sum to the sender's whole-window integral to round-off: the exchange
is conservative in time by construction, not by a separate check.

## Running it

```cpp
fuse::WindowCouplerConfig cfg;
cfg.window_length = fuse::Rational{13};
cfg.acceleration = fuse::AccelConfig{.kind = fuse::AccelKind::Aitken};
cfg.convergence = /* ... */;

auto coupler = fuse::WindowCoupler::create(members, channels, cfg).value();
for (int w = 0; w < n_windows; ++w) {
  const auto r = coupler->run_window();  // one window, sweeping to convergence
}
```

Convergence and acceleration act on the WHOLE window's exchanged trajectory
— every substep sample, not only the window-end value — exactly as
[](convergence.md) describes for a single-clock region.

## What this does not do yet

`WindowCoupler` is a runtime driver you construct directly from
`WindowMember`/`WindowChannel`, not (yet) something `schedule()` produces
from an `IterationRegionDecl` on its own. Wiring the two together —
resolving `WindowMember::provided`/`required` from the `FieldRegistry` the
way a single-clock region does — is designed but not built.
