# Mutation evidence — `fsi`

Run 2026-09-25 against build `b0`. Regenerate with `tools/mutate.py --write-log`; do not edit.

| Mutation | File | Verdict | Killed by |
|---|---|---|---|
| `fsi-traction-is-the-force-on-the-fluid` — KNOWN_ISSUES D9's root cause, reproduced exactly: the wall traction delivered to the beam carries the sign of the force on the FLUID (-p + 2 mu dv/dy) instead of on the wall. A negative added mass: the coupled step map's spectral radius was 1.115466 on the diagnostic grid and the energy grew 1.228 per step, while every self-consistency check (power balance, adjointness, energy budgets) passed, because each compared the delivered vector with itself. Killed only by a test that states the physics: a pressurised channel pushes its wall outward, and accelerating the wall away from the fluid is opposed. | `examples/fsi_channel/stokes_solver.hpp` | **KILLED** (test fails) | `verification.fsi_coupled` |

**1 killed / 1 total.**
