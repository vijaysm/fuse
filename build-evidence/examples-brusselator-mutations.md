# Mutation evidence — `examples/brusselator`

Run against build `b0`. Regenerate with `tools/mutate.py --write-log`; do not edit.

| Mutation | File | Verdict | Killed by |
|---|---|---|---|
| `brusselator-v-exchange-wrong-sign` — flips the sign of the exchange sink in V's own backward-Euler RHS (V_t = a*Lap(V) - q becomes + q) -- V gains mass from the exchange instead of losing it, silently destroying the NUMERICS.md 'U consumes +q; V consumes -(P_UV q)' design decision the whole conservative-transfer story depends on. | `examples/brusselator/species_component.hpp` | **KILLED** (test fails) | `example.brusselator` |
| `brusselator-exchange-bookkeeping-drops-trapezoid` — replaces the trapezoidal exchange-conservation bookkeeping (0.5*dt*(q_prev_.mean()+q_new.mean())) with a plain Riemann term (dt*q_new.mean()) -- UComponent::provided_ and the physical exchange U actually delivers are unchanged (still q_new, still fed to WindowCoupler exactly the same way), but the DIAGNOSTIC accumulator UComponent uses to measure 'what U reports it exchanged' no longer matches the quadrature WindowCoupler's own Linear-waveform reconstruction actually uses when integrating that same reported signal for V, so the exchange-conservation check sees a spurious O(dt) mismatch that is an artifact of the bookkeeping, not of the physics. | `examples/brusselator/species_component.hpp` | **KILLED** (test fails) | `example.brusselator` |
| `brusselator-u-consumes-start-of-step-reaction` — The first version of this port: U's step consumed dt*q_prev (the start-of-step rate) while WindowCoupler delivered V the trapezoid 0.5*dt*(q_prev+q_new) of the same samples, so the two species exchanged different amounts of one reaction and the Toy's discrete budget identity held only to first order (1.686e-3 per window). Restoring it must fail the round-off budget thresholds. | `examples/brusselator/species_component.hpp` | **KILLED** (test fails) | `example.brusselator` |

**3 killed / 3 total.**
