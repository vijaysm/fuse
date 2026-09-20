# Mutation evidence — `runtime`

Run 2026-09-19 against build `build-fuse-s1`. Regenerate with `tools/mutate.py --write-log`; do not edit.

| Mutation | File | Verdict | Killed by |
|---|---|---|---|
| `epoch-schedule-sends-downward` — a participant hands its status to a higher index instead of a lower one, so no chain reaches participant 0 and some sends have no matching receive: a hang | `include/fuse/runtime/epoch_schedule.hpp` | **KILLED** (test fails) | `runtime.epoch_schedule` |
| `epoch-schedule-drops-the-answer` — participant 0 keeps the agreed status to itself, so every other participant waits for a message that never comes | `include/fuse/runtime/epoch_schedule.hpp` | **KILLED** (test fails) | `runtime.epoch_schedule` |
| `cross-rank-region-check-fires-unconditionally` — R-10: the cross-rank iteration-region refusal (E-RT-REGION-MPI) fires even with no RunnerConfig::base_communicator configured, wrongly refusing every single-process caller that places a region's components on different declared ranks | `src/runtime/iteration.cpp` | **SURVIVED** (suite still green) | `runtime.runner` |
| `region-residual-skips-the-cross-rank-reduction` — R-10: run_region() stops agreeing the residual across the region's participants and falls back to each rank's own local Euclidean sum, exactly the defect the fix closes. On RunnerTransport.RegionConvergesInTheSerialIterationCount's unequal 2/1/1-style split this desynchronises which sweep each rank exits on -- the expected kill is the iteration-count assertion failing outright on the ranks that finish before the timeout, or the whole job timing out (a slower rank's partners move on to the region's own exit instructions while it is still sweeping, which is the hang rank-agreement.md's Epoch exists to prevent and this reduction is the region-local analogue of); either is a correct kill for this mutation, and the 120s TIMEOUT test property makes the second case an unambiguous FAILED rather than a silent PASS. | `src/runtime/iteration.cpp` | **NOT-ASSESSED** (needs a working MPI launcher; run with FUSE_MUTATE_MPI=1 on a host that has one) | `parallel.runner_transport.2` |

**2 killed / 4 total.**
