# Mutation evidence — `compiler`

Run 2026-09-17 against build `build-fuse-s1`. Regenerate with `tools/mutate.py --write-log`; do not edit.

| Mutation | File | Verdict | Killed by |
|---|---|---|---|
| `dag-allows-second-producer` — an arc accepts a second producer, so the IR is no longer SSA | `include/fuse/ir/dag.hpp` | **KILLED** (test fails) | `ir.ir_dag` |
| `dump-locale-unpinned` — the IR dump formats through the host locale, so golden bytes depend on LC_NUMERIC | `include/fuse/ir/dump.hpp` | **KILLED** (test fails) | `ir.ir_dump` |
| `clock-accepts-nonpositive-interval` — a component may declare a zero or negative time step | `include/fuse/schedule/clock.hpp` | **KILLED** (test fails) | `schedule.clock` |
| `period-tick-limit-ignored` — an unusably long common period is passed downstream instead of refused | `include/fuse/schedule/clock.hpp` | **KILLED** (test fails) | `schedule.clock` |
| `overflow-escapes-as-exception` — an unrepresentable common period escapes as an exception instead of E-SCHED-PERIOD | `include/fuse/schedule/clock.hpp` | **KILLED** (test fails) | `schedule.clock` |
| `scale-emitted-as-double` — the unit conversion loses exactness, so 1/1000 becomes a rounded double | `src/ir/rules_channel.cpp` | **KILLED** (test fails) | `ir.scale` |
| `unlowerable-step-silently-dropped` — an authorized operation the compiler cannot emit is skipped instead of refused | `src/ir/rules_channel.cpp` | **EQUIVALENT** (No graph the model can currently express makes the connection check authorize a step outside {UnitScale, SpatialTransfer, TemporalTransfer}: the graph has no way to declare a transform, and a component-count mismatch is caught earlier by the IR type check (E-IR-TYPE). The branch is therefore defensive, and its classification is covered exhaustively by Scale.EveryAuthorizableOperationIsEitherEmittedOrRefused. When declared transforms reach the graph -- the checker already supports them -- this becomes reachable and must be killed.) | `ir.scale` |
| `merge-keeps-last-writer` — the old last-writer-wins binding, which deletes every other writer's dependence | `src/ir/lowering.cpp` | **KILLED** (test fails) | `schedule.acyclicity_sweep` |
| `merge-typed-at-first-writer` — the combined value inherits the first writer's time level, so a lagged writer declared first cuts the unlagged writers' dependence | `src/ir/lowering.cpp` | **KILLED** (test fails) | `schedule.acyclicity_sweep` |
| `epoch-not-emitted-after-transport` — a data move with no agreement point, so a failure on one rank is found only at the next collective | `src/ir/rules_channel.cpp` | **KILLED** (test fails) | `schedule.epoch` |
| `reserved-multiscale-op-scheduled` — a reserved multiscale operator schedules as a no-op instead of being refused | `src/schedule/scheduler.cpp` | **KILLED** (test fails) | `schedule.epoch` |
| `lowering-skips-move-on-any-sharing` — the old condition: any shared rank suppresses the data move, so a partially shared placement delivers nothing to the ranks it does not share and never reaches the refusal either | `src/ir/rules_channel.cpp` | **KILLED** (test fails) | `ir.lowering_channel` |

**11 killed / 12 total.**
