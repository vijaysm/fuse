# Mutation evidence — `compiler`

Run 2026-09-15 against build `build-fuse-s1`. Regenerate with `tools/mutate.py --write-log`; do not edit.

| Mutation | File | Verdict | Killed by |
|---|---|---|---|
| `dag-allows-second-producer` — an arc accepts a second producer, so the IR is no longer SSA | `include/fuse/ir/dag.hpp` | **KILLED** (test fails) | `ir.ir_dag` |
| `dump-locale-unpinned` — the IR dump formats through the host locale, so golden bytes depend on LC_NUMERIC | `include/fuse/ir/dump.hpp` | **KILLED** (test fails) | `ir.ir_dump` |
| `clock-accepts-nonpositive-interval` — a component may declare a zero or negative time step | `include/fuse/schedule/clock.hpp` | **KILLED** (test fails) | `schedule.clock` |
| `period-tick-limit-ignored` — an unusably long common period is passed downstream instead of refused | `include/fuse/schedule/clock.hpp` | **KILLED** (test fails) | `schedule.clock` |
| `overflow-escapes-as-exception` — an unrepresentable common period escapes as an exception instead of E-SCHED-PERIOD | `include/fuse/schedule/clock.hpp` | **KILLED** (test fails) | `schedule.clock` |

**5 killed / 5 total.**
