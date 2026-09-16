# Mutation evidence — `runtime`

Run 2026-09-16 against build `build-fuse-s1`. Regenerate with `tools/mutate.py --write-log`; do not edit.

| Mutation | File | Verdict | Killed by |
|---|---|---|---|
| `epoch-schedule-sends-downward` — a participant hands its status to a higher index instead of a lower one, so no chain reaches participant 0 and some sends have no matching receive: a hang | `include/fuse/runtime/epoch_schedule.hpp` | **KILLED** (test fails) | `runtime.epoch_schedule` |
| `epoch-schedule-drops-the-answer` — participant 0 keeps the agreed status to itself, so every other participant waits for a message that never comes | `include/fuse/runtime/epoch_schedule.hpp` | **KILLED** (test fails) | `runtime.epoch_schedule` |

**2 killed / 2 total.**
