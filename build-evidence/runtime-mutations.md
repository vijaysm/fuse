# Mutation evidence — `runtime`

Run 2026-09-16 against build `build-fuse-s1`. Regenerate with `tools/mutate.py --write-log`; do not edit.

**`epoch-schedule-drops-the-answer`'s anchor was stale at 2026-09-19** (REVIEW.md
R-21 / BT-F5): the code it targeted, a flat `result_send_to.push_back()` loop,
had been replaced by the binomial-tree `result_send_count` before this KILLED
verdict was recorded, so `text.count(old) == 0` and the verdict below was
unverifiable against the implementation that shipped. The anchor in
`build-evidence/mutations.toml` has been rewritten against the current
`result_send_count` line, reproducing the same "drops the answer" defect. The
table row below is therefore the PRIOR run's verdict against the OLD anchor,
kept for history, not a re-verified one: `tools/mutate.py --area runtime
--build-dir <dir> --write-log` needs to run against a clean tree (this file's
own convention) to produce a current KILLED/SURVIVED verdict for the rewritten
anchor. `tools/mutate.py --check-anchors` (no build) does confirm the new
anchor text exists exactly once in the current source.

| Mutation | File | Verdict | Killed by |
|---|---|---|---|
| `epoch-schedule-sends-downward` — a participant hands its status to a higher index instead of a lower one, so no chain reaches participant 0 and some sends have no matching receive: a hang | `include/fuse/runtime/epoch_schedule.hpp` | **KILLED** (test fails) | `runtime.epoch_schedule` |
| `epoch-schedule-drops-the-answer` — participant 0 keeps the agreed status to itself, so every other participant waits for a message that never comes | `include/fuse/runtime/epoch_schedule.hpp` | **KILLED** (test fails) | `runtime.epoch_schedule` |

**2 killed / 2 total.**
