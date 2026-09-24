# Developer guide

FUSE unifies two verified code bases: a coupling compiler and runtime, and a coupling spine
contributing the semantic model, compatibility inference, collective reconciliation and provider ABI.
The import baselines are pinned in `UPSTREAMS.lock` at the repository root.

Read [](invariants.md) first — it is short, and every page after it assumes those rules. Then
[](layering.md) for where code goes, [](conventions.md) for how it is written, and [](testing.md) for
what "done" means here, and [](rank-agreement.md) for how the compiler keeps a failure on one
rank from becoming a hang on all of them.

How code moves out of `upstream/` and into `fuse::` — the procedure, the naming map, and the
retirement schedule — is [](porting.md). The staged build order lives in [](roadmap.md).
[](coupling-windows.md) covers the multi-clock iteration-window mechanism (schedule-time opt-in
plus runtime driver) for whoever next touches it. [](nodal-fields.md) covers how a field's
discretization decides its buffer extent (`ContinuousGLL` order 1 is sized by vertex count, not
cell count) and what lowest-rank vertex ownership means for a nodal field in parallel.
