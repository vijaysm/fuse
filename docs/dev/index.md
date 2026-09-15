# Developer guide

FUSE unifies two verified code bases: a coupling compiler and runtime, and a coupling spine
contributing the semantic model, compatibility inference, collective reconciliation and provider ABI.
The import baselines are pinned in `UPSTREAMS.lock` at the repository root.

Read [](invariants.md) first — it is short, and every page after it assumes those rules. Then
[](layering.md) for where code goes, [](conventions.md) for how it is written, and [](testing.md) for
what "done" means here.

The staged build order lives in [](roadmap.md).
