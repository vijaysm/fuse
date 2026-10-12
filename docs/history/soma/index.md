# Soma design history

Historical design, planning and handoff documents for `soma::`, the coupler library imported
byte-identically as `upstream/soma` (parent repository `coupe`, commit `e52b981`, imported
2026-09-14 per `UPSTREAMS.lock`) and in the process of being retired as its behaviour is ported
natively into `fuse::`. Every page here is preserved **verbatim** from that commit; none describes
FUSE's current architecture. See [](../../dev/soma-import.md) for the per-file and per-case record
of what was ported, superseded, or remains an open gap, and [](../../dev/porting.md) for the
general porting procedure this track followed.

```{toctree}
:maxdepth: 1

specs/2026-09-02-soma-design
specs/2026-09-11-comm-graph-design
plans/2026-09-02-soma-implementation
plans/soma-phase-0
plans/soma-phase-1
plans/soma-phase-2
plans/soma-phase-3
plans/soma-phase-4
plans/soma-phase-5
plans/soma-phase-6
plans/soma-phase-7
plans/soma-phase-8
BUILDING
HANDOFF
```
