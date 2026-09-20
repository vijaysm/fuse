# Mutation evidence — `transport`

Run 2026-09-19 against build `build-fuse-s1`. Regenerate with `tools/mutate.py --write-log`; do not edit.

| Mutation | File | Verdict | Killed by |
|---|---|---|---|
| `peer-plan-skips-the-numbering-check` — the peer-wise plan is accepted without checking that each peer's expectation matches what its partner intends to send, so a communicator mismatch delivers to the wrong rank instead of failing at setup | `src/transport/mpi_transport.cpp` | **NOT-ASSESSED** (needs a working MPI launcher; run with FUSE_MUTATE_MPI=1 on a host that has one) | `parallel.mpi_transport.2` |
| `peer-plan-drops-the-local-copies` — same-rank matches are silently discarded, so under partial overlap the shared ranks keep stale values | `src/transport/mpi_transport.cpp` | **NOT-ASSESSED** (needs a working MPI launcher; run with FUSE_MUTATE_MPI=1 on a host that has one) | `parallel.mpi_transport.2` |
| `build-plan-skips-the-components-agreement` — R-30: MpiTransport::build_plan() stops checking that every rank in the src/dst rank set agrees on TransportSpec::components, so two ranks calling it with different component counts size MPI_Alltoallv's count arrays inconsistently instead of refusing at setup | `src/transport/mpi_transport.cpp` | **NOT-ASSESSED** (needs a working MPI launcher; run with FUSE_MUTATE_MPI=1 on a host that has one) | `parallel.mpi_transport.2` |

**0 killed / 3 total.**
