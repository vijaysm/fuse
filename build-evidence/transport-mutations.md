# Mutation evidence — `transport`

Run 2026-09-21 against build `build-tier1`. Regenerate with `tools/mutate.py --write-log`; do not edit.

| Mutation | File | Verdict | Killed by |
|---|---|---|---|
| `peer-plan-skips-the-numbering-check` — the peer-wise plan is accepted without checking that each peer's expectation matches what its partner intends to send, so a communicator mismatch delivers to the wrong rank instead of failing at setup | `src/transport/mpi_transport.cpp` | **NO-BASELINE** (`parallel.mpi_transport.2` already fails without the mutation, so its failure with it applied is not evidence) | `parallel.mpi_transport.2` |
| `peer-plan-drops-the-local-copies` — same-rank matches are silently discarded, so under partial overlap the shared ranks keep stale values | `src/transport/mpi_transport.cpp` | **NO-BASELINE** (`parallel.mpi_transport.2` already fails without the mutation, so its failure with it applied is not evidence) | `parallel.mpi_transport.2` |
| `build-plan-skips-the-components-agreement` — R-30: MpiTransport::build_plan() stops checking that every rank in the src/dst rank set agrees on TransportSpec::components, so two ranks calling it with different component counts size MPI_Alltoallv's count arrays inconsistently instead of refusing at setup | `src/transport/mpi_transport.cpp` | **NO-BASELINE** (`parallel.mpi_transport.2` already fails without the mutation, so its failure with it applied is not evidence) | `parallel.mpi_transport.2` |
| `one-transport-shared-across-every-plan` — D2's fix: every CommGraph-backed plan gets its OWN Transport. Sharing one instance across plans puts the second channel's move() on a Transport planned for the first, which MpiTransport::prepare() refuses -- so the second channel stops working while the first still does. | `src/runtime/runner.cpp` | **NO-BASELINE** (`parallel.runner_transport.2` already fails without the mutation, so its failure with it applied is not evidence) | `parallel.runner_transport.2` |

**0 killed / 4 total.**
