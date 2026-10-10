# FUSE

**F**lexible, **U**nified, **S**calable **E**xchange — a C++20 library for coupling multiphysics and
multiscale simulations. Portable by construction, performant by measurement.

A coupled simulation is authored as a graph, compiled to an inspectable operator IR, scheduled with
exact rational time and an initialisation-time deadlock proof, and executed by an allocation-free
runtime on Kokkos — over a native distributed unstructured mesh or MOAB, with collective failure
reconciliation, conservative non-matching remap, waveform and space-time coupling, quasi-Newton
acceleration, distributed Krylov and direct linear solvers, two C ABIs (participant and provider,
C and Fortran) over one canonical descriptor, and a coupling split across more than one separately
built executable.

FUSE unifies two verified code bases: a coupling compiler and runtime, and a coupling spine
contributing the semantic model, compatibility inference, collective reconciliation and provider ABI.
Import baselines are pinned in [`UPSTREAMS.lock`](UPSTREAMS.lock).

## Documentation

Built with Sphinx and MyST; the sources are in [`docs/`](docs/).

```bash
pip install -r docs/requirements.txt
sphinx-build -W -b html docs docs/_build/html
```

- **User guide** — building (dependency-tiered build configurations), quickstart, a map of every
  public namespace, the fluent API, coupling an existing code through C or Fortran, coupling across
  separately launched executables
- **Developer guide** — design invariants, layering, conventions, testing, ABI rationale, roadmap

## Status

**S0 through S6 are complete** (S6's conformance case closed 2026-10-07; of its planned legs only the
Python front end and Python provider kind are absent — `FUSE_ENABLE_PYTHON` exists as an option with
nothing behind it). `fuse::` now spans all thirteen
of its public namespaces — `base`, `model`, `ir`, `schedule`, `runtime`, `num`, `backend` (the
serial mesh, MOAB, and a native distributed unstructured mesh with its own halo exchange and
repartition/migration), `linalg` (a row-distributed vector and sparse matrix, preconditioned PCG
and GMRES(*m*), and an exact Schur-complement direct solver), `partition`, `search`, `transport`,
`field` and `abi` — and both C ABIs (participant and provider, each with a Fortran binding) are
implemented and tested. A coupled run can also be split across more than one separately built
executable, launched together under one `mpiexec` (MPMD) or as independently started programs that
rendezvous over MPI's dynamic process management; see
[`docs/user/multiple-executables.md`](docs/user/multiple-executables.md).

**Host-verified this cycle** (this sandbox cannot start MPI at all; every parallel capability is
compile-checked here and run on the user's own machine, recorded by date in
[`KNOWN_ISSUES.md`](KNOWN_ISSUES.md)): the MOAB mesh backend's full parallel suite (12/12 rows,
2026-10-08); the coupling-window runtime across more than one rank; and, for the first time, a
genuine multi-executable run of the Brusselator example at 3 and 5 ranks
(`verification.brusselator.participants.mpmd.{3,5}`) — `KNOWN_ISSUES.md` records the 3-rank row's
reported means as bit-identical to the single-process reference, and the 5-rank row, where one
participant's own physics is itself decomposed across two ranks through the new distributed
Krylov/Schur solvers, as agreeing to 8.9e-14 / 9.4e-14, consistent with that solve's own round-off.
**Blocked by this host's own MPI, not by a defect in the library:** the accept/connect launch mode
for a truly independent multi-executable launch (`KNOWN_ISSUES.md` D23 — this host's MPICH cannot
merge an intercommunicator between two same-node jobs; a configure-time probe detects this and
disables the mode rather than let it abort inside MPI) and LowFive's distributed, cross-process
transport (`KNOWN_ISSUES.md` D14 — the installed LowFive library needs rebuilding against a newer
`fmt`). LowFive's single-process interception path is implemented and measured.

**Open, and recorded:** `FUSE_LINALG_BACKEND=builtin` does not currently configure a full build —
several tests and examples link Eigen directly regardless of the chosen backend (D26); the native
mesh backend's remap halo has no reverse-exchange counterpart yet, needed only for a
conservation-audit tool nothing in the tree currently calls (D27). `S7` has the native linear
solvers above but no PETSc-based monolithic/implicit solver plugin or adjoint verification; `S8`
(`physics-kit`) and `S9` (portability, scale, an adversarial suite) have not started.
Checkpoint/restart to disk does not exist — only in-memory rollback within an iteration region.
Both parent trees ([`upstream/soma`](upstream/soma), [`upstream/karma`](upstream/karma)) are still
present — retiring them is S6's and S8's own exit condition and has not been done.

The [roadmap](docs/dev/roadmap.md) lists every stage and its exit condition, and every page marks
whether a capability is verified, implemented, designed only, a deliberately rejected configuration,
or absent.

## License

BSD 3-Clause. © 2006–2026 Vijay Mahadevan. See [`LICENSE`](LICENSE).

Open items, and which of them need the repository owner rather than a
contributor, are in [KNOWN_ISSUES.md](KNOWN_ISSUES.md).
