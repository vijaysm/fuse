# FUSE

**F**lexible, **U**nified, **S**calable **E**xchange — a C++20 library for coupling multiphysics and
multiscale simulations. Portable by construction, performant by measurement.

A coupled simulation is authored as a graph, compiled to an inspectable operator IR, scheduled with
exact rational time and an initialisation-time deadlock proof, and executed by an allocation-free
runtime on Kokkos — with collective failure reconciliation, conservative non-matching remap,
waveform and space-time coupling, quasi-Newton acceleration, and two C ABIs over one canonical
descriptor.

FUSE unifies two verified code bases: a coupling compiler and runtime, and a coupling spine
contributing the semantic model, compatibility inference, collective reconciliation and provider ABI.
Import baselines are pinned in [`UPSTREAMS.lock`](UPSTREAMS.lock).

## Documentation

Built with Sphinx and MyST; the sources are in [`docs/`](docs/).

```bash
pip install -r docs/requirements.txt
sphinx-build -W -b html docs docs/_build/html
```

- **User guide** — building (dependency-tiered build configurations), quickstart, the fluent API, coupling an existing code through
  C or Fortran
- **Developer guide** — design invariants, layering, conventions, testing, ABI rationale, roadmap

## Status

**S1, in progress.** `fuse::base` — exact time, units, ids, versions, diagnostics, canonical bytes, and the error/backtrace machinery — is ported, transformed and tested at the base build configuration (serial, Debug and Release). Before that: **S0 done** — Both parents are frozen, tagged `fuse-import-0`, and imported byte-identical
under [`upstream/`](upstream/) — 693 files, verified against the tagged trees by
`tools/verify-import.py`. The imported coupler suite runs from this repository's own superbuild:
**770/770 passing** in the dependency-free configuration. The imported spine tree builds clean in
Release; running its suite needs a host with working MPI. No `fuse::` source yet — that starts at S1.

The [roadmap](docs/dev/roadmap.md) lists the stages and their exit conditions, and every page marks
whether a capability is verified, implemented, designed only, a deliberately rejected configuration, or absent.

## License

BSD 3-Clause. © 2006–2026 Vijay Mahadevan. See [`LICENSE`](LICENSE).

Open items, and which of them need the repository owner rather than a
contributor, are in [KNOWN_ISSUES.md](KNOWN_ISSUES.md).
