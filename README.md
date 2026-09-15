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

- **User guide** — building (tiers), quickstart, the fluent API, coupling an existing code through
  C or Fortran
- **Developer guide** — design invariants, layering, conventions, testing, ABI rationale, roadmap

## Status

S0: repository and tooling. No library source yet; nothing is imported. The
[roadmap](docs/dev/roadmap.md) lists the stages and their exit conditions, and every page marks
whether a capability is verified, implemented, designed only, a deliberate refusal, or absent.

## License

BSD 3-Clause. © 2006–2026 Vijay Mahadevan. See [`LICENSE`](LICENSE).
