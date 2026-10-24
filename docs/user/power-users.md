# Power-user guide

For runs that must be fast, scalable, reproducible and debuggable.

## Choose the build

Match the build to what you run ([](build.md)). A Release build with `-DFUSE_ALLOW_FETCH=OFF` against
your own tuned Kokkos is the production choice. Fetched Kokkos is a generic build and should not be
used for performance work. Every configure writes `fuse-resolved-dependencies.txt` into the build
directory; keep it with your results.

## Execution space and threads

Kernels run on a Kokkos execution space. Set it at start-up with `FUSE_EXECUTION_SPACE` or through
Kokkos' own options. For OpenMP configure Kokkos with `-DKokkos_ENABLE_OPENMP=ON` and set
`OMP_NUM_THREADS` and `OMP_PROC_BIND` as for any OpenMP code. Use Kokkos' usual `--kokkos-*` flags to
select devices.

## Reproducibility

Results are meant to be bit-for-bit identical across thread counts. `FUSE_FP_REPRODUCIBLE` is ON by
default and applies a build-wide floating-point policy that avoids contraction and reordering that
would change a sum. Turn it off (`-DFUSE_FP_REPRODUCIBLE=OFF`) only if you accept results that depend
on the compiler's choices. Components declare their determinism in `capabilities()`, and FUSE refuses a request for a
guarantee a member cannot give.

Convergence measures are independent of resolution and partitioning, so an iteration count does not
change when you change the rank count, though the last digits of a result can.

## Placement and scale

- Put heavy components on large rank sets and a cheap one on a small set. Disjoint sets run
  concurrently.
- Move remapping onto separate coupler ranks when it dominates ([](placement.md)).
- Build remap weights once and reuse them ([](transfer-maps.md)). Writing a map costs one run;
  reading it costs almost nothing, at any rank count.
- Coupling period is the least common multiple of component steps. Step choices that produce a long
  period make a long schedule; prefer commensurate steps.
- Use `.accumulate(...)` rather than exchanging every sub-step when a slower partner needs only an
  average.

## Launch modes

In one executable, `mpiexec -n N ./app`. Several separately built executables can be launched
together or joined later ([](multiple-executables.md)). Unmodified codes talk over HDF5 with LowFive
([](lowfive.md)).

## Linear algebra and Newton

The library has distributed vectors and sparse matrices, preconditioned CG and GMRES, and a
Jacobian-free Newton-Krylov solver. With `-DFUSE_ENABLE_PETSC=ON` PETSc can provide the substrate
([](petsc-reaction-diffusion.md)). A derivative-checking toolkit (adjoint identity, Taylor remainder)
tests any operator that claims a Jacobian or a transpose; use it on your own operators before trusting
a Newton run.

## Diagnostics

| Variable | Effect |
|---|---|
| `FUSE_ON_ERROR` | `throw`, `backtrace`, `debugger` or `abort`: what happens at a failure |
| `FUSE_SCHEDULE_DEBUG` | Print each scheduled instant |
| `FUSE_TRANSPORT_DEBUG` | Trace data movement between ranks |
| `FUSE_COLLECTIVE_TRACE` | Trace collective operations, for a hang |
| `FUSE_CONSERVATION_DEBUG` | Explain a conservation check |
| `FUSE_NEWTON_DEBUG` | Trace Newton-Krylov iterations |
| `FUSE_EXECUTION_SPACE` | Select the Kokkos execution space |

Set them in the environment of every rank. Output is verbose; pipe it to a file per rank.

## Automating checks

An example is only useful if it can fail. Give your own coupling a gate in the same style as
`examples/mms_pair`: compare against an exact or reference solution, check the conservation counters
in `runner.stats()`, and return a non-zero exit code on failure. Keep `--dump` output as a golden file
so a change to the compiled coupling shows up in review.
