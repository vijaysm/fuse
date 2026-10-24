# Power-user guide

For runs that must be fast, scalable, reproducible and debuggable. This page assumes you have a
coupling that works and want to run it well; [](environment.md) is the reference behind every
variable and field named here, and [](drivers.md) shows how to put them on your own command line.

## Choose the build

Match the build to what you run ([](build.md)). A Release build with `-DFUSE_ALLOW_FETCH=OFF`
against your own tuned Kokkos is the production choice; a fetched Kokkos is a generic build and
should not be used for performance work. Add only the rungs you need: MPI for more than one rank,
MOAB for mesh files and TempestRemap, PETSc for its solvers. Every configure writes
`fuse-resolved-dependencies.txt` into the build directory; keep it with your results, because it
is the only complete record of which Kokkos, MPI and linear-algebra backend produced them.

## Execution space and threads

Kernels run on a Kokkos execution space. The Runner's elementwise instructions run on the host
space by default and on the device space when `RunnerConfig::execution_space` is `Device`, or when
it is left at `Default` and `FUSE_EXECUTION_SPACE=device` is set. Transport, epochs, sinks,
accelerators and conservation checks always run on the host.

For OpenMP, configure Kokkos with `-DKokkos_ENABLE_OPENMP=ON` and set `OMP_NUM_THREADS` and
`OMP_PROC_BIND` as for any OpenMP code, or pass `--kokkos-num-threads=N`. Kokkos' `--kokkos-*`
flags are parsed and removed by `Kokkos::ScopeGuard(argc, argv)`, so they work with any example
that constructs the guard before its own parser (all but `brusselator`). Under MPI, each process
is a Kokkos instance: `mpiexec -n 4 ./app --kokkos-num-threads=8` is 32 threads.

## Reproducibility

Results are meant to be bit-for-bit identical across thread counts. `FUSE_FP_REPRODUCIBLE` is ON
by default and adds `-ffp-contract=off` to every target in every build type, so Release reproduces
Debug and a vectorised loop cannot mix fused and unfused rounding. Turn it off
(`-DFUSE_FP_REPRODUCIBLE=OFF`, or the `tier0-release-fast` preset) only if you accept results that
depend on the compiler's choices ([](build.md#floating-point-reproducibility-vs-speed)). The
option governs the host compilers; a CUDA build has the separate `-fmad=false`.

Components declare their determinism in `capabilities()` (`DeterminismLevel::Bitwise` and
weaker levels), and FUSE refuses a request for a guarantee a member cannot give. The Runner's own
reductions use deterministic kernels (`fuse::det_sum`, `fuse::det_dot`), which is what lets the
`example.*.omp_identical` test rows diff a run at one thread against a run at every core byte for
byte (`ctest -L omp`). Run the same check on your own driver with
`tools/compare-runs.cmake` ([](environment.md#benchmark-rows-and-tools)).

Convergence measures are independent of resolution and partitioning, so an iteration count does
not change when you change the rank count, though the last digits of a result can.

## Placement and scale

- Put heavy components on large rank sets and a cheap one on a small set. Disjoint sets run
  concurrently; overlapping sets serialise on the shared ranks ([](placement.md)).
- Move remapping onto separate coupler ranks when it dominates: a `CouplerDecl` with its own
  `RankSet` and `.remap_on(coupler)` on the channel. The layout driver's `--coupler e:f` is the
  quickest way to measure whether it helps ([](layout-driver.md)).
- Build remap weights once and reuse them ([](transfer-maps.md)). Writing a map costs one run;
  reading it costs almost nothing, at any rank count.
- The coupling period is the least common multiple of the component intervals. Steps that produce
  a long period produce a long plan; prefer commensurate steps, or a coupling window when the
  clocks are genuinely incommensurate ([](coupling-windows.md)).
- Use `.accumulate(...)` rather than exchanging every sub-step when a slower partner needs only a
  mean or an integral.

## Timing a run

FUSE has no internal timers. Time phases in the driver and reduce over ranks:

```cpp
#include <mpi.h>
#include <vector>

struct Stat { double min_s, mean_s, max_s; };

Stat reduce_stat(double local_s) {
  int ranks = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  double mn = 0, sum = 0, mx = 0;
  MPI_Allreduce(&local_s, &mn,  1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&local_s, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local_s, &mx,  1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return {mn, sum / ranks, mx};
}

// const double t0 = MPI_Wtime();
// const auto r = runner.run_period(components);
// const Stat period = reduce_stat(MPI_Wtime() - t0);
```

Time `initialize()` separately from the periods: it loads meshes, builds remap weights and the
transport plan, and is paid once. A large `max_s - min_s` on a period is load imbalance; look at
which component sits on the slow ranks. `examples/layout_driver/main.cpp` writes these rows as
`phase,ranks,layout,n,min_s,mean_s,max_s` CSV with `--timing out.csv`; in a serial program use
`std::chrono::steady_clock` in the same places.

## Scaling studies

For a strong or weak scaling study of the coupling machinery itself (transport, redistribution,
remap) without any physics, use the layout driver and `tools/scale-sweep.py`:

```sh
python3 tools/scale-sweep.py \
    --driver build/examples/fuse_example_layout_driver \
    --ranks 2,4,8,16 --shapes disjoint,identical --mode weak --entities 100000 --steps 5 \
    --check --out weak.csv
```

`--mode strong` keeps the total entity count fixed, `weak` scales it with the rank count. Each
cell is one `mpiexec -n <ranks> fuse_example_layout_driver --comp1 .. --comp2 .. --timing` run
and the rows are concatenated into `mode,ranks,layout,shape,n,phase,min_s,mean_s,max_s`. Pass
`--dry-run` to print the launch lines for a batch system. For a real application, drive the same
sweep over your own executable with the CSV format above, so that both can be plotted together.

Interpret with the right expectation: a disjoint layout moves every element across ranks on each
period, an identical layout moves nothing (the transport is local), and a partial overlap is in
between. If `period_k` does not fall with rank count in strong mode, check
`RunStats::remap_weight_nonzeros` per rank for imbalance and whether an epoch (rank agreement) per
transport dominates at small message sizes.

## Device runs

A device run needs a Kokkos with a CUDA, HIP or SYCL backend; FUSE refuses `Device` on any other
build with `E-RT-DEVICE`. Then:

```sh
FUSE_EXECUTION_SPACE=device mpiexec -n 4 ./app        # or RunnerConfig::execution_space = Device
```

Register fields in device memory (`fuse::MemorySpace::Device` in the `FieldRef`) wherever the
component keeps them there, otherwise every instruction that reads them refreshes a mirror. Read
the device counters in `runner.stats()` afterwards: `device_dispatches` must be non-zero or the
device path did not run; `device_staged_copies` is what to drive toward zero;
`device_transport_moves` is non-zero only with a GPU-aware MPI and tells you the transport stayed
on the device. `FUSE_TRANSPORT_HOST_STAGING=1` forces the host-staged transport for an A/B
comparison. `tools/bench-device.py` runs the scaled benchmark on host and device and reports
medians ([](environment.md#benchmark-rows-and-tools)). With several processes per node,
`KOKKOS_DEVICE_ID` overrides the round-robin device choice, and the environment variable takes
precedence over `--kokkos-device-id=`.

## Launch modes

In one executable, `mpiexec -n N ./app` with `transport = "mpi"`, `base_communicator =
MPI_Comm_c2f(MPI_COMM_WORLD)` and a `transport_setup` that calls `MpiTransport::set_communicator`
([](drivers.md#a-generic-yaml-driver)). Several separately built executables can be launched
together (`mpiexec -n 2 ./a : -n 3 ./b`) or started independently and joined through a port file
([](multiple-executables.md)). Unmodified HDF5 codes talk over LowFive ([](lowfive.md)).

### Multi-executable launch tips

- In launcher-style MPMD the `:`-separated segments must appear in the graph's component order;
  `connect()` checks `MPI_APPNUM` against the graph and refuses a mismatch with
  `E-CONNECT-APPNUM`.
- In independent launches start the acceptor first and in the background; the connector polls
  for the port file for `ConnectOptions::timeout_s` (30 s by default) and then fails with
  `E-CONNECT-TIMEOUT`. On a shared file system give the port file an absolute path visible to
  both jobs.
- Call `disconnect()` after the Runner and transport are destroyed and before `MPI_Finalize`;
  after finalize it refuses with `E-CONNECT-FINALIZED`.
- A build without `FUSE_ENABLE_CONNECT_ACCEPT` refuses independent launches with
  `E-CONNECT-ACCEPT-UNAVAILABLE`; only MPMD is available there.
- Under Open MPI the test rows set `OMPI_MCA_coll_tuned_use_dynamic_rules=1`,
  `OMPI_MCA_coll_tuned_alltoallv_algorithm=2` and the oversubscription variables
  ([](environment.md#openmp-kokkos-and-mpi-settings-fuse-relies-on)).
- Log per rank. The participant scripts name files `<log>.<rank>.log`, so a failure on one rank
  is one file to read.

## Linear algebra and Newton

The library has distributed vectors and sparse matrices, preconditioned CG and GMRES, and a
Jacobian-free Newton-Krylov solver for iteration regions ([](linear-algebra.md), [](convergence.md)). With
`-DFUSE_ENABLE_PETSC=ON` PETSc can provide the substrate ([](petsc-reaction-diffusion.md)); the
Brusselator's `--substrate native|petsc` switches between the two on one problem. A
derivative-checking gate (adjoint identity, Taylor remainder) tests any operator that claims a
Jacobian or a transpose and refuses a shape mismatch with `E-GATE-*`; use it on your own operators
before trusting a Newton run. `FUSE_NEWTON_DEBUG` prints every Newton step's residual norm,
forcing term, GMRES count and line-search step.

## Reading RunStats for performance

`runner.stats()` is per rank; the fields that tell you where the time goes:

| Field | Reading |
|---|---|
| `remap_weight_builds` | Matrices built in `initialize()`. More than one per distinct channel pair means a plan was not shared; 0 means every remap was the proved-coincident identity and no weights are involved at all |
| `remap_weight_nonzeros` | The cost of each `Remap`. Equal to the cell count for a permutation, larger for real overlap. Very different values across ranks is imbalance in the remap, not in the physics |
| `epochs_reconciled` | Rank agreements this rank took part in. Each is a collective; one per transport per period is normal, and a count that grows faster than the period count means agreements inside an iteration region |
| `total_iterations`, `windows_not_converged` | Sweeps and failures. The accelerator, not the rank count, is the lever ([](dn-heat.md)) |
| `instructions_executed / periods` | Plan length. A long period from incommensurate steps shows up here before it shows up in time |
| `device_*` | See [](#device-runs) |

Print them at the end of every run, even a production one; they are cheap and are the only record
of several things the delivered values cannot show.

## Diagnostics

Every variable, with the exact lines each prints, is in
[](environment.md#debug-and-trace-variables). The ones to reach for first:

| Variable | Effect |
|---|---|
| `FUSE_COLLECTIVE_TRACE` | One line per collective inside an iteration sweep; for a hang |
| `FUSE_TRANSPORT_DEBUG` | Transport planning and agreement outcomes per rank |
| `FUSE_SCHEDULE_DEBUG` | Each scheduled instant of a graph with iteration regions |
| `FUSE_CONSERVATION_DEBUG` | Before/after sums at each conservation check |
| `FUSE_NEWTON_DEBUG` | Newton-Krylov iterations |
| `FUSE_DEVICE_DEBUG` | Instruction classes and arc placement in device mode |
| `FUSE_EXECUTION_SPACE` | Select the Kokkos execution space |

Set them in the environment of every rank. Output is verbose; redirect it to a file per rank.

The error-policy variables (`FUSE_ON_ERROR=throw|backtrace|debugger|abort`, `FUSE_DEBUGGER`,
`FUSE_DEBUGGER_RANKS`, `FUSE_BACKTRACE`) are different: they take effect only in a program that
installs the environment policy. None of the examples does, so put this at the top of your own
`main()`:

```cpp
fuse::set_error_policy(fuse::error_policy_from_environment());
fuse::install_terminate_handler();
```

Then `FUSE_ON_ERROR=debugger FUSE_DEBUGGER=gdb FUSE_DEBUGGER_RANKS=2 mpiexec -n 4 ./app` opens a
debugger on rank 2 at the first thrown error, and `FUSE_ON_ERROR=abort` leaves a core file. The
policy covers thrown errors (assertions, invalid arguments, `FUSE_REFUSE`); a refusal that comes
back as a `Result` is data your driver handles ([](environment.md#error-policy-variables)).

## Automating checks

An example is only useful if it can fail. Give your own coupling a gate in the same style as
`examples/mms_pair`: compare against an exact or reference solution, check the conservation
counters in `runner.stats()` (`conservation_checks > 0` as well as `conservation_violations ==
0`, since zero checks make the second vacuous), check `windows_not_converged == 0`, and return a
non-zero exit code on failure. Register the binary in `ctest` so that the gate runs with the
suite.

### Golden `--dump` files

The IR and plan are a pure function of the graph and the build, and `--dump` prints them without
running. Keep the output under version control next to the graph:

```sh
./app --dump > golden/app.dump                 # once, after review
./app --dump | diff -u golden/app.dump -       # in CI: non-empty output is a changed coupling
```

A diff is a review item: either the graph changed on purpose and the golden file is updated, or
something changed the compiled coupling that should not have. Make the test assert on the dump's
*content*, because an empty dump also exits 0. The `generated:` mesh sources, exact rational
times (`t=1/4000`) and stable node numbering are what make the file diffable across machines
([](ir.md)).

## See also

- [](environment.md): the full reference for every variable, config field and statistic.
- [](drivers.md): building a command line that exposes them.
- [](troubleshooting.md): a debugging workflow and the error codes.
- [](placement.md), [](layout-driver.md), [](multiple-executables.md): where things run.
