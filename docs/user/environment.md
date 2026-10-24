# Environment variables and runtime configuration

A reference for everything that changes how a FUSE program behaves without recompiling it:
environment variables, `RunnerConfig` and the other runtime structs, the statistics a run
reports, and the measurement tools in `tools/`. Use it when you need to know exactly what a
variable does and when it is read; [](drivers.md) explains how to expose these settings from
your own executable and [](power-users.md) when to reach for them.

Unless a row says otherwise, "set" means present in the environment with any value, and an empty
value counts as unset. Set a variable in the environment of *every* rank (`mpiexec -x NAME=value`
or your launcher's equivalent), because each rank reads it independently.

## Configuration variables

### `FUSE_EXECUTION_SPACE`

Which Kokkos execution space the Runner dispatches its elementwise kernels on (`Interpolate`,
`Remap`, `Sample`, `Accumulate`, `Extrapolate`, `Merge`, `Scale`, `Relax`, `Residual`).
Transport, epochs, sinks, accelerators and conservation checks are host-only regardless.

| Value (case-insensitive) | Effect |
|---|---|
| unset, empty, `host` | Host space |
| `device` | The device space (CUDA, HIP or SYCL) of the Kokkos FUSE was built against |
| anything else | `E-RT-SPACE` from `Runner::initialize` (`E-SEARCH-SPACE` in the search module) |

Precedence: the variable is read only when `RunnerConfig::execution_space` is left at
`Default`. A program that sets `Host` or `Device` explicitly is never overridden. Asking for
`device` on a build whose Kokkos has no device backend is a configuration error, `E-RT-DEVICE`,
not a silent fall-back (the search module alone degrades to host in that case). The `.device`
rows of the test suite set this variable to run every test under the device interpreter
unchanged.

### `FUSE_TRANSPORT_HOST_STAGING`

A non-empty value whose first character is not `0` forces `mpi_is_gpu_aware()` to report false,
so device-resident transports take the host-staged path instead of handing device buffers to
MPI. Use it to compare the two paths, or to work around an MPI that claims GPU awareness it does
not have. The effect is visible in `RunStats::device_transport_moves`, which stays 0.

### `FUSE_LOWFIVE_BINDINGS=<graph.yaml>`

Read only by the loadable `fuse-lowfive-vol` HDF5 plugin, which has no other way to learn the
graph. The plugin calls `load_yaml_file` on the path and builds its dataset bindings from the
graph's LowFive components; unset, it prints a warning and intercepts nothing. It works together
with HDF5's own `HDF5_VOL_CONNECTOR` and `HDF5_PLUGIN_PATH`, which tell HDF5 to load the plugin at
all ([](lowfive.md)). The C coupler participant of the Brusselator example unsets all three
before `MPI_Init` because it must not be intercepted itself.

### `KOKKOS_DEVICE_ID`

An integer device index. Processes that initialise Kokkos through the participant ABI
(`fuse_create`) or the test harness pick a device by local rank within the node (round-robin over
the visible devices); this variable overrides that choice. Precedence as the code behaves: the
environment variable is consulted *first*, and a `--kokkos-device-id=N` argument is used only when
the variable is unset. The ABI path has no `argv` at all, so there the variable is the only
override. `FUSE_DEVICE_DEBUG` prints the decision as
`[device-bind] local_rank=.. device_count=.. override=.. -> device_id=..`.

## Error policy variables

These variables configure what happens between "an error has been constructed" and "it is
thrown", in the style of PETSc's `-on_error_*` options.

:::{warning}
They are **opt-in**. FUSE parses them only when the program calls
`fuse::error_policy_from_environment()` and installs the result. No library code and none of the
shipped examples do that, so setting `FUSE_ON_ERROR=debugger` before `fuse_example_dn_heat` has
no effect. Your own driver enables them with three lines at the top of `main()`:

```cpp
#include <fuse/base/error.hpp>

int main(int argc, char** argv) {
  fuse::set_error_policy(fuse::error_policy_from_environment());  // FUSE_ON_ERROR and friends
  fuse::set_error_rank(/* MPI rank, once MPI is up */ 0);          // tag every report with it
  fuse::install_terminate_handler();                              // report an uncaught error, then abort
  // ...
}
```

Without the first line the default policy applies: print a report with a backtrace to
`stderr`, then throw.
:::

| Variable | Values | Meaning |
|---|---|---|
| `FUSE_ON_ERROR` | `throw`, `backtrace` (default), `debugger`, `abort` | `throw`: just throw. `backtrace`: print the report to stderr, then throw. `debugger`: spawn a debugger attached to this process, wait, then throw. `abort`: print the report, then `std::abort()` for a core file with no unwinding |
| `FUSE_DEBUGGER` | `lldb` (default), `gdb` | Which debugger `debugger` spawns; looked up on `PATH` |
| `FUSE_DEBUGGER_TERMINAL` | a terminal program (default `xterm`), or `none` | The terminal the debugger opens in; `none` attaches in the current terminal |
| `FUSE_DEBUGGER_RANKS` | comma-separated integers; empty (default) means every rank | Ranks that get a debugger; the others print and throw. Ranges (`0-3`) and the word `all` are not accepted |
| `FUSE_DEBUGGER_WAIT` | non-negative integer seconds, default 15 | How long the process blocks for the debugger to attach |
| `FUSE_BACKTRACE` | `0` or `1` (default on) | Whether each error captures a stack trace |

An unrecognised value is itself an error, not a silent default: with `FUSE_ON_ERROR=bogus` the
prologue throws `InvalidArgument` with the message
`FUSE_ON_ERROR='bogus': expected throw|backtrace|debugger|abort`. The debugger command that
`debugger` runs is `<terminal> -T 'fuse rank R pid P' -e <lldb|gdb> -p <pid>`.

The policy governs *thrown* errors: invariant violations (`FUSE_ASSERT`), invalid arguments and
the refusals that use `FUSE_REFUSE`. Most refusals in FUSE are not thrown; they come back to your
code as a failed `Result` carrying an `E-...` code, and your driver decides what to do with them.
If a run prints an `E-...` line and exits cleanly, that was a `Result`, and no setting of
`FUSE_ON_ERROR` would have stopped in a debugger there. Put a breakpoint on the message instead,
or run under the debugger from the start.

## Debug and trace variables

Each prints to `stderr` on every rank that reaches the code, so redirect per rank
(`mpiexec -n 4 ... 2>trace.%r` or your launcher's equivalent). Output volume is proportional to
instruction count times periods; use them on a reduced problem.

| Variable | Prints | Use it when |
|---|---|---|
| `FUSE_TRANSPORT_DEBUG` | MPI transport planning and agreement: `[build_plan rank r/n] ...` with the components involved, `[agreement rank ...]`, and per exchange `[outcome rank r] local=.. agreed=..` | Data arrives on the wrong rank, or a transport refusal names a rank you did not expect |
| `FUSE_SCHEDULE_DEBUG` | At schedule time, one block per instant: `[instant lo..hi] t=.. phase=.. members=.. entangled=..` followed by each instruction in it. Printed only for graphs that have at least one iteration region | An iteration region's membership or ordering is not what you declared |
| `FUSE_NEWTON_DEBUG` | `[newton-debug] region=.. newton_n=..`, one `block channel=.. length=.. newton_offset=..` line per block, then per Newton step `it= norm_f= eta= gmres_it= gmres_res= ls_lambda= precond_refreshed= mffd_evals=` and a final `status= final= initial=` | A Newton region stalls, the line search keeps cutting, or GMRES hits its cap |
| `FUSE_COLLECTIVE_TRACE` | One line per collective inside an iteration-region sweep: `FUSE_COLLECTIVE_TRACE rank= region= period= sweep= instr= kind= comm=base ordinal=` | A hang. The last line each rank printed shows where they diverged |
| `FUSE_CONSERVATION_DEBUG` | At each conservation check: `[conservation rank r period p arc %a<-%b] own before=.. (n of m weights non-zero) own after=.. total before=.. after=.. reduced=..` | A conservation violation, to see which side of the remap lost mass |
| `FUSE_DEVICE_DEBUG` | In device mode, the class of each instruction (generic, space-aware, host-only) and where each arc was placed; `[device-bind] ...` for the device choice | `device_staged_copies` is higher than expected, or a device run differs from host |
| `FUSE_NODAL_DEBUG` | `[nodal-dist rank r] arc %.. rows= in.elements= sum|in|=` for the distributed nodal remap | A nodal (vertex) field remap delivers zeros or wrong sums on some ranks |
| `FUSE_REMAP_HALO_TRACE` | `FUSE_REMAP_HALO_TRACE rank= channel='..' remap_weight_nonzeros= remap_weight_builds=` followed by every weight (`row(target)= col(source)= val=`), and from the halo builder the per-rank source and target bounding boxes and cell counts | A conservative remap across ranks has missing coverage (`E-REMAP-COVERAGE`) or a wrong weight; the output is large, so use a tiny mesh |
| `FUSE_CONNECT_CHAIN_TRACE=1` | Each step of an N-way `connect()` rendezvous (the value must be exactly `1`) | Three or more separately launched executables fail to meet ([](multiple-executables.md)) |

Example-only variables, which the library itself never reads: `FUSE_LOWFIVE_ATTACH_TRACE=1` and
`FUSE_LOWFIVE_LOG=<spdlog level>` in the Brusselator LowFive participants, and `FUSE_BRUSS_DEBUG`
in the Brusselator species components.

## `RunnerConfig`

Passed to `Runner::initialize(components, registry, config)`; the default-constructed value is
what every single-process example uses (`include/fuse/runtime/runner.hpp`).

| Field | Type | Default | Meaning |
|---|---|---|---|
| `transport` | `std::string` | `"copy"` | Registry name of the transport: `copy` (in-process), `mpi`, `lowfive`. An unknown name is `E-TRANSPORT-BACKEND` |
| `check_conservation` | `bool` | `true` | Run the `Constrain` checks the graph's `Extensive` fields imply |
| `conservation_tolerance` | `double` | `1e-12` | Relative tolerance of those checks |
| `abort_on_violation` | `bool` | `false` | `true` fails the period on a violation; `false` counts it in `RunStats` and continues |
| `abort_on_nonconvergence` | `bool` | `false` | `false` counts an iteration region that hit `max_iterations` in `windows_not_converged` and continues; `true` fails the period with `E-RT-NOCONVERGE` |
| `force_distributed_nodal_remap` | `bool` | `false` | Testing aid: take the distributed nodal-remap path even on one rank. Not a performance setting |
| `base_communicator` | `std::optional<int>` | unset | `MPI_Comm_c2f` of a communicator containing every rank of every component. Unset, the transport moves whole arcs (what `copy` does) |
| `force_this_process` | `std::optional<int>` | unset | Testing aid: pretend to be this rank with no communicator |
| `transport_setup` | `std::function<Result<void>(Transport&)>` | empty | Called once, right after the named transport is constructed. Required for `mpi` (`MpiTransport::set_communicator`) and `lowfive` (`set_location`); a no-op for `copy` |
| `execution_space` | `RunnerExecutionSpace {Default, Host, Device}` | `Default` | `Default` resolves from `FUSE_EXECUTION_SPACE` inside `initialize()`; the other two are never overridden |
| `resolved_execution_space()` | method | | What `Default` resolves to on this process, as `Result<RunnerExecutionSpace>` |

:::{note}
With `abort_on_nonconvergence = false`, `windows_not_converged` is the *only* record that an
answer in the run is not converged. A driver that does not print or check it is reporting an
unconverged answer as a converged one.
:::

## `ConnectOptions`

Passed to `fuse::connect()` by a program that is one executable of several
([](multiple-executables.md)):

| Field | Default | Meaning |
|---|---|---|
| `my_component` | | The `ComponentId` this executable implements |
| `timeout_s` | `30.0` | How long the connecting side waits for the acceptor's port file before `E-CONNECT-TIMEOUT` |
| `poll_interval_s` | `0.05` | How often it looks for the file |

The per-component launch description (`LaunchSpec`: `mode`, `appnum`, `is_acceptor`,
`port_file`, `service_name`, `plugin_path`) is part of the graph, not the runtime configuration;
see [](multiple-executables.md) and [](yaml.md).

## `ErrorPolicy`

What `error_policy_from_environment()` fills in and `set_error_policy()` installs
(`include/fuse/base/error.hpp`). Set it programmatically when you would rather not depend on the
environment:

| Field | Default | Environment variable |
|---|---|---|
| `on_error` | `OnError::PrintBacktrace` | `FUSE_ON_ERROR` |
| `capture_backtrace` | `true` | `FUSE_BACKTRACE` |
| `debugger` | `Debugger::Lldb` | `FUSE_DEBUGGER` |
| `terminal` | `"xterm"` (`""` attaches in the current terminal) | `FUSE_DEBUGGER_TERMINAL` |
| `debugger_ranks` | `""` (every rank) | `FUSE_DEBUGGER_RANKS` |
| `debugger_wait_seconds` | `15` | `FUSE_DEBUGGER_WAIT` |

```cpp
fuse::ErrorPolicy p;
p.on_error = fuse::OnError::Abort;   // a core file at the first invariant violation
fuse::set_error_policy(p);
```

## OpenMP, Kokkos and MPI settings FUSE relies on

FUSE reads none of these itself; they reach it through Kokkos and MPI.

| Setting | Notes |
|---|---|
| `OMP_NUM_THREADS`, `OMP_PROC_BIND` | Thread count and binding for a Kokkos OpenMP build. The thread-determinism tests run at 1 thread and at every host core with `OMP_PROC_BIND=false` and require byte-identical output |
| `--kokkos-num-threads=N`, `--kokkos-device-id=N`, other `--kokkos-*` | Parsed and removed by `Kokkos::ScopeGuard(argc, argv)`. Only `--kokkos-device-id=` is also inspected by FUSE code (the device-binding logic above); everything else goes to Kokkos untouched. The guard must be constructed before your own argument parser sees `argv` |
| `mpiexec -n P` | One process per rank. Component `RankSet`s must lie inside `[0, P)` and, for the `mpi` transport, inside the communicator passed as `base_communicator` |

:::{tip}
Open MPI users: the test rows that launch several executables run with
`OMPI_MCA_coll_tuned_use_dynamic_rules=1`, `OMPI_MCA_coll_tuned_alltoallv_algorithm=2`,
`OMPI_MCA_rmaps_base_oversubscribe=1` and `PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe`.
The first two select a stable `MPI_Alltoallv` algorithm; the last two allow more ranks than cores
on a workstation. If an MPMD or connect/accept launch misbehaves only under Open MPI, try them.
:::

## Profiling and measurement

### `RunStats`

`runner.stats()` after `initialize()` or any number of periods. Every counter is per rank.

| Field | Type | Meaning |
|---|---|---|
| `periods` | `int64` | Coupling periods completed |
| `instructions_executed` | `int64` | Plan instructions executed, summed over periods |
| `conservation_checks` | `int64` | `Constrain` evaluations. 0 means no transfer was checked, so the violation count says nothing |
| `conservation_violations` | `int64` | Checks that exceeded `conservation_tolerance` |
| `worst_conservation_error` | `double` | Largest relative error seen by any check; 0 both when checks were perfect and when none ran |
| `total_iterations` | `int64` | Sweeps of iteration-region bodies, summed over regions and periods |
| `windows_not_converged` | `int64` | Region-periods that hit `max_iterations`. Non-zero means some answer is not converged |
| `remap_weight_builds` | `int64` | Conservative remap matrices built in `initialize()`, one per distinct remap plan. 0 when every remap took the proved-coincident identity |
| `remap_weight_nonzeros` | `int64` | Non-zeros summed over those matrices; `n` for a permutation of `n` cells, more for genuine overlap |
| `checkpoint_verifications` | `int64` | Times the runtime compared a component's registered storage against its checkpoint after `restore_state()` |
| `epochs_reconciled` | `size_t` | Rank-agreement points this rank took part in; 0 in a serial run |
| `device_dispatches` | `int64` | Instructions whose kernels ran on the device space; exactly 0 under `Host` |
| `device_resident_arcs` | `int64` | Coupler-owned arcs allocated in device memory |
| `device_boundary_copies` | `int64` | Host/device crossings in device mode (field mirrors refreshed, cross-space sinks and epochs) |
| `device_staged_copies` | `int64` | Per-instruction staging copies for host-side operands; the number to drive to zero in a device run |
| `device_transport_moves` | `int64` | Transport exchanges that stayed in device memory (GPU-aware MPI or same-process `deep_copy`); 0 under `FUSE_TRANSPORT_HOST_STAGING` |

### `IterationStats`

`runner.iteration_stats()` returns one entry per iteration region for the most recent period:

| Field | Meaning |
|---|---|
| `iterations` | Sweeps taken this period |
| `converged` | Whether the criterion was met |
| `residual_history` | The 2-norm of the whole concatenated residual `x_new - x_old` after each sweep |

The residual history is what the convergence-order tables in [](dn-heat.md) are computed from.

### Timing

FUSE has no internal timers and no profiling API; Kokkos Tools and `Kokkos::Profiling` regions
are not used. Time phases in the driver, around `initialize()` and each `run_period()`, with
`MPI_Wtime` (reduce min, mean and max over ranks) or `std::chrono::steady_clock` in a serial
program. `examples/layout_driver/main.cpp` and the bench binary are the reference implementations;
both write `phase,ranks,...,min_s,mean_s,max_s` CSV rows with phases `initialize` and
`period_<k>` ([](power-users.md#timing-a-run)).

An external Kokkos Tools connector sees only four kernels by name, all reductions:
`fuse::accum`, `fuse::det_dot`, `fuse::det_sum`, `fuse::det_nsd`. The other kernels are unlabelled
and appear under generated names.

### Benchmark rows and tools

| What | Invocation | Notes |
|---|---|---|
| Scaled device benchmark | `ctest -L bench` (`bench.scaled_device_benchmark.{1,2,3}`); the binary is `fuse_bench_scaled_device_benchmark --n N --periods 5 --fields host|device [--verify] [--timing out.csv]` | Needs MPI; `.device` twins exist only with a CUDA, HIP or SYCL Kokkos and set `FUSE_EXECUTION_SPACE=device`. CI excludes the label (`-LE bench`). Refusals: `E-BENCH-PARSE`, `E-BENCH-TIMING` |
| Rank and shape sweep | `python3 tools/scale-sweep.py --driver <build>/examples/fuse_example_layout_driver --ranks 2,3,4 --shapes identical,disjoint,partial --mode strong|weak --entities 100 --steps 5 [--check] [--mpiexec mpiexec --numproc-flag -n] --out scale-sweep.csv [--dry-run]` | Output schema `mode,ranks,layout,shape,n,phase,min_s,mean_s,max_s`; `partial` needs at least 3 ranks; `--dry-run` prints the `mpiexec` lines |
| Host vs device report | `python3 tools/bench-device.py --binary <path to fuse_bench_scaled_device_benchmark> --ranks 1 2 3 --n 1000 --periods 5 --repeats 5 --max-load 2.0 --out report.md [--json raw.json] [--mpiexec ...]` | Runs host and device `repeats` times, reports median and minimum per phase as Markdown, and marks the result PRELIMINARY if `/proc/loadavg` reached `--max-load` |
| Thread-count determinism | `cmake -DBIN=<exe> -DTHREADS_HI=<n> -DOUT=<dir> [-DARGS=...] -P tools/compare-runs.cmake` | Runs at `OMP_NUM_THREADS=1` and `THREADS_HI`, strips Kokkos and environment noise, requires byte-identical stdout of at least 200 bytes. Behind `example.*.omp_identical` (`ctest -L omp`) |
| Test-cost audit | `python3 tools/check-test-cost.py [BUILD_DIR] [--threshold 10]` | Reads `Testing/Temporary/CTestCostData.txt`; exit 1 if a test at or above the threshold is missing from `cmake/FuseSlowTests.cmake`, 2 if nothing was measured |

## Variables used only by the test suite

Listed so that you can recognise them; none affects a library user.

`FUSE_TEST_PRINT_ALL_RANKS` (keep the GoogleTest printer on ranks other than 0),
`FUSE_UPDATE_GOLDEN` (regenerate golden IR and plan files), `FUSE_SWEEP_GRAPHS`,
`FUSE_BRUSSELATOR_RESTART_MPI_FIXED_PATH`, `FUSE_BRUSSELATOR_RESTART_MPI_ROLE`,
`FUSE_TRANSFER_MAP_RELOAD_MPI_FIXED_PATH`, `FUSE_TRANSFER_MAP_RELOAD_MPI_ROLE`,
`FSI_DIAG_DUMP_DIR`, `PFUSE_TEST_DATA_DIR` (Python tests), and `FUSE_MUTATE_MPI`,
`FUSE_MUTATE_PETSC`, `FUSE_MUTATE_LOGDIR` (`tools/mutate.py`).

Compile-time switches that are sometimes mistaken for environment variables: `FUSE_HAVE_MPI`,
`FUSE_HAVE_PETSC`, `FUSE_CONNECT_ACCEPT_ENABLED`, `FUSE_MPI_GPU_AWARE` and
`FUSE_FP_REPRODUCIBLE` are CMake options baked into `<fuse/base/config.hpp>`; setting them in
the shell does nothing ([](build.md)).

## See also

- [](drivers.md): exposing these settings from your own executable.
- [](power-users.md): when to use them for performance, scale and reproducibility.
- [](troubleshooting.md): the error codes and a debugging workflow.
- [](runtime.md): the Runner's stepping model in full.
