# Troubleshooting

What to do when a FUSE program refuses, produces a wrong answer or hangs. It covers a debugging
workflow, every family of error code and what to try first, the questions that come up most
often, build problems, and what to put in a bug report. [](environment.md) is the reference for
the variables named here.

## Debugging a failure

1. **Read the code.** FUSE refuses with a stable code such as `E-PLACEMENT-EMPTY` and a message
   that names what was wrong and, usually, what to change. Find the prefix in the
   [error-code table](#error-codes), then search the documentation for the full code.
2. **Validate first.** Call `fuse::validate(graph)` and print `to_string()` of the result. It
   lists every defect at once rather than the first one, and `from_yaml`/`load_yaml_file` do not
   run it for you.
3. **Print what was built.** `dump_text(dag)` and `plan.to_string()`, or `--dump` in a program
   that offers it ([](cookbook.md#see-what-fuse-built)). A channel that is missing from the plan,
   a `Remap` where you expected an identity or a `lag` you did not declare are visible here before
   anything runs.
4. **Check the counters.** `runner.stats()` after the run: `conservation_checks == 0` means no
   transfer was checked; `windows_not_converged > 0` means some answer is not converged even
   though the run returned success ([](environment.md#runstats)).
5. **Reduce.** Run on one rank, then two, with the smallest mesh that still fails. A failure only
   at scale points at placement, transport or the remap halo; a failure on one rank is in the
   graph, the components or the schedule.
6. **Trace.** Set one of the trace variables for the area the code names
   ([](environment.md#debug-and-trace-variables)). For a hang use `FUSE_COLLECTIVE_TRACE` and
   compare the last line each rank printed.
7. **Stop in a debugger.** Run with `FUSE_ON_ERROR=debugger` (or `abort` for a core file). The
   shipped examples, `import pfuse` and the C ABI's `fuse_create()` already read it; a driver of
   your own must call `fuse::initialize()` at the top of `main()`
   ([](environment.md#error-policy-variables)):

   ```cpp
   fuse::initialize();                 // reads FUSE_ON_ERROR and friends
   fuse::install_terminate_handler();
   ```

   Without that call the variable is ignored. The policy stops at *thrown* errors (assertions, invalid arguments, `FUSE_REFUSE`). A refusal
   that reached you as a `Result` with an `E-...` code was returned, not thrown, so set a
   breakpoint on the message or on the function the code names instead
   ([](environment.md#error-policy-variables)).

## Error codes

Codes are stable and prefixed by area. The library returns most of them as a failed `Result`
(`r.error().code`, `r.error().message`) or as a `Diagnostics` entry from `validate()`,
`lower()` or `schedule()`; a few are thrown as `fuse::Refusal`. Example programs add their own
prefixes, listed at the end.

| Prefix | Area | Typical cause and first thing to try |
|---|---|---|
| `E-ABI-*` | C and Fortran participant ABI | A call out of order (`fuse_create` before `load_graph`, `setup` before `run`), a component name not in the graph, or an exception inside a callback (`E-ABI-EXCEPTION`). Check the call sequence in [](c-abi.md) and the `my_component` string |
| `E-ACCEL-*` | Accelerators | Unknown kind, a configuration a kind does not accept, a non-finite or diverging iterate (`E-ACCEL-NONFINITE`, `E-ACCEL-DIVERGED`). Try `aitken` first, then tighten the components' own tolerances ([](dn-heat.md)) |
| `E-ACCUM-*`, `E-COMMGRAPH-*`, `E-CROUTER-*` | Distributed communication graph and crystal router | Component counts or owners that disagree between ranks, or a communicator that does not contain every rank. Check that all ranks built the same graph and that `base_communicator` covers every `RankSet` |
| `E-BASE-*`, `E-TIME-*` | Rationals, times and unit algebra | A `Rational` that overflows or is unreduced, an affine unit (`degC`) where a scale was needed. Put times on a fixed grid before building the graph ([](drivers.md#reproducibility-checklist)) |
| `E-CONN-*`, `E-CONSERV`, `E-SHAPE`, `E-INTENT`, `E-INTERVAL`, `E-LAG`, `E-DUP-*`, `E-COUPLER-*`, `E-CHANNEL-*`, `E-MULTI-*`, `E-SELF`, `E-MERGE-*` | Graph validation | A channel into an `Extensive` field without a conservative transfer (`E-CONN-CONSERVATION`), mismatched component counts, a `Provides` wired to a `Provides`, a non-positive interval, a duplicate name, a coupler with fields, a second writer with no `.merge(...)`, a `remap_method` string no weight builder recognises (`E-CHANNEL-REMAP-METHOD`). `validate()` prints all of them; fix the graph and re-run it |
| `E-CONNECT-*` | Multi-executable rendezvous | Segment order different from the graph's component order (`E-CONNECT-APPNUM`), acceptor not started or port file not visible (`E-CONNECT-TIMEOUT`), `disconnect()` after `MPI_Finalize` (`E-CONNECT-FINALIZED`), build without connect/accept (`E-CONNECT-ACCEPT-UNAVAILABLE`). See [What `connect()` checks](multiple-executables.md#what-connect-checks-and-what-it-refuses) |
| `E-DESC-*` | Binary graph descriptor | A restart or transfer-map file whose embedded graph descriptor is not a FUSE descriptor, is truncated or carries an unknown enum: the file was written by a different graph or a different build. Regenerate it |
| `E-EXTENT-*`, `E-FIELD-*` | Field registry and extents | A null or rebound pointer, a layout whose strides alias, a field never bound (`E-FIELD-UNBOUND`), an extent outside the region. Check `register_fields` binds every `FieldId` in `graph.component(id).fields` exactly once ([](component-interface.md)) |
| `E-GATE-*`, `E-NORM-*` | Derivative gate and residual norms | An operator whose Jacobian or transpose has the wrong shape, a norm over fields with different units, a non-finite residual. Run the derivative gate on the operator alone |
| `E-INTERP-*`, `E-MLS-*` | Interpolation and moving-least-squares remap | A target point outside the source domain, too few neighbours, a degenerate stencil, a discretisation the method does not accept. Enlarge the search radius or change the method ([](point-cloud-remap.md)) |
| `E-IR-*` | Lowering | A channel with no lowering rule for its discretisation pair (`E-IR-NORULE`), a field with several writers and no merge (`E-IR-MULTI-WRITER`), an empty region. Print `lowered.diags.to_string()` |
| `E-ITER-*` | Iteration regions | An exchanged field with no convergence measure (`E-ITER-MEASURE`), a member that cannot roll back (`E-ITER-NOROLLBACK`), a member whose restore did not reproduce its registered storage (`E-ITER-ROLLBACK`), a non-deterministic member in a region that demands determinism. Implement `save_state`/`restore_state` element-wise and declare `capabilities()` honestly ([](convergence.md)) |
| `E-LAUNCH-*` | Launch description | An MPMD component without an `appnum`, a connect component with neither `port_file` nor `service_name`, a plugin with no path ([](multiple-executables.md)) |
| `E-LINALG-*`, `E-OP-*`, `E-PETSC-*` | Linear algebra and PETSc | No convergence in CG or GMRES, a row owned by another rank, a Schur backend that is unavailable, a PETSc call failure or a shape mismatch between a PETSc operator and the Newton block. Check the operator's shape first, then preconditioning |
| `E-LOWFIVE-*` | LowFive transport | Duplicate dataset path, a field on no LowFive component, port or attach failures. See [LowFive error codes](lowfive.md#error-codes) |
| `E-MAP-*` | Transfer-map files | Checksum, magic, shape or coverage mismatch: the map was built for a different mesh pair or truncated on write. Rebuild it ([What a failed read tells you](transfer-maps.md#what-a-failed-read-tells-you)) |
| `E-MESH-*`, `E-MOAB-*` | Meshes | A mesh that cannot be read, a dimension that does not match the region, missing geometry or connectivity, a halo that crosses placements, a cell that is not star-shaped, MOAB without MPI. Check the `source` string and the backend's requirements ([](geometry.md)) |
| `E-MULTISCALE-*`, `E-SURROGATE-*` | Multiscale and surrogate components | A surrogate without a contract or an applicability check, or a feature that is not supported; the message names the missing declaration ([](multiscale.md), [](surrogates.md)) |
| `E-NEWTON-*` | Newton-Krylov coupling | A member without the residual capability (`E-NEWTON-CAPABILITY`), a linearisation token of the wrong shape (`E-NEWTON-SHAPE`), a preconditioner that does not cover the block, PETSc requested without PETSc. Start from Picard or Aitken on the same region to separate solver problems from coupling problems |
| `E-PART-*`, `E-SEARCH-*` | Partitioning and spatial search | Mixed-dimension entities, no entities on a rank, a weight array of the wrong size, no MPI in a build that needs it, a bad `FUSE_EXECUTION_SPACE` value in the search module (`E-SEARCH-SPACE`) |
| `E-PLACEMENT-*`, `E-RANKSET-*` | Rank sets | A component on no ranks where one was required (`E-PLACEMENT-EMPTY`), ranks outside the job. Check each `RankSet` against `mpiexec -n` ([](placement.md)) |
| `E-REMAP-*` | Remapping | Unsupported geometry, dimension or backend for the chosen method, a target cell no source cell covers (`E-REMAP-COVERAGE`), a degraded method where an exact one was requested. `FUSE_REMAP_HALO_TRACE` on a tiny mesh shows the weights and the per-rank boxes ([](geometry.md)) |
| `E-RESTART-*` | Restart files | A file that does not match the coupling, a descriptor hash mismatch, a feature the restart format does not carry. See [Refusals](restart.md#refusals) |
| `E-RT-*` | Runtime (Runner) | An unbound arc or component at `initialize()`, a conservation violation with `abort_on_violation` (`E-RT-CONSERVATION`), a non-converged region with `abort_on_nonconvergence` (`E-RT-NOCONVERGE`), a bad `FUSE_EXECUTION_SPACE` (`E-RT-SPACE`), `Device` on a host-only build (`E-RT-DEVICE`), a checkpoint peer that disagrees. The message names the instruction; find it in the plan |
| `E-SCHED-*` | Scheduling | An unbroken dependency cycle (`E-SCHED-CYCLE`): add `.lag(1)` to one channel or put the loop in an iteration region. A dangling channel, missing epoch or buffer plan failure points at the lowered graph; print `sched.diags.to_string()` |
| `E-ST-*`, `E-WF-*` | Space-time interpolants and waveforms | An interval or basis degree the interpolant cannot represent, a sample outside the window, a full waveform buffer. Check the channel's `spacetime` or `accumulate.window` against the component intervals ([](coupling-windows.md)) |
| `E-TRANSPORT-*` | Data movement | Unknown transport name (`E-TRANSPORT-BACKEND`), `mpi` without `transport_setup` or `base_communicator`, communicator or buffer mismatch, LowFive location errors. `FUSE_TRANSPORT_DEBUG` prints each rank's plan and the agreed outcome |
| `E-UNITS-*` | Units | Incompatible units across a channel, an affine unit where a scale is required, an undeclared unit. Declare both fields' units; FUSE inserts a scale only between compatible ones |
| `E-YAML-*` | YAML input | Syntax (`E-YAML-PARSE`), an unresolved name (`E-YAML-REF`), unknown units (`E-YAML-UNITS`), a file that cannot be opened (`E-YAML-IO`), a duplicate key. The message carries the line; see [](yaml.md) |

Prefixes that belong to example programs rather than to the library: `E-LAYOUT-*`
(the layout driver's own argument checks, [](layout-driver.md)), `E-BENCH-*` (the scaled
benchmark), `E-BRUSS-*` ([](brusselator.md)), `E-EXAMPLE-*` and `E-EX-*` (the other examples, for
instance `E-EXAMPLE-ACCEL` for an unknown `--accel` name and `E-EX-CFL` for an unstable explicit
step) and `E-PY-*` (the Python module). Their meaning is on the example's page.

## Common questions

**My run hangs.** Use `FUSE_COLLECTIVE_TRACE` and compare the last line on each rank. The usual
cause is an application MPI call that not every rank reaches, or a component that advanced on
some ranks and returned an error on others before your driver reduced the status. FUSE's own
collectives agree on failure through epochs and do not hang on their own. In an MPI driver reduce
every local `Result` to an agreed status before acting on it
([](drivers.md#mpi-agreed-exit)).

**Results change with the thread count.** Check that `FUSE_FP_REPRODUCIBLE` has not been turned
off ([](power-users.md#reproducibility)) and that every component declares and honours
`DeterminismLevel::Bitwise`; an unordered reduction inside `advance()` is the usual culprit. Run
`tools/compare-runs.cmake` on your binary to confirm.

**The coupled answer is no better than uncoupled.** Check that the channel is actually in the
plan (`--dump`). A field named differently on the two sides, or a component with an empty
`RankSet` (which switches it and its transfers off), is a common cause.

**Conservation violations.** Declare `Conservation::Extensive` on both fields, use a conservative
remap (`.conservative()` or `remap: {flags: [Conservative]}`) and `Integral` accumulation for
fluxes. `FUSE_CONSERVATION_DEBUG` prints the sums before and after each check, per rank, so you
can see which side lost mass.

**An iteration region will not converge.** Try a different accelerator ([](dn-heat.md)),
tighten or change the measure ([](convergence.md)), and check that every member restores its
state exactly: `checkpoint_verifications` in `RunStats` counts the comparisons FUSE made, and
`E-ITER-ROLLBACK` is what a failed one looks like. With `abort_on_nonconvergence = false` the
run still succeeds; read `windows_not_converged`.

**Fields appear stale.** A component swapped or reallocated a registered array. Write in place
and restore checkpoints element-wise ([](quickstart.md), [](component-interface.md)).

**`--kokkos-*` flags are rejected.** The program parses its arguments before constructing
`Kokkos::ScopeGuard`. Construct the guard first ([](drivers.md)); among the examples only
`brusselator` has this order.

**`FUSE_ON_ERROR=debugger` does nothing.** Either the program is your own driver and never called
`fuse::initialize()` (add it to `main()`, see [Debugging a failure](#debugging-a-failure)), or the
error was not thrown: a refusal that comes back as a `Result` is returned, not thrown, and the
policy never sees it.

**`E-RT-DEVICE` or `E-RT-SPACE` at `initialize()`.** `FUSE_EXECUTION_SPACE` is set in the
environment, perhaps left over from another run. Unset it, or build against a Kokkos with a
device backend.

**A remap refuses with `E-REMAP-COVERAGE` only on several ranks.** The target has cells no
source cell on any rank covers after partitioning. Run with `FUSE_REMAP_HALO_TRACE` on a small
mesh and compare the per-rank source and target bounding boxes.

**The plan is enormous.** The coupling period is the least common multiple of the component
intervals; `1/3` and `1/7` give a 21-step plan per period. Use commensurate steps or a coupling
window ([](coupling-windows.md)).

## Build problems

| Symptom | Fix |
|---|---|
| Configure cannot fetch a dependency | The host blocks the download. Install it and pass `-DCMAKE_PREFIX_PATH`, or for Eigen use `-DFUSE_LINALG_BACKEND=builtin` |
| `Kokkos` not found | Install it and set `CMAKE_PREFIX_PATH`, or add `-DFUSE_ALLOW_FETCH=ON` |
| A feature option does nothing | It needs its dependency: MOAB needs MPI, the connect/accept launch mode needs an MPI whose dynamic-process support passes the configure probe ([](build.md)) |
| Some examples or tests are missing | The builtin backend skips examples that need Eigen; `layout_driver`, `demo_b` and the `bench` rows need MPI; `.device` rows need a device Kokkos |
| Two Kokkos installations | A test compares `FUSE_KOKKOS_VERSION_EXPECTED` with the headers' `KOKKOS_VERSION`; a mismatch means the include path and the linker picked different installs. Pass one `CMAKE_PREFIX_PATH` |
| Docs build fails | Warnings are errors; fix the warning it prints |

## What to include in a bug report

Open an issue at <https://github.com/vijaysm/fuse/issues> with:

1. The full error line: code and message, and the rank it came from.
2. The graph: the YAML (`fuse::to_yaml(g)` if you build it in code) and the `--dump` output.
3. `fuse-resolved-dependencies.txt` from the build directory, and the CMake options you passed.
4. The launch line (`mpiexec -n ...`, `OMP_NUM_THREADS`, `FUSE_EXECUTION_SPACE`, any `--kokkos-*`
   flags) and the smallest rank count at which it reproduces.
5. `runner.stats()` if the run got that far, and the relevant trace output
   ([](environment.md#debug-and-trace-variables)) from the failing rank.

## See also

- [](environment.md): every variable and configuration field.
- [](power-users.md): performance, scale and reproducibility.
- [](drivers.md): writing a driver that reports failures well.
- [](cookbook.md): recipes for the common tasks.
