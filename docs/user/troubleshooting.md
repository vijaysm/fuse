# Troubleshooting

## Debugging a failure

1. **Read the code.** FUSE refuses with a stable code such as `E-PLACEMENT-EMPTY`. The message names
   what was wrong and, usually, what to change. Search this page, then the documentation, for the
   code.
2. **Validate first.** Call `fuse::validate(graph)` and print `to_string()`. It lists every defect at
   once.
3. **Print what was built.** `dump_text(dag)` and `plan.to_string()` ([](cookbook.md#see-what-fuse-built)).
4. **Reduce.** Run on one rank, then two. A failure only at scale points at placement or transport.
5. **Trace.** Set `FUSE_ON_ERROR=debugger` to stop at the failing rank, or one of the trace variables
   in [](power-users.md#diagnostics). For a hang use `FUSE_COLLECTIVE_TRACE`.

## Error codes

Codes are stable and prefixed by area.

| Prefix | Area | Usual cause |
|---|---|---|
| `E-PLACEMENT-*` | Rank sets | A component on no ranks, or ranks outside the job |
| `E-ITER-*` | Iteration | An exchanged field with no convergence measure |
| `E-SCHED-*` | Scheduling | An unbroken cycle, or steps with no common period |
| `E-REMAP-*` | Remapping | Unsupported geometry or dimension for the chosen remap |
| `E-MESH-*` | Meshes | A mesh that cannot be read, or whose entities moved |
| `E-YAML-*` | YAML input | A syntax error or an unresolved reference |
| `E-TRANSPORT-*` | Data movement | Communicator or buffer mismatch |
| `E-RESTART-*` | Restart | A file that does not match the coupling ([](restart.md#refusals)) |
| `E-CONNECT-*` | Multi-executable | A launch description that does not match ([](multiple-executables.md)) |
| `E-NEWTON-*` | Newton-Krylov | A region the solver cannot handle |

## Common questions

**My run hangs.** Use `FUSE_COLLECTIVE_TRACE`. The usual cause is an application MPI call that not
every rank reaches; FUSE's own collectives agree on failure and do not hang on their own.

**Results change with the thread count.** Check that `FUSE_FP_REPRODUCIBLE` has not been turned off
([](power-users.md#reproducibility)).

**The coupled answer is no better than uncoupled.** Check that the channel is actually in the plan
(`--dump`). A field named differently on the two sides is a common cause.

**Conservation violations.** Declare `Conservation::Extensive` on both fields, use `.conservative()`
remap and `Integral` accumulation for fluxes. `FUSE_CONSERVATION_DEBUG` shows the offending values.

**An iteration region will not converge.** Try a different accelerator ([](dn-heat.md)), tighten or
change the measure ([](convergence.md)), and check that every member restores its state exactly.

**Fields appear stale.** A component swapped or reallocated a registered array. Write in place
([](quickstart.md)).

## Build problems

| Symptom | Fix |
|---|---|
| Configure cannot fetch a dependency | The host blocks the download. Install it and pass `-DCMAKE_PREFIX_PATH`, or for Eigen use `-DFUSE_LINALG_BACKEND=builtin` |
| `Kokkos` not found | Install it and set `CMAKE_PREFIX_PATH`, or add `-DFUSE_ALLOW_FETCH=ON` |
| A feature option does nothing | It needs its dependency: MOAB needs MPI ([](build.md)) |
| Some examples are missing | The builtin backend skips examples that need Eigen |
| Docs build fails | Warnings are errors; fix the warning it prints |

If you still cannot resolve it, open an issue at <https://github.com/vijaysm/fuse/issues> with the
failing code, the `dump_text` output and `fuse-resolved-dependencies.txt`.
