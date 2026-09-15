# Gray-Scott Coupler

The Gray-Scott prototype is a two-component MPI coupling example. Component U advances the `u` species on one 2-D finite-element mesh; component V advances `v` on a different nonmatching mesh. The components exchange the opposite species through distributed continuous-Galerkin projection. Serial and 2-rank replicated-SPMD executions of the complete scheduler are verified, and independently partitioned U/V trajectories pass stable-`DofKey` serial-reference comparisons at 2/3/4 ranks. The example remains a qualitative demonstration rather than a production pattern-certification workflow.

## Algorithm

The coupled equations are

\[
u_t-D_u\Delta u=-uv^2+F(1-u),
\qquad
v_t-D_v\Delta v=uv^2-(F+k)v.
\]

Each component freezes its incoming transferred field during a substep and solves a Crank-Nicolson nonlinear reaction-diffusion problem with PETSc SNES. The scheduler compiles one of two instruction sequences:

- **Lie:** exchange `v -> u`, advance U for `dt`, exchange `u -> v`, advance V for `dt`.
- **Strang ABA:** exchange `v -> u`, advance U for `dt/2`, exchange `u -> v`, advance V for `dt`, exchange `v -> u`, advance U for `dt/2`.

Every macro step has an in-memory checkpoint. Transfers write edge-local staging vectors. Component states become committed only after every instruction succeeds. A failure is reconciled across MPI ranks before any rank enters a later collective; both components and the scheduler position are restored.

## Build

The example target is optional:

```bash
cmake -S . -B build-gray-scott \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH='/opt/petsc/download-moab;/opt/eigen3' \
  -DKARMA_ENABLE_EXAMPLES=ON
cmake --build build-gray-scott --target karma_gray_scott_operator_split -j4

./build-gray-scott/examples/karma_gray_scott_operator_split
mpiexec -n 2 ./build-gray-scott/examples/karma_gray_scott_operator_split
```

The spatial demonstration defaults to generated nonmatching `16x16` (U) and
`19x17` (V) meshes, `dt=0.01`, final time `0.1`, and Strang splitting. It uses
nonzero diffusion and a deterministic circular perturbation centered at
`(0.5,0.5)`. A compact smoke run is:

```bash
./build-gray-scott/examples/karma_gray_scott_operator_split \
  -gs_scheme strang -gs_u_resolution 6,6 -gs_v_resolution 7,5 \
  -gs_dt 0.002 -gs_final_time 0.01 \
  -gs_diffusion_u 0.16 -gs_diffusion_v 0.08 \
  -gs_feed 0.060 -gs_kill 0.062 \
  -gs_perturbation 0.05 -gs_perturbation_radius 0.12 \
  -gs_output_dir gray-scott-output -gs_checkpoint_cadence 5
```

Use `-gs_scheme lie` for Lie splitting. Replace either generated resolution
with `-gs_u_mesh file.h5m` or `-gs_v_mesh file.h5m`; mesh files supplied to an
MPI launch must already be suitable for the selected communicator/partitioning.
`-gs_checkpoint_cadence 0` disables output. Final time must be an integer
multiple of the timestep. PETSc solver options remain available under the
existing `-karma_transient_reaction_diffusion_` and `-karma_cg_transfer_`
prefixes. Run with `-help` for the application option summary.

MPI launches require the complete effective option set, including help mode,
mesh paths, output path, and numerical parameters, to agree on every rank. The
driver parses locally, reconciles a rank-qualified parse failure, and compares
canonical option signatures before constructing either mesh. It then checks
mesh-file readability and output-directory creation/write access on every rank
in separate collective preflight epochs. Thus an MPMD launch with differing
arguments or a path visible to only some ranks is rejected before any peer
enters mesh, scheduler, or checkpoint collectives.

Every committed step prints extrema, Euclidean norms, consistent FE masses,
SNES reasons, transfer residual norms and quadrature coverage, and step/solve
timings. The final line reports transfer KSP reasons and accumulated transfer
setup/apply and wall timings. Positive PETSc reason integers denote successful
convergence; residual acceptance remains scale-aware in the library.

## Verification Modes

The tests separate numerical effects:

- The temporal Lie/Strang gate uses homogeneous fields and an explicit exact-constant exchange hook, isolating schedule order from multimesh projection error.
- The real connection tests use nonmatching meshes and `KARMAFEMTransfer::project_cg` in both directions.
- The transient assembly test checks Crank-Nicolson residual and Jacobian factors independently, including nonzero diffusion.
- Rollback tests inject exchange, advance, and commit failures.
- The partitioned projection regression batches all target quadrature points before collective source evaluation, preventing rank-asymmetric collective deadlock.

Measured temporal acceptance is approximately first order for Lie and second order for Strang in the exact-constant exchange isolation gate. Coupled nonmatching spatial MMS is also verified on the tested refinement sequence, with approximately second-order U, V, and combined rates. These are regression/oracle results; they do not establish positivity, morphology, parameter selection, or production mesh/time convergence.

## MPI Meshes

Both components use `PETSC_COMM_WORLD`. The current complete example generates a different mesh for each component on every rank and therefore exercises replicated SPMD with nonmatching meshes. The lower-level transfer suite separately partitions source and target H5M fixtures with `mbpart`; stable DoF identities, not PETSc indices or MOAB handles, are used for those comparisons and persistence.

A representative existing fixture workflow is:

```bash
ctest --test-dir build-gray-scott -R 'karma_fem_transfer_(partition|verify|project_cg)' \
  --output-on-failure -j1
```

MPI tests have finite timeouts. If `mbpart` is unavailable, the partitioned fixture tests are not registered and that coverage must be reported as unavailable rather than inferred.

## Safe-point restart

The native transient adapters support a public three-file restart bundle at a committed macro-step boundary. The U and V files are exact H5M checkpoints. The versioned metadata file records the canonical compiled-graph signature, split scheme, committed scheduler time and step, and digests of both H5M files.

```cpp
#include "karma_operator_split_checkpoint.h"

KARMA::OperatorSplitCheckpointBundle files{
    "gray-scott-U.h5m", "gray-scott-V.h5m", "gray-scott-scheduler.karma"};

// Call only after scheduler.advance() has committed the macro step.
KARMA::write_operator_split_checkpoint_bundle(
    files, compiled_graph, {&u_adapter, &v_adapter}, scheduler.position());

// Fresh adapters must be initialized before validation. No live state changes
// until both H5M files and all scheduler metadata have passed validation.
restarted_u.initialize();
restarted_v.initialize();
auto restart = KARMA::prepare_operator_split_restart(
    files, compiled_graph, {&restarted_u, &restarted_v});
restart.apply();
KARMA::KARMACouplingScheduler resumed(
    compiled_graph, {&restarted_u, &restarted_v}, restart.position());
```

`prepare_operator_split_restart()` rejects missing files, changed file contents, incompatible metadata versions, graph signatures, split schemes, scheduler positions, field layouts, stable DoF-key sets, and canonical mesh geometry/topology before mutating either live component. Checkpoint coefficients are associated with the destination through `DofKey`, never by PETSc ordinal. `apply()` snapshots both live states and reconciles each publication epoch collectively; failure while publishing either component rolls both components back before reporting one deterministic rank-qualified diagnostic. The scheduler constructor taking `SchedulerPosition` resumes the next interval at the persisted committed time and step.

This bundle is **not** an atomic multi-file durable transaction. Each H5M checkpoint is published independently and the metadata file is published last. A crash can therefore leave orphaned component files or an older metadata file. Loading detects incomplete or mixed bundles through metadata and file digests and rejects them; applications that need graph-wide transactional durability must provide an external generation/rename protocol.

## Inspecting output in ParaView

Each output event creates `step-N-U.h5m`, `step-N-V.h5m`, and
`step-N.karma`. The metadata file makes the two H5M files a validated safe-point
bundle; it is not a visualization file. Open each H5M component in a ParaView
build with the MOAB reader plugin, select the transient state field, and use
**Surface** plus **Color By** to inspect it. The U and V meshes are independent,
so they should be loaded as separate sources; comparisons may use linked views
or ParaView resampling, but such resampling is visualization-only and is not
the coupler's cG transfer. If the ParaView installation lacks a MOAB/H5M reader,
convert copies with a MOAB-aware tool; do not rename `.h5m` to `.vtk`.

## Limitations

This prototype is intentionally not the complete plugin architecture. It does not provide dynamic plugins, Python/Fortran adapters, MPMD/disjoint communicators, arbitrary graph topology or fan-in, Anderson acceleration, monolithic Newton, dynamic graph mutation, or an atomic durable transaction spanning the restart bundle. The native fixed-point path is limited to the scheduler's two reciprocal components, uses consistent-mass RMS metrics with constant or safeguarded Aitken relaxation, rejects custom exchange hooks, and does not claim general graph fixed-point support. Its setup, sweep, staging, and commit failures are collectively reconciled; recoverable post-checkpoint failures restore the macro-step state. Initial-condition callbacks are evaluated and reconciled before a private vector is assembled, so thrown or nonfinite rank-local results leave committed and trial state unchanged. The example is a qualitative spatial-dynamics demonstration: it does not enforce positivity, establish a morphology oracle, prove parameter-dependent pattern selection, or claim production mesh/time convergence. Norms, extrema, masses, residuals, coverage, and solver reasons are diagnostics rather than validation of a particular pattern.
