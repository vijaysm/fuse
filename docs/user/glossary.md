# Glossary

This page defines the terms used in this documentation, and gives for each one the term used in the
scientific-computing literature. The literature term is the one to use in prose. Where the code still
uses an older identifier, the table lists it in the first column so that you can map what you read in
a header or an error message to the concept. Identifiers are renamed to the literature terms in a
single pass later; until then, this page is the translation.

References are listed at the end; a bracketed number points into that list.

## Components and fields

| Identifier in the code | Term | Definition |
|---|---|---|
| `Component`, `ComponentDecl` | component | An independently written solver or model (a fluid code, a structural code, an emulator) that advances its own state in time and exchanges fields with other components [6, 8]. |
| `Intent::Provides` / `Intent::Requires` | export field / import field | A field a component makes available to others, or reads from others. The ESMF terms export state and import state [10] name the same thing. |
| `Region` (of a field) | mesh entity set | The set of mesh entities of one dimension (cells, faces or vertices) on which a field's values are stored. Not related to the implicit coupling group below. |
| `Discretization`, nodal / cell | vertex-centred / cell-centred field | Where the degrees of freedom sit. |
| `SemanticRole::Prognostic`, `Diagnostic` | prognostic / diagnostic variable | A variable advanced in time by the component, or one computed from the prognostic state. |
| `SemanticOwnership::Owned`, `Ghosted`, `Replicated` | owned / ghost / replicated entity | Standard domain-decomposition terms: an entity whose value this process is responsible for, a copy of a neighbour's owned entity, or an entity every process holds. |
| `capabilities()`: rollback, determinism level, hidden state | checkpoint–restart support, reproducibility level, unregistered state | Whether a component can restore its state to the start of a step (needed for iterative coupling), whether its results are bit-reproducible, and whether it carries state not exposed to the coupler. |
| `ImplicitComponent::evaluate_residual()` | residual | A component's own contribution to the coupled residual at a frozen time level, given the interface values it imports, without advancing its state [4, 5]. The quantity a Newton iteration drives to zero, as opposed to `Residual` (below), the *coupling* residual `r = x̃ − x` a fixed-point iteration measures. |
| `LinearizationToken` | linearisation | A component's own local approximation of how its residual changes with its interface inputs, about one evaluation point — what a Newton step needs from each participant instead of the whole coupled problem's exact Jacobian. |

## Coupling specification

| Identifier in the code | Term | Definition |
|---|---|---|
| `CouplingGraph` | coupling specification | The declaration of the components, their fields and the transfers between them. |
| `Channel`, "channel" | field transfer | One directed transfer of one export field of a source component to one import field of a target component, together with its time treatment, remapping and unit conversion. OASIS calls this a coupling field [9]. |
| "several writers of one field", `Merge` | merging | Combining several transfers into one import field, for example area-fraction-weighted sums over land, ocean and sea ice. This is the CESM coupler's term [7]. |
| lag, `TimeLevel` shift | lagged (staggered) coupling | The target uses the source value from the previous coupling step, which makes a cyclic dependence explicit instead of iterative [6, 8]. |
| `IterationRegionDecl`, "iteration region" | implicit coupling group | A set of components whose coupling within a step is solved to convergence by fixed-point (sub-)iterations. This is strong or implicit partitioned coupling, as opposed to explicit or loose coupling [4, 6, 8]. |
| "participant" (of an iteration region) | processes of the implicit coupling group | Every MPI process that holds part of a component in the group, or part of the coupler processor set the group's transfers pass through. |
| `Relax` | under-relaxation | x ← ω x̃ + (1 − ω) x, applied to the interface value between iterations [4]. |
| `Residual` | interface residual | r = x̃ − x, the change of the interface value over one fixed-point iteration, which the convergence criterion measures [4, 5]. |
| `Accelerate`, `AccelKind::Aitken`, IQN-ILS | convergence acceleration | Aitken's dynamic relaxation [4], or interface quasi-Newton with inverse least squares [5]. |
| `Constrain` | conservation check | A check that a transferred integral is preserved to a stated tolerance; it reports, and aborts only when configured to. |
| `Diagnose` | diagnostic output | Values computed for inspection only; they do not feed back into the coupling. |

## Time

| Identifier in the code | Term | Definition |
|---|---|---|
| `interval`, "tick" | time step | One step of a component, of length Δt, which the component chooses. |
| "clock" | time grid | The sequence of a component's step times. |
| "multirate", "two clocks" | multirate coupling | Components on different time steps [11]. |
| "period" | coupling cycle | The shortest time after which the whole pattern of steps and transfers repeats. For commensurate steps this is the least common multiple of the time steps. Time windows (below) are how components with incommensurate steps are coupled without it. |
| "lockstep" | equal-step coupling | Every component uses the same time step. |
| "slab" | interval of the common refinement | The union of two components' time grids splits time into intervals that contain no step boundary of either; accumulation over a target step is a sum over these intervals. |
| `Sample` | point evaluation in time | The source value at one instant. |
| `Accumulate` | time integration (time averaging) | The integral, or mean, of a source quantity over the target's step. Used for fluxes, so that what one component loses the other gains over the same interval. |
| `Interpolate` / `Extrapolate` | temporal interpolation / extrapolation | Reconstruction of a source value at a time between, or beyond, its known values. Extrapolation is the predictor of a predictor–corrector coupling. |
| `CouplingWindowScope`, "coupling window", "window-scoped region" | time window of a waveform iteration | A time interval over which an implicit coupling group of components with different time steps is iterated as a whole: each component advances across the whole window against its partners' interface values from the previous iteration, and the iteration repeats until they agree. This is waveform relaxation [1, 2], and in quasi-Newton form waveform iteration [3]. |
| "trajectory", K samples | waveform | The interface data of one component as a function of time over one time window, represented by its values at the component's step times within the window and a temporal interpolant between them. |
| "substeps", `WindowAdvance` | subcycling | Taking several of a component's own time steps inside one time window [8, 11]. |
| "straddling step" | step not aligned with the window end | A time step that begins inside a time window and ends after it. |
| `DtPolicy::Adjustable` / `DtPolicy::Fixed` | step adapted to the window / fixed step | The component shortens its last step to end on the window boundary, or keeps its step and the coupler handles the step not aligned with the window end. |
| `E-RT-WINDOW-STRADDLE`, `E-RT-WINDOW-LONGSTEP`, "straddle envelope" | supported configurations of time windows | Configurations outside the supported set are rejected with these error codes rather than computed. |

## Space: meshes and remapping

| Identifier in the code | Term | Definition |
|---|---|---|
| `Remap` | remapping | Transfer of a field between two non-matching meshes by a sparse linear map (the weight matrix) [12, 13]. |
| "conservative" remap | conservative remapping | The integral of the field over the domain is preserved, by integrating over the common refinement (supermesh) of the two meshes [12, 13]. |
| "consistent" remap, `consistent-transpose` | consistent interpolation, and its transpose | Target values interpolated from the source; the transpose maps target loads back to source nodes and conserves their sum, the usual pairing in fluid–structure interaction. |
| MLS remap | moving least-squares interpolation | Meshless interpolation from a point cloud. |
| supermesh, `intx` | common refinement (intersection mesh) | The mesh of all intersections of source and target cells [12, 13]. |
| `RemapPlan`, "weights" | remapping weight matrix | The sparse matrix that applies the remap. |

## Parallel layout

| Identifier in the code | Term | Definition |
|---|---|---|
| `RankSet`, "placement", "PE range" | processor set (processor layout) | The set of MPI processes a component runs on. E3SM calls the assignment of processor sets to components the PE layout [14]. |
| disjoint / partially overlapping / fully overlapping placements | disjoint, partially overlapping and coincident processor sets | Whether two components share processes. |
| `placed_view`, "per-placement decomposition" | a component's own partition of the mesh | The distribution of a mesh's entities over one component's processor set, which can differ from another component's partition of the same mesh. |
| `ComponentKind::Coupler`, "coupler placement" | coupler processor set | A processor set on which remapping is performed, separate from the processors of either component. This is the hub-and-spoke arrangement of the CESM/E3SM coupler [7, 14]. |
| `RemapSpec::remap_on` | remapping processor set | The processor set a transfer is remapped on: the coupler processor set, or by default the target's. |
| `Transport` | data redistribution | Moving a distributed field from one processor set's partition to another's. MCT calls this a rearranger [15], and MOAB builds a communication graph for it [14]. |
| `CommGraph` | communication graph | For a pair of partitions, which process sends which entities to which process [14]. |
| "hop elided" | redistribution omitted | No data is moved when the source and destination processor sets and partitions coincide. |
| halo | halo (ghost layer) | Copies of neighbouring processes' entities kept locally so that a local computation can read them. |
| `Epoch`, "rank-agreement point" | synchronisation point with error agreement | A point every process of a transfer reaches in the same order, at which all of them learn whether any failed. A failure on one process then stops all of them instead of leaving the rest blocked in the next communication. |
| "layout driver", `--coupler e:f` | processor-layout test driver | The example that runs a coupling specification on processor sets given on the command line. |

## Execution

| Identifier in the code | Term | Definition |
|---|---|---|
| "lowering", `lower()` | compilation into a dataflow graph | Translation of the coupling specification into a directed graph of the operators above (`Sample`, `Accumulate`, `Remap`, `Transport`, …) connected by the values they pass. |
| IR, `OperatorDAG` | dataflow graph (intermediate representation) | That graph. It is printed in a fixed text format, which the reference outputs record. |
| "arc" | edge of the dataflow graph | One value passed from one operator to another. |
| `Source` / `Sink` | read from / write into a component | The operators at which the graph reads a component's export field and writes a component's import field. |
| `Scale` | unit conversion | Multiplication by an exact rational factor. |
| `schedule()`, `ExecutionPlan` | static schedule | A fixed linear order of every operator instance over one coupling cycle, identical on every process. |
| "instant", `TickPhase` | time level, and before/after the component steps | Operators at the same time are ordered relative to the component steps that begin at that time. |
| `Runner` | runtime | The part of FUSE that executes the schedule. |
| descriptor, "narrow waist" | serialised coupling specification | The versioned binary encoding of the coupling specification shared by the C++, C/Fortran and YAML front ends. |

## Linear algebra

| Identifier in the code | Term | Definition |
|---|---|---|
| `fuse::linalg::DistVector` | row-distributed dense vector | A dense vector split across participants by row, each holding an owned block in its own local order plus a ghost region — copies of entries other participants own, refreshed on request. |
| `fuse::linalg::DistSparseMatrix` | row-distributed sparse matrix | A sparse matrix split across participants by whole rows, each row held as a diagonal block (this participant's own columns) and an off-process block (every other column). |
| `fuse::linalg::Reducer`, `GhostExchanger` | global reduction, ghost update | The two communication facades `DistVector`/`DistSparseMatrix` ask of communication: a global reduction over owned entries, and a one-shot exchange refreshing a ghost region along a pattern built once. |
| `fuse::linalg::LinearOperator`, `MatrixFreeOperator` | matrix, Jacobian-free operator | The structural (duck-typed) concept a Krylov solver needs from "a matrix": `apply(x, y)` computing `y = A x`. A `MatrixFreeOperator` wraps a caller-supplied callable computing a Jacobian-vector product without ever assembling the Jacobian. |
| `LinearizationToken::apply_jacobian()` | Jacobian action | The product `J x` of a residual's Jacobian with a vector, computed without ever forming `J` itself — one component's own contribution to a matrix-free (Jacobian-free) Newton-Krylov step. |
| `LinearizationToken::apply_preconditioner()` | physics-based preconditioner | A component's own approximate inverse of its diagonal block, built from the physics it knows (a lagged operator, a multigrid cycle, a lower-fidelity model) rather than from the matrix's entries alone — the kind of preconditioner a generic algebraic one (such as the Jacobi preconditioner, below) cannot construct. |
| `fuse::linalg::pcg` | preconditioned conjugate gradient (PCG) | An iterative solver for a symmetric positive-definite system, right-preconditioned by the Hestenes–Stiefel recurrence [17]. |
| `fuse::linalg::gmres` | restarted GMRES, GMRES(*m*) | An iterative solver for a general (non-symmetric) system by minimising the residual over a Krylov subspace, restarted every *m* steps [18]. |
| `fuse::linalg::Preconditioner`, `JacobiPreconditioner` | preconditioner, Jacobi (diagonal) preconditioner | An approximate solve, `apply(x, y)` computing `y = M⁻¹x`, substituted for the exact inverse to accelerate a Krylov iteration. The Jacobi preconditioner uses the matrix's own diagonal. |
| `fuse::linalg::SchurSlabSolver` | Schur-complement direct solver | An exact distributed direct solver for a banded system, by eliminating each participant's interior unknowns locally and solving the much smaller Schur complement system over the interface unknowns alone [19]. |
| "row band" (`fuse::native::rank_slice`) | subdomain | One participant's contiguous range of rows of a row-partitioned grid — the domain-decomposition literature's term is *subdomain* [19]; this project keeps "row band" in code and prose to avoid colliding with "slab", the unrelated time-domain term above. |

## Element types (restart typed fields)

| Identifier in the code | Term | Definition |
|---|---|---|
| `fuse::ElementType` | element type | A closed enumeration naming the scalar storage format a restart field is written/read as: `Int8`..`Int64`, `UInt8`..`UInt64`, `Binary16`, `BFloat16`, `Binary32`, `Binary64`, `LongDouble`, `Bool`. Stored in the restart file as per-field metadata (`element_type`, `byte_size`, `significand_bits`, `exponent_bits`, `exponent_bias`, `byte_order`, `is_signed`), never inferred from the reading process's own type. |
| `Binary16`, `Binary32`, `Binary64` | binary16 / binary32 / binary64 | The IEEE 754-2019 names [20] for the formats C++ calls `_Float16`/`float`/`double`: 1 sign bit, an implicit leading significand bit, and 5/8/11 exponent bits respectively (`Binary16`: 10 explicit significand bits; `Binary32`: 23; `Binary64`: 52). |
| `fuse::bfloat16` | bfloat16 | A FUSE-defined storage type (there is no standard C++ type for it on every platform): 1 sign bit, 8 exponent bits (the same width and bias as `Binary32`), 7 explicit significand bits. `to_float()` is always an exact widening; `from_float()` rounds to nearest, ties to even, and preserves quiet-NaN payloads. |
| `fuse::ElementType::LongDouble` | long double / x86 extended precision | The platform's native `long double`. On Apple Silicon (this project's primary development target) it is bit-identical to `Binary64` (53-bit significand); on x86-64 it is typically the 80-bit extended format with an EXPLICIT (not implicit) leading significand bit -- HDF5 calls this `H5T_NORM_MSBSET` as opposed to `Binary16`/`32`/`64`'s `H5T_NORM_IMPLIED`. A restart field written as `LongDouble` on one platform and read on another goes through the same `ReadConversion` machinery as any other type pair; it is never assumed to match. |
| `fuse::ReadConversion::Exact`, `Widening`, `AllowNarrowing` | read-conversion mode | The caller's declared tolerance for a stored field's element type differing from the type it is read back as. `Exact` (the default) refuses by name on any mismatch. `Widening` additionally permits a conversion that is proven lossless from the two types' own metadata (significand/exponent coverage for floats, bit-width/signedness rules for integers) -- a float narrowing or a signed-to-unsigned conversion is never "widening" and always refuses. `AllowNarrowing` additionally permits a conversion that may lose information, and reports the largest change actually measured over the values read (`fuse::ReadConversionReport::max_absolute_change`); integer overflow refuses (`E-RESTART-INTEGER-OVERFLOW`) even under `AllowNarrowing` -- this mode bounds silent error, it does not disable the overflow check. |

## Verification and testing

| Identifier in the code | Term | Definition |
|---|---|---|
| "tier 0 / 1 / 2" | build configuration | Tier 0 is serial, tier 1 adds MPI, tier 2 adds MOAB. |
| "golden" files | reference outputs | Files a regression test compares against byte for byte. |
| "refusal", "refuses", `E-…` codes | rejected configuration, with an error code | An input FUSE declines to compute rather than compute incorrectly. A rejection counts as tested only when a test triggers it. |
| mutation, `KILLED`, `SURVIVED`, `NOT-ASSESSED` | mutation testing | A deliberate defect is put into the code; a test suite that then fails has *killed* it, one that still passes lets it *survive*, and a mutant that cannot be run in a given build configuration is *not assessed* [16]. |
| "five states" | verification status | Designed, implemented, implemented and untested, tested, verified: how far a feature's claims are supported by evidence. |
| MMS | method of manufactured solutions | Verification against an exact solution constructed by adding a source term. |

## References

1. E. Lelarasmee, A. E. Ruehli and A. L. Sangiovanni-Vincentelli. The waveform relaxation method for time-domain analysis of large scale integrated circuits. *IEEE Trans. Computer-Aided Design* 1(3):131–145, 1982.
2. M. J. Gander. 50 years of time parallel time integration. In *Multiple Shooting and Time Domain Decomposition Methods*, Springer, 69–113, 2015.
3. B. Rüth, B. Uekermann, M. Mehl, P. Birken, A. Monge and H.-J. Bungartz. Quasi-Newton waveform iteration for partitioned surface-coupled multiphysics applications. *Int. J. Numer. Methods Eng.* 122(19):5236–5257, 2021.
4. U. Küttler and W. A. Wall. Fixed-point fluid–structure interaction solvers with dynamic relaxation. *Comput. Mech.* 43:61–72, 2008.
5. J. Degroote, K.-J. Bathe and J. Vierendeels. Performance of a new partitioned procedure versus a monolithic procedure in fluid–structure interaction. *Comput. Struct.* 87:793–801, 2009.
6. D. E. Keyes et al. Multiphysics simulations: challenges and opportunities. *Int. J. High Perform. Comput. Appl.* 27(1):4–83, 2013.
7. A. P. Craig, M. Vertenstein and R. Jacob. A new flexible coupler for earth system modeling developed for CCSM4 and CESM1. *Int. J. High Perform. Comput. Appl.* 26(1):31–42, 2012.
8. C. A. Felippa, K. C. Park and C. Farhat. Partitioned analysis of coupled mechanical systems. *Comput. Methods Appl. Mech. Eng.* 190:3247–3270, 2001.
9. S. Valcke. The OASIS3 coupler: a European climate modelling community software. *Geosci. Model Dev.* 6:373–388, 2013.
10. C. Hill, C. DeLuca, V. Balaji, M. Suarez and A. da Silva. The architecture of the Earth System Modeling Framework. *Comput. Sci. Eng.* 6(1):18–28, 2004.
11. C. W. Gear and D. R. Wells. Multirate linear multistep methods. *BIT* 24:484–502, 1984.
12. P. W. Jones. First- and second-order conservative remapping schemes for grids in spherical coordinates. *Mon. Weather Rev.* 127:2204–2210, 1999.
13. P. A. Ullrich and M. A. Taylor. Arbitrary-order conservative and consistent remapping and a theory of linear maps: Part I. *Mon. Weather Rev.* 143:2419–2440, 2015.
14. V. S. Mahadevan, I. Grindeanu, R. Jacob and J. Sarich. Improving climate model coupling through a complete mesh representation: a case study with E3SM (v1) and MOAB (v5.x). *Geosci. Model Dev.* 13:2355–2377, 2020.
15. J. Larson, R. Jacob and E. Ong. The Model Coupling Toolkit: a new Fortran90 toolkit for building multiphysics parallel coupled models. *Int. J. High Perform. Comput. Appl.* 19(3):277–292, 2005.
16. R. A. DeMillo, R. J. Lipton and F. G. Sayward. Hints on test data selection: help for the practicing programmer. *Computer* 11(4):34–41, 1978.
17. M. R. Hestenes and E. Stiefel. Methods of conjugate gradients for solving linear systems. *J. Res. Natl. Bur. Stand.* 49(6):409–436, 1952.
18. Y. Saad and M. H. Schultz. GMRES: a generalized minimal residual algorithm for solving nonsymmetric linear systems. *SIAM J. Sci. Stat. Comput.* 7(3):856–869, 1986.
19. B. Smith, P. Bjørstad and W. Gropp. *Domain Decomposition: Parallel Multilevel Methods for Elliptic Partial Differential Equations*. Cambridge University Press, 1996.
20. IEEE Std 754-2019 (Revision of IEEE Std 754-2008). *IEEE Standard for Floating-Point Arithmetic*. IEEE, 2019.
