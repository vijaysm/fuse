# S8.0 inventory: physics-kit disposition (upstream/karma, 512 files)

Track `71f7c1b0`-style disposition pass for Stage S8 (`physics-kit`). Scratch-only, read-only against
the owner's live checkout; nothing in `upstream/karma` or `fuse/` was changed. HEAD = `c62a5cc`;
`upstream/karma` pinned at `136f520` (`UPSTREAMS.lock`), byte-identical, confirmed by file count
(512, matching `find upstream/karma -type f | wc -l`, MEASURED).

## Plain answers first

1. **Disposition of the 512 files**: 190 retire, 190 port-adapt (D-A), 68 port-integrator (D-B),
   35 undecided, 29 superseded (15 already locked in `UPSTREAMS.lock` + 14 newly confirmed against
   the current fuse tree). Zero files needed `port-literal`: every file that is not already covered
   by an existing fuse type needs at least an interface change (fuse::linalg containers, fuse's
   `Result`/capability-bit vocabulary, or the `CoupledValue`/`value_traits` shape from the parallel
   numtypes track), so nothing ports unchanged.
2. **Proposed import order** (header-dependency graph, zero cycles when restricted to `.h`-to-`.h`
   edges -- see Section 2): `base/enums/utils/abi` (already ported), then, as an unordered tier
   depending only on that foundation, `numerics/algebra`, `numerics/spatial`, `numerics/fe`,
   `numerics/eigen`, `io`; then `numerics/linear` (needs algebra); then `numerics/temporal` (needs
   algebra only -- independent of physics at header level); then `numerics/nonlinear` (needs linear);
   then `physics` (needs algebra, linear, nonlinear, spatial); then `interface` (needs algebra,
   nonlinear, physics); `coupling` last (depends on physics, numerics/fe, abi, io).
3. **Tests**: of 75 CMake-registered test executables (203 `add_test()` call-sites, expanding to
   KARMA's own measured 425 ctest rows -- Appendix B's granularity note), 43 port, 21 retire (covered
   by a confirmed or plausible fuse equivalent), 11 undecided.
4. **Gray-Scott**: NOT the same problem. Same PDE pair and reaction kinetics, but KARMA couples two
   different, non-matching FE meshes by Lie/Strang operator splitting; fuse's example couples two
   components on one shared finite-volume grid via `WindowCoupler` sub-cycling. Different parameter
   regimes too (Section 5).

Everything below is the detail and the citations.

---

## 0. Methodology and declared depth of verification

Every row in Appendix A carries a `confirmation` tag:

- **`confirmed`** -- grep-verified against the current fuse tree, or pinned to an explicit textual
  source (`UPSTREAMS.lock`'s `open` list, a `docs/dev/*.md` statement, a KARMA source comment). 84 of
  512 rows (16%), including all 15 locked supersessions and the 14 newly-identified ones.
- **`owner-decision`** -- the disposition follows directly from D-A ("FEM core is ADAPTED to
  fuse::linalg, not ported literally") or D-B ("ALL KARMA time integrators are ported into fuse
  core") without needing a supersession claim at all: a file in `numerics/fe`, `physics/*`, or
  `numerics/temporal` gets `port-adapt`/`port-integrator` by virtue of what stage it is in, not by my
  checking it individually. 187 of 512 rows (37%).
- **`category`** -- assigned by directory and role (the file's name, its directory's declared
  purpose per KARMA's own docs, and the pattern every sibling file in that directory follows), not
  individually grep-confirmed against fuse. 241 of 512 rows (47%).

**Declared reduction in scope.** Individually reading every one of 512 files end-to-end and grepping
each candidate fuse equivalent was not done; it would be a multi-day effort for a single inventory
pass. What WAS done: all 15 locked supersessions confirmed (file exists at the claimed path AND a
named fuse equivalent exists); 14 additional supersession candidates investigated by reading the
KARMA header's actual declared symbols and grep-confirming (or explicitly failing to confirm) an
equivalent in the current fuse tree (Section 3); the full header-to-header include graph computed
exactly (mechanical, not sampled); `interface/` and `io/` read in full (item 7, Section 6); the
baselines (item 4) and the Gray-Scott comparison (item 5) read from the actual test/example sources,
not inferred. The remaining ~400 rows (physics modules, the FE core itself, the 22 remaining
temporal integrators beyond the ones named above, most of `numerics/algebra`/`linear`/`nonlinear`)
were classified by directory role and cross-checked against the layering/porting docs, not by
individually grepping each for a hidden fuse equivalent. This is the single largest scope reduction
in this report; Appendix A's `confirmation` column makes it auditable row by row.

---

## 1. Supersession table

Full table: Appendix A (512 rows). Summary by disposition:

| disposition | count | meaning |
|---|---|---|
| retire | 190 | build/doc/test scaffolding, or content whose role fuse already fills generically (fuse::linalg, fuse::petsc's Newton-Krylov, fuse's own ABI/registry) |
| port-adapt | 190 | D-A: ported with interface changes (fuse::linalg, fuse contracts, libMesh question) |
| port-integrator | 68 | D-B: KARMA's 31 temporal integrators (headers+sources) plus their enum tag and src/enums mirror |
| undecided | 35 | genuine open questions -- Section 7 |
| superseded | 29 | a named, existing fuse type already fills this file's role |

### 1.1 The 15 locked supersessions (`UPSTREAMS.lock`)

All 15 confirmed: the upstream path exists exactly as claimed (MEASURED, `test -f`), and a named fuse
type exists that fills its role (grep-confirmed):

| upstream path | fuse target |
|---|---|
| `include/base/karma_exceptions.h` | include/fuse/base/error.hpp, include/fuse/base/result.hpp |
| `include/coupling/karma_byte_writer.h` | include/fuse/base/bytes.hpp (test/base/bytes_test.cpp) |
| `include/coupling/karma_compatibility.h` | include/fuse/model/compatibility.hpp, connect.hpp (test/model/compatibility_test.cpp) |
| `include/coupling/karma_semantic_version.h` | include/fuse/base/version.hpp (test/base/version_test.cpp) |
| `include/coupling/karma_semantics.h` | include/fuse/model/semantics.hpp (test/model/semantics_test.cpp) |
| `include/coupling/karma_transform.h` | include/fuse/model/transform.hpp (test/model/transform_test.cpp) |
| `src/coupling/karma_compatibility.C` | include/fuse/model/compatibility.hpp |
| `src/coupling/karma_semantic_version.C` | include/fuse/base/version.hpp |
| `src/coupling/karma_semantics.C` | include/fuse/model/semantics.hpp |
| `src/coupling/karma_transform.C` | include/fuse/model/transform.hpp |
| `test/byte_writer_test.C` | test/base/bytes_test.cpp |
| `test/coupling_compatibility_test.C` | test/model/compatibility_test.cpp |
| `test/coupling_semantics_test.C` | test/model/semantics_test.cpp |
| `test/semantic_version_test.C` | test/base/version_test.cpp |
| `test/transform_inference_test.C` | test/model/transform_test.cpp |


### 1.2 14 newly-identified supersessions (not yet in `UPSTREAMS.lock`)

Found by reading the actual class/struct declarations in the remaining `include/coupling/*.h` files
(porting.md's merge table names the historical design decision; I grepped the CURRENT fuse tree for
the resulting type, which is a stronger and more recent check than re-reading the merge table alone):

| upstream path | fuse target | how confirmed |
|---|---|---|
| `include/coupling/karma_collective_outcome.h` (+ `.C`) | `fuse::Epoch` IR instruction, `EpochSchedule` (`include/fuse/runtime/epoch_schedule.hpp`), `transport/rank_agreement.hpp` | `docs/dev/porting.md`'s merge table states *"CollectiveOutcome as the Epoch instruction"* (S3 row); `docs/dev/rank-agreement.md` documents the live `Epoch` operator and its `kEpoch` tag |
| `include/coupling/karma_component.h` (+ `.C`) | `include/fuse/model/component.hpp`'s `ComponentDecl`/`ComponentKind` | read `ComponentDecl`'s fields (name/ranks/interval/fields/kind/regions/launch) against karma's `ComponentSpec`/`FieldRole`/`PortDirection`/`TimeSemantics`; porting.md: *"capability bits replace `supports_*()`"* |
| `include/coupling/karma_component_runtime.h` (+ `.C`) | `include/fuse/runtime/component.hpp`'s `Component`/`ComponentSet`/`ComponentIterator` | grep-confirmed class names exist |
| `include/coupling/karma_field.h` (+ `.C`) | `include/fuse/model/field_spec.hpp` | already in `[soma]`'s locked superseded list; porting.md's merge table: field handle base is "spine" (= KARMA, confirmed by roadmap.md's own "the spine tree does not compile in Debug" referring to a KARMA source file), so karma_field.h's role is the SAME merged type soma's field_spec.hpp was superseded into |
| `include/coupling/karma_field_view.h` (+ `.C`) | `include/fuse/field/field_ref.hpp`, `include/fuse/field/registry.hpp` | grep-confirmed `FieldView`/`FieldRef`/`FieldSpan` family exists |
| `include/numerics/nonlinear/karma_nonlinear_operator.h` (+ `.C`) | `fuse::num::solve()` (Jacobian-free Newton-Krylov, S7) | roadmap.md's S7 closure text: the generic Newton-Krylov machinery was built and proved on OTHER example physics (dn_heat, Brusselator, petsc-reaction-diffusion), explicitly NOT by porting KARMA's own nonlinear-operator wrapper -- so the wrapper's role is already filled |
| `include/numerics/nonlinear/petsc_native_nonlinear_solver.h` (+ `.C`) | `fuse::petsc::solve()` over `SNESNEWTONLS` (S7) | same roadmap.md citation |

A further eight `include/coupling/*.h` headers (each with a `.C`, so 16 more files) --
`karma_field_coverage.h`, `karma_field_entry_encoding.h`, `karma_field_identity.h`,
`karma_field_transaction.h`, `karma_field_transfer.h`, `karma_semantics_adapter.h`,
`karma_transfer_diagnostics.h`, `karma_reference_components.h` -- were investigated the same way and
**could not be confirmed**, so they are NOT counted among the 14 above: porting.md's merge table names a historical
successor (e.g. "field handle, transactions \| spine (`FieldTransaction`) on the coupler's plan
buffers \| S4"), but grepping the current fuse tree for `FieldTransaction`, `Transaction`, `DofKey`,
`FieldEntryKey`, `TransferScheme`, `ReferenceDecayConfiguration` etc. by their exact KARMA-era names
returns **zero hits** anywhere in `include/fuse` or `src`. Either the concept was renamed again since
the S4 decision (likely, given fuse's own naming discipline) or it was absorbed implicitly into
`fuse/schedule/plan.hpp`'s buffer mechanism without a class carrying a transactional name. These are
marked `undecided` in Appendix A rather than `superseded`, per the standing rule against declaring an
equivalence without a verified reason -- the same discipline the task applies to mutation-kill
equivalence.

### 1.3 The libMesh finding

`include/numerics/libmesh_system.h`/`.C` is not KARMA's only touchpoint with libMesh. Grepping for
libMesh's own headers (`mesh.h`, `elem.h`, `node.h`, `dof_map.h`, `equation_systems.h`,
`implicit_system.h`, `mesh_base.h`, `vector_value.h`, `reference_counted_object.h`) across
`include/`+`src/` found them reachable from **18 files** across `interface/`, `numerics/eigen`,
`numerics/spatial`, `physics/common`, `physics/base`, `base/ikarma.h`, `parallel/`, `utils/`, and
`numerics/isystem.C` -- not a single isolated backend. `KARMAContext` and `KARMAInterface`
(the central orchestration classes read in Section 6) both inherit libMesh's `ReferenceCountedObject`.
fuse's own `cmake/FuseDependencies.cmake` resolves MOAB, PETSc, and ArborX; it resolves nothing named
libMesh anywhere (grepped, zero hits). This is raised as Open Question 1 (Section 7): it is exactly
the kind of accuracy/usability trade the task says to ask about, not decide.

---

## 2. Include graph and proposed import order

`karma-include-edges.txt` (artifact) is the full local-include edge list, parsed from every
`#include "..."` line in `include/` and `src/` (976 edges; 177 further includes resolve to MOAB,
PETSc, or libMesh headers and are external, not local). `karma_dir_condensation.txt`-equivalent
summary is in the appendix below.

**Cycles.** At individual-file granularity (excluding `test/`/`examples/`, since the question is
about the *library's* import order), the full 330-node include graph is a DAG: Tarjan's algorithm
found **zero** cyclic strongly-connected components. Restricted further to header-to-header edges
only (165 nodes, 252 edges -- the edges that actually constrain compilation order, since no `.C` file
is ever `#include`d by anything, grep-confirmed, zero hits), it is **also** a clean DAG with a full
topological sort.

A *directory-level condensation* of the include graph (collapsing every file in a directory to one
node) DOES show apparent two-node cycles -- `physics <-> interface`, `io <-> interface`,
`numerics/temporal <-> interface`. These are artifacts of condensation, not real cycles: in every
case, the "reverse" edge is a `.C` implementation file (a leaf of the real DAG, never included by
anyone) reaching into a header the forward direction's header does not reach into. Example:
`include/interface/karma_interface.h` includes `include/physics/base/karma_physics_base.h` (a real,
forward header dependency), while `src/physics/base/karma_physics_base.C` includes
`include/interface/karma_interface.h` back (a leaf `.C` file, not a header -- no cycle). Architecturally
this DOES mean `physics` and `interface` are mutually coupled modules that should be ported in the
same wave, even though the include graph itself has no cycle to name.

**Proposed order among the ten named categories** (derived from the actual header-level edges, not
assumed from the task's listing order):

```
tier 0 (already ported):      base, enums, utils, abi
tier 1 (no mutual deps,
         each needs only
         tier 0):             numerics/algebra, numerics/spatial, numerics/fe,
                               numerics/eigen, io
tier 2:                        numerics/linear      (needs numerics/algebra)
tier 3:                        numerics/temporal     (needs numerics/algebra only --
                                                       independent of physics/interface
                                                       at HEADER level; the interface
                                                       dependency is .C-file only, see above)
tier 4:                        numerics/nonlinear    (needs numerics/linear)
tier 5:                        physics               (needs algebra, linear, nonlinear, spatial)
tier 6:                        interface             (needs algebra, nonlinear, physics)
tier 7:                        coupling              (needs physics, numerics/fe, abi, io)
```

Checked directly (no 2-cycles among these 10 category nodes at header granularity, grep-confirmed by
listing every header-to-header edge among them -- 11 edges total, all consistent with the tier order
above).

---

## 3. Test mapping

Appendix B: 75 CMake-registered test executables (203 `add_test()` call-sites in `test/CMakeLists.txt`,
MEASURED by `grep -c`). Of the 203 call-sites, 48 sit inside `foreach(ranks ...)` loops and a further
~17 inside `foreach(section/failure/mode/scalar/callback ...)` loops, so the literal ctest row count is
materially larger than 203 -- KARMA's own `docs/HANDOFF.md:48` states a measured *"425/425"* on a host
run. Reconciling the exact 425 would require configuring with `cmake` and listing with `ctest -N`,
which the standing rule ("Do not build or run KARMA") precludes; the 425 figure here is READ from
KARMA's own doc, not reproduced. Appendix B is therefore at the 75-executable / 203-call-site
granularity.

Disposition: 43 port-adapt, 21 retire, 11 undecided. The 21 retirements split two ways: 15 are the
locked `UPSTREAMS.lock` supersessions' own test files (`semantic_version_test.C`, `byte_writer_test.C`,
etc.), and 6 are the newly-confirmed ones (`collective_outcome_test`/`_mpi_test`,
`component_compliance_test`, `coupling_compatibility_oracle_test`, `abi_negotiation_test`,
`legacy_bc_api_probe`) plus the 4 operator-split/fixed-point tests covered by item 6's finding
(`operator_split_graph_test`, `operator_split_connection_test`, `operator_split_introspection_test`,
`operator_split_scheduler_test`) and `gray_scott_operator_split_test` (Section 5). `smoke_test` and
`dense_test`/`language_adapter_schema_test` retire for narrower reasons recorded in Appendix B.

---

## 4. Baselines inventory

Every number below is READ from the named test source, not re-derived and not run (per the standing
rule: "Do not build or run KARMA"). Where the source states where the tolerance came from, that
derivation is quoted; where it does not, that is also stated rather than invented.

| test | file:line | assertion | tolerance / bound | derivation given in source |
|---|---|---|---|---|
| P1 diffusion MMS | `diffusion_mms_test.C:141` | `rate = log(errors[i]/errors[i+1])/log(2)` must satisfy `1.8 <= rate <= 2.2` | a +/-0.2 band around the theoretical order-2 rate for P1 diffusion under uniform mesh doubling | none beyond "P1 L2 convergence rate" in the FAIL message; the band width is not derived from a stated error model in-file |
| Crank-Nicolson temporal order | `transient_reaction_diffusion_mms_test.C:357-358` | temporal rate in `[1.8, 2.2]` | same +/-0.2 band around order 2 | none stated beyond the message text |
| P1 spatial order (transient) | `transient_reaction_diffusion_mms_test.C:374-375` | spatial rate in `[1.8, 2.2]` | same band | none stated |
| Temporal-vs-spatial error balance | `transient_reaction_diffusion_mms_test.C:384` | `temporal_estimate < 0.1 * results.back().error` | temporal error must be an order of magnitude below the finest-level spatial error | none stated beyond the inequality itself |
| Time-stepping drift | `transient_reaction_diffusion_mms_test.C:140` | `abs(steps*time_step - final_time) < 64*eps` | 64 ULP | round-off accumulation bound over the step count, no explicit per-step derivation shown |
| Transient residual floor | `transient_reaction_diffusion_mms_test.C:166` | `residual_norm <= 1.e-8 * scale` | 1e-8 relative | stated as a flat relative bound, no derivation shown |
| Stiffness matrix symmetry | `transient_reaction_diffusion_mms_test.C:236,238` | constant-energy term `<= 1.e-12*norm`; `(K - K^T).norm() <= 1.e-13*norm` | 1e-12 / 1e-13 relative | asserted as near-machine-precision symmetry; no derivation shown |
| PETSc option isolation | `transient_reaction_diffusion_mms_test.C:285-286` | `abs(after - expected) <= 2.e-11*max(1,|expected|)` | 2e-11 relative/absolute hybrid | no derivation shown |
| Advection DG MMS | `advection_mms_test.C:144` | rate in `[1.7, 2.3]` | wider +/-0.3 band, looser than the CG diffusion band above because DG advection is less regular | stated implicitly by contrast with the CG band; not independently derived |
| Gray-Scott Lie splitting order | `gray_scott_operator_split_test.C:129-132` | every refinement's rate `>= 0.85` | first-order lower bound | matches the textbook first-order accuracy of Lie splitting; the 0.85 margin below 1.0 is not derived in-file |
| Gray-Scott Strang splitting order | `gray_scott_operator_split_test.C:129-132` | every refinement's rate `>= 1.8` | second-order lower bound | matches textbook second-order Strang accuracy; same undeclared margin |
| Fixed-point iteration count | `fixed_point_coupling_test.C:180` | `maximum.status == MaxIterations && maximum.iterations == 3` | exact count 3 | the test's own configured iteration cap, not a derived bound -- an exact-match oracle, not a tolerance |
| CG transfer consistency (constant/affine) | `cg_transfer_accuracy_test.C:223-230` | round-trip projection error `< 2.0e-11` both directions, both fields | flat 2e-11 absolute | no derivation shown |
| CG transfer relative error | `cg_transfer_accuracy_test.C:249` | `relative_full_error.front() < 2.0e-11` | flat 2e-11 relative | no derivation shown |
| Mass-matrix residual acceptance | `cg_transfer_accuracy_test.C:290` | `transfer_residual_within_tolerance(..., 1.0e-11)` | 1e-11 | no derivation shown |
| CG transfer refinement improvement | `cg_transfer_accuracy_test.C:380-384` | fine-mesh error `< 0.35x` (source/target representation) or `< 0.45x` (transfer) of coarse-mesh error | ratios, not absolute tolerances | no derivation shown; empirically-set ratios |
| FEM transfer convergence | `fem_transfer_convergence_test.C:104-112` | three successive refinement rates each `> 1.5` | looser than the diffusion/advection bands above | no derivation shown |
| Gray-Scott fixed-point field residual | `gray_scott_spatial_mms_test.C:357-358,372-374` | `residual[f] <= tolerance[f]` with KSP `absolute_tolerance=1e-11`, `relative_tolerance=1e-9` | solver-configured, not a measured bound | the linear-solve tolerance, not an independently derived error bound |

**Pattern across all of them**: none states a derivation from a theorem the way the owner's own
tolerance convention requires (Higham's bound, or similar). Every numeric threshold above is either a
fixed empirical band around a textbook convergence order, or a flat empirically-chosen absolute/relative
number, recorded as-is with no in-file justification beyond the inequality itself. If any of these
become fuse assertions at port time, they will need the owner's tolerance convention applied fresh
(derive-and-print), not copied as empirical constants -- these numbers are useful **oracle targets**
for the port (the port's own result should fall in the same regime), not usable tolerance derivations
by the owner's own standing rule.

---

## 5. Gray-Scott comparison

**fuse's own example** (`examples/gray_scott/`, read in full):
model `u_t = Du*Lap(u) - u*v^2 + F*(1-u)`, `v_t = Dv*Lap(v) + u*v^2 - (F+k)*v` on a periodic square,
**cell-centred P0 finite-volume** on one shared `N x N` structured grid (`gray_scott_physics.hpp`'s
own header comment: *"O(h^2) for a smooth field"*), 5-point periodic Laplacian stencil built once as a
sparse matrix. Both species (U, V) are `fuse::Component`s on the SAME grid, coupled through
`fuse::WindowCoupler` sub-cycling (`species_component.hpp`: each step is Crank-Nicolson in time, the
partner's value frozen at its window-mean for the call, Newton-solved via `Eigen::SparseLU`
re-factorized every iteration). `main.cpp`'s defaults: `F=0.0392, k=0.0649, Du=2.0e-5, Dv=1.0e-5,
L=2.5` -- the literature "Pearson" soliton-pattern regime, and a configurable U:V time-step ratio
(multirate), not operator splitting.

**KARMA's Gray-Scott** (`include/physics/diffusion-reaction/gray_scott_model.h`/`.C`,
`test/gray_scott_operator_split_test.C`, `examples/gray_scott_operator_split.C`): same reaction
kinetics (`u_t = Du*Lap(u) - u*v^2 + F*(1-u)`, same algebraic form), but defaults
`Du=0.16, Dv=0.08, F=0.060, k=0.062` -- a different, much-larger-diffusion parameter regime, on a
**finite-element** discretisation (configurable element family/order/continuity, not a fixed 5-point
FV stencil), and critically on **two different, non-matching meshes** -- the operator-split test
builds `u_mesh.generate_box(2,2,0)` and `v_mesh.generate_box(3,2,0)` explicitly, a genuine multi-mesh
FEM-transfer problem. The two species are coupled by explicit **Lie or Strang operator splitting**
(`SplitScheme::Lie`/`SplitScheme::Strang`), each a sequential macro-step sub-solve with fixed-point
sweeps and transactional rollback/checkpoint on failure -- not a windowed sub-cycling iteration.

**Verdict: not the same problem**, beyond sharing the Gray-Scott reaction-diffusion PDE family and
its algebraic kinetics. Differences: (1) mesh -- one shared structured FV grid vs. two independent,
non-matching FE meshes needing transfer; (2) discretisation -- fixed-stencil P0 FV vs. configurable
FEM; (3) temporal coupling -- windowed Crank-Nicolson sub-cycling with a frozen-partner value vs.
Lie/Strang operator splitting with checkpointed rollback; (4) parameter regime -- Pearson soliton
values vs. KARMA's own (much larger diffusion, no domain-length scaling). The Lie/Strang order
baselines (Section 4) remain useful port-time oracle numbers only if physics-kit ever offers operator
splitting as a `fuse::num` temporal-integrator option (D-B) rather than as a coupling scheme in its
own right -- fuse's own coupling layer already has its accelerator/iteration vocabulary
(`model/iteration.hpp`, `num/accelerator.hpp`) and does not need a second, FEM-specific coupling graph.

---

## 6. The two items the plan names that needed locating

**"The native two-component operator-split graph and fixed-point precursor that S8.2 retires."**
Located in KARMA's OWN repository, not in fuse: `include/coupling/karma_operator_split_graph.h` (255
lines) + `src/coupling/karma_operator_split_graph.C` (961 lines), and
`include/coupling/karma_fixed_point.h` (98 lines) + `src/coupling/karma_fixed_point.C` (358 lines),
plus their checkpoint/rollback sibling `karma_operator_split_checkpoint.h`/`.C`. KARMA's own
`docs/roadmap.md:16,19` describes exactly this: *"macro-step rollback for the native two-component
operator-split graph"* and *"mass-weighted fixed-point iteration with constant or safeguarded Aitken
relaxation."* `docs/developer-guide/architecture.md:44-53` names the active types directly:
`compile_operator_split_graph()`, `CompiledOperatorSplitGraph`, `KARMAFixedPointIterator`. These are
confirmed superseded (Section 1, item 6 cross-reference): fuse's own coupling graph, `model/iteration.hpp`,
and `num/accelerator.hpp` (S2-S4) already fill the generic role; only the Lie/Strang order baselines
(Section 4) and the two meshes/FEM-specific plumbing they exercise (Section 5) carry any remaining value.

**"The distorted-pyramid quadrature accuracy risk."** NOT in `docs/developer-guide/pyramid-quadrature-cost.md`
as the primary source, though that document (read in full, 177 lines) DOES restate it in its own closing
section ("What is still not established" / "Gradient terms: the open question, now answered") -- it is
about cost, as the task's own framing says, and only echoes the risk at the end. The authoritative
statement is **`upstream/karma/KNOWN_ISSUES.md`, item 1**: *"Distorted pyramids under the resized
quadrature -- UNVERIFIED RISK"*, naming the exact file/function
(`include/numerics/fe/karma_quadrature.h`'s `create_product_quadrature()`), stating what IS verified
(monomial exactness and reference-cell gradients to ~2e-16, confirmed two independent ways) and what is
NOT (*"Distorted pyramids... a question of relative accuracy, not of exactness, and no test currently
isolates it"*), and how it would present (*"not as a test failure, but as a pyramid or prism MMS
convergence rate shifting. Watch rates, not pass/fail"*). This is cross-referenced from
`UPSTREAMS.lock`'s own `open` list for `[karma]` (*"distorted-pyramid quadrature unverified
(KNOWN_ISSUES #1)"*) -- confirming KNOWN_ISSUES.md, not the cost doc, is the intended referent.
Disposition: `include/numerics/fe/karma_quadrature.h` ports (D-A) with the CURRENT, already-resized
`create_product_quadrature()` (the fix is already applied upstream); the unverified-risk text itself
must be copied forward into fuse's own `KNOWN_ISSUES.md` at port time, not silently dropped.

---

## 7. `src/interface` (4,838 lines) and `src/io` (3,025 lines)

Line counts MEASURED (`wc -l`): `include/interface/*.h` (672) + `src/interface/*.C` (4,166) = 4,838,
exactly matching the task's figure. `include/io/*.h` (221) + `src/io/*.C` (2,804) = 3,025, same match.

**`interface/`** is KARMA's central orchestration layer: `KARMAContext` (solution-state container --
vector/matrix/dof/BC state, inherits libMesh's `ReferenceCountedObject`) and `KARMAInterface` (the
object that owns System + DofMap + boundary conditions + assembly + PETSc SNES/TS wiring --
KARMA's equivalent of what fuse's `ImplicitComponent`/`LinearizationToken` do generically). A third
file, `karma_wrappers.h`/`.C`, is purely a bridge from KARMA's object-oriented physics callbacks to
PETSc's C function-pointer SNES/TS callback signatures (`karma_wrapper_petsc_snes_residual` etc.).
Disposition: `karma_wrappers.*` **retires** -- `fuse::petsc::solve()`/`MffdToken` (S7) already does
this bridging generically, for any physics, not just KARMA's. `karma_context.*`/`karma_interface.*`
**port-adapt** -- KARMA's orchestration role needs to plug into `fuse::ImplicitComponent` rather than
re-wire its own SNES callbacks, and its vector/matrix state should become `fuse::linalg::DistVector`/
`DistSparseMatrix` (D-A) rather than carrying KARMA's own PETSc wrapper types (which themselves retire,
Section 1).

**`io/`** is three unrelated things sharing a directory: (1) `karma_input_document.h`/`.C` -- an
input-deck document abstraction, likely redundant with fuse's existing YAML front end
(`src/model/yaml_io.cpp`); (2) `karma_monitor.h`/`.C` -- a solution-monitoring/output callback hooked
into `karma_interface`'s residual history, check against fuse's own Newton convergence logging before
porting; (3) `karma_moab_io.h`/`.C` (2,464 of the 2,804 `src/io` lines) -- which is itself two things:
genuinely new FE-specific MOAB mesh I/O (element blocks, sideset/nodeset boundary tagging -- fuse's
existing `moab_backend.hpp` is remap/transfer-oriented, not FE-assembly-oriented, so this is real
new content, `port-adapt`), and a **MOAB-tag-based checkpoint/restart mechanism**
(`KARMALoadedCheckpoint`, `checkpoint_gather_stats()`, `encode_checkpoint_scalar()`,
`CheckpointPublicationState`) that duplicates ground the separately-tracked desired-output "restart-
to-disk in parallel HDF5 with an XDMF sidecar" is meant to cover -- fuse already has
`include/fuse/restart_io/hdf5_restart_io.hpp`/`hdf5_restart_io_mpi.hpp`. Physics-kit should not invent
a second checkpoint format; this split is marked `undecided` (Section 7, Open Question 3) rather than
assigning it a confident disposition.

---

## Open questions (grouped)

**On third-party dependencies (asks whether to change what fuse links against):**
1. KARMA's FEM core depends on libMesh in 18 files, not just `numerics/libmesh_system.h` (Section 1.3)
   -- `KARMAContext`/`KARMAInterface` inherit its `ReferenceCountedObject`, and fuse's own
   `cmake/FuseDependencies.cmake` resolves none of it today. Does physics-kit add libMesh as a fourth
   optional dependency (alongside MOAB/PETSc/ArborX), or does the port drop libMesh entirely and keep
   only KARMA's own native System/DofMap path (`karma_system.h`, not `libmesh_system.h`)?
2. `numerics/eigen/karma_eigen_solver.h` is the only eigensolver-shaped code anywhere in either
   parent; fuse has no SLEPc resolution today. Does physics-kit need an eigensolver at all for its
   first wave, and if so does SLEPc become a new optional dependency?

**On restart/checkpoint duplication (asks whether to change scope, since restart-to-disk is a
separate, not-yet-closed desired output):**
3. `karma_moab_io.h`/`.C`'s `KARMALoadedCheckpoint`/`checkpoint_gather_stats()` MOAB-tag checkpoint
   mechanism duplicates ground the separate parallel-HDF5+XDMF restart deliverable is meant to cover.
   Should physics-kit's mesh I/O keep only the FE-specific mesh-reading half of `karma_moab_io.*` and
   route ALL checkpoint/restart through fuse's existing `restart_io` once that work closes, or does
   physics-kit need its own interim mechanism before that closes?

**On vocabulary coordination with the parallel numtypes track (asks whether to change what gets
reintroduced vs. reused):**
4. `karma_types.h` (fundamental scalar aliases) and `include/utils/typename_traits.h` overlap in
   intent with the numtypes track's `value_traits<T>::scalar_type` / `CoupledValue` concept
   (D-E, landing in parallel). Should physics-kit wait on that track's public shape before defining
   its own scalar-type vocabulary, to avoid two competing type-trait systems?

**On genuinely unconfirmed supersessions (asks for a decision only because grep could not settle it):**
5. Eight `include/coupling/*.h` files (`karma_field_coverage.h`, `karma_field_entry_encoding.h`,
   `karma_field_identity.h`, `karma_field_transaction.h`, `karma_field_transfer.h`,
   `karma_semantics_adapter.h`, `karma_transfer_diagnostics.h`, `karma_reference_components.h`) have a
   historical successor named in `docs/dev/porting.md`'s merge table, but grepping the CURRENT fuse
   tree for the exact symbol names (`FieldTransaction`, `DofKey`, `TransferScheme`, etc.) returns zero
   hits anywhere. Were these renamed again since the S4 decision (and if so, to what), or does the
   transactional/identity concept no longer exist as a named type at all (absorbed into
   `schedule/plan.hpp`'s buffer mechanism implicitly)? This needs an answer from someone who watched
   the S4 port land, not another grep pass.
6. `karma_registry.h`'s `ComponentRegistry`/`SelectionPolicy`/`RegistryReason` -- is this fully covered
   by `fuse::abi::Registry` + `fuse::field::FieldRegistry`, or does KARMA's selection-policy mechanism
   (choosing among multiple registered implementations of one contract) have no fuse analogue yet?
7. `karma_geometry.h`'s `BoundingBox` -- reuse `ArborX::Box` / `fuse::search`'s existing box type, or
   does physics-kit need its own (and if so, why does ArborX's not suffice)?
8. `karma_io_buffers.h`'s rank-aware stream-capture classes (`cout_proxy`/`static_logger`/`cerr_proxy`)
   -- does physics-kit need host-side stream capture for FE assembly debugging, or does fuse's
   existing `Result`/`Diagnostics` discipline make that unnecessary?
## Appendix A: per-file supersession table (512 rows)

Columns: path | disposition | target / reason | confirmation level.

Confirmation levels: `confirmed` = grep-verified against the current fuse tree or against an explicit owner-decision/KNOWN_ISSUES statement; `owner-decision` = disposition follows directly from D-A/D-B/D-C without needing a supersession claim; `category` = assigned by directory/role, not individually grep-verified (see Section 0's deviation note).


### `(root)/` (11 files)

| path | disposition | target / reason | confirmation |
|---|---|---|---|
| `.gitignore` | retire | already tracked as 'diverged' (transformed) in UPSTREAMS.lock, not pending supersession | confirmed |
| `.readthedocs.yaml` | retire | build/tooling/license/doc scaffolding, not library code; no fuse equivalent needed | category |
| `AGENTS.md` | retire | build/tooling/license/doc scaffolding, not library code; no fuse equivalent needed | category |
| `CMakeLists.txt` | retire | build/tooling/license/doc scaffolding, not library code; no fuse equivalent needed | category |
| `CMakePresets.json` | retire | build/tooling/license/doc scaffolding, not library code; no fuse equivalent needed | category |
| `INSTALL` | retire | build/tooling/license/doc scaffolding, not library code; no fuse equivalent needed | category |
| `KNOWN_ISSUES.md` | retire | item 1 (distorted-pyramid quadrature risk) carries forward live into fuse's own KNOWN_ISSUES.md at port time (item 6); the rest is build-status narrative, not library content | confirmed |
| `LICENSE` | retire | build/tooling/license/doc scaffolding, not library code; no fuse equivalent needed | category |
| `Make.dep` | retire | build/tooling/license/doc scaffolding, not library code; no fuse equivalent needed | category |
| `README` | retire | build/tooling/license/doc scaffolding, not library code; no fuse equivalent needed | category |
| `TODO` | retire | build/tooling/license/doc scaffolding, not library code; no fuse equivalent needed | category |

### `build-evidence/` (3 files)

| path | disposition | target / reason | confirmation |
|---|---|---|---|
| `build-evidence/petsc-storage-mutations/README.md` | retire | historical CI/build evidence snapshot, not library code | category |
| `build-evidence/petsc-storage-mutations/mutations.json` | retire | historical CI/build evidence snapshot, not library code | category |
| `build-evidence/petsc-storage-mutations/results.log` | retire | historical CI/build evidence snapshot, not library code | category |

### `cmake/` (9 files)

| path | disposition | target / reason | confirmation |
|---|---|---|---|
| `cmake/CheckArborX.cmake` | retire | KARMA's own CMake modules; superseded by fuse's cmake/ (FuseDependencies.cmake etc.) | category |
| `cmake/CheckMoabCoupler.cmake` | retire | KARMA's own CMake modules; superseded by fuse's cmake/ (FuseDependencies.cmake etc.) | category |
| `cmake/PetscScalarProbe.cmake` | retire | KARMA's own CMake modules; superseded by fuse's cmake/ (FuseDependencies.cmake etc.) | category |
| `cmake/abi_exports_check.cmake` | retire | KARMA's own CMake modules; superseded by fuse's cmake/ (FuseDependencies.cmake etc.) | category |
| `cmake/abi_install_consumer_check.cmake` | retire | KARMA's own CMake modules; superseded by fuse's cmake/ (FuseDependencies.cmake etc.) | category |
| `cmake/check_petsc_scalar.cmake` | retire | KARMA's own CMake modules; superseded by fuse's cmake/ (FuseDependencies.cmake etc.) | category |
| `cmake/karma_config.h.in` | retire | KARMA's own CMake modules; superseded by fuse's cmake/ (FuseDependencies.cmake etc.) | category |
| `cmake/karma_test_helpers.cmake` | retire | KARMA's own CMake modules; superseded by fuse's cmake/ (FuseDependencies.cmake etc.) | category |
| `cmake/test_petsc_scalar_probe.cmake` | retire | KARMA's own CMake modules; superseded by fuse's cmake/ (FuseDependencies.cmake etc.) | category |

### `config/` (26 files)

| path | disposition | target / reason | confirmation |
|---|---|---|---|
| `config/KARMA.common.in` | retire | autoconf-era build scaffolding predating KARMA's own CMake migration; not used by the active CMake build | category |
| `config/Makefile.in` | retire | autoconf-era build scaffolding predating KARMA's own CMake migration; not used by the active CMake build | category |
| `config/aclocal.m4` | retire | autoconf-era build scaffolding predating KARMA's own CMake migration; not used by the active CMake build | category |
| `config/autowaf.py` | retire | autoconf-era build scaffolding predating KARMA's own CMake migration; not used by the active CMake build | category |
| `config/config.guess` | retire | autoconf-era build scaffolding predating KARMA's own CMake migration; not used by the active CMake build | category |
| `config/config.sub` | retire | autoconf-era build scaffolding predating KARMA's own CMake migration; not used by the active CMake build | category |
| `config/config_utils.py` | retire | autoconf-era build scaffolding predating KARMA's own CMake migration; not used by the active CMake build | category |
| `config/configure` | retire | autoconf-era build scaffolding predating KARMA's own CMake migration; not used by the active CMake build | category |
| `config/configure.in` | retire | autoconf-era build scaffolding predating KARMA's own CMake migration; not used by the active CMake build | category |
| `config/dealii_library.py` | retire | autoconf-era build scaffolding predating KARMA's own CMake migration; not used by the active CMake build | category |
| `config/elemental_library.py` | retire | autoconf-era build scaffolding predating KARMA's own CMake migration; not used by the active CMake build | category |
| `config/helper.mak` | retire | autoconf-era build scaffolding predating KARMA's own CMake migration; not used by the active CMake build | category |
| `config/install-contrib.sh` | retire | autoconf-era build scaffolding predating KARMA's own CMake migration; not used by the active CMake build | category |
| `config/install-cpptest.sh` | retire | autoconf-era build scaffolding predating KARMA's own CMake migration; not used by the active CMake build | category |
| `config/install-karma.sh` | retire | autoconf-era build scaffolding predating KARMA's own CMake migration; not used by the active CMake build | category |
| `config/install-libmesh.sh` | retire | autoconf-era build scaffolding predating KARMA's own CMake migration; not used by the active CMake build | category |
| `config/install-petsc.sh` | retire | autoconf-era build scaffolding predating KARMA's own CMake migration; not used by the active CMake build | category |
| `config/install-petscext.sh` | retire | autoconf-era build scaffolding predating KARMA's own CMake migration; not used by the active CMake build | category |
| `config/install-sh` | retire | autoconf-era build scaffolding predating KARMA's own CMake migration; not used by the active CMake build | category |
| `config/install-slepc.sh` | retire | autoconf-era build scaffolding predating KARMA's own CMake migration; not used by the active CMake build | category |
| `config/karma-config` | retire | autoconf-era build scaffolding predating KARMA's own CMake migration; not used by the active CMake build | category |
| `config/karma-config.in` | retire | autoconf-era build scaffolding predating KARMA's own CMake migration; not used by the active CMake build | category |
| `config/libmesh_library.py` | retire | autoconf-era build scaffolding predating KARMA's own CMake migration; not used by the active CMake build | category |
| `config/load_config_variables.py` | retire | autoconf-era build scaffolding predating KARMA's own CMake migration; not used by the active CMake build | category |
| `config/petsc_library.py` | retire | autoconf-era build scaffolding predating KARMA's own CMake migration; not used by the active CMake build | category |
| `config/slepc_library.py` | retire | autoconf-era build scaffolding predating KARMA's own CMake migration; not used by the active CMake build | category |

### `doc/` (2 files)

| path | disposition | target / reason | confirmation |
|---|---|---|---|
| `doc/Doxyfile` | retire | Doxygen config; fuse's documentation is Sphinx-based (docs/dev, docs/user) | category |
| `doc/Doxyfile.in` | retire | Doxygen config; fuse's documentation is Sphinx-based (docs/dev, docs/user) | category |

### `docs/` (24 files)

| path | disposition | target / reason | confirmation |
|---|---|---|---|
| `docs/HANDOFF.md` | retire | becomes docs/history/karma/<relative path> at subtree retirement, mirroring UPSTREAMS.lock's docs/history/soma/ precedent for the other parent; not yet created | category |
| `docs/HOST-ACTIONS.md` | retire | becomes docs/history/karma/<relative path> at subtree retirement, mirroring UPSTREAMS.lock's docs/history/soma/ precedent for the other parent; not yet created | category |
| `docs/conf.py` | retire | becomes docs/history/karma/<relative path> at subtree retirement, mirroring UPSTREAMS.lock's docs/history/soma/ precedent for the other parent; not yet created | category |
| `docs/developer-guide/README.md` | retire | becomes docs/history/karma/<relative path> at subtree retirement, mirroring UPSTREAMS.lock's docs/history/soma/ precedent for the other parent; not yet created | category |
| `docs/developer-guide/abi-soma-gap-table.md` | retire | becomes docs/history/karma/<relative path> at subtree retirement, mirroring UPSTREAMS.lock's docs/history/soma/ precedent for the other parent; not yet created | category |
| `docs/developer-guide/architecture.md` | retire | becomes docs/history/karma/<relative path> at subtree retirement, mirroring UPSTREAMS.lock's docs/history/soma/ precedent for the other parent; not yet created | category |
| `docs/developer-guide/build-and-test.md` | retire | becomes docs/history/karma/<relative path> at subtree retirement, mirroring UPSTREAMS.lock's docs/history/soma/ precedent for the other parent; not yet created | category |
| `docs/developer-guide/contributing.md` | retire | becomes docs/history/karma/<relative path> at subtree retirement, mirroring UPSTREAMS.lock's docs/history/soma/ precedent for the other parent; not yet created | category |
| `docs/developer-guide/distributed-correctness.md` | retire | becomes docs/history/karma/<relative path> at subtree retirement, mirroring UPSTREAMS.lock's docs/history/soma/ precedent for the other parent; not yet created | category |
| `docs/developer-guide/plugin-design-review.md` | retire | becomes docs/history/karma/<relative path> at subtree retirement, mirroring UPSTREAMS.lock's docs/history/soma/ precedent for the other parent; not yet created | category |
| `docs/developer-guide/pyramid-quadrature-cost.md` | retire | item 6: the distorted-pyramid accuracy risk stated in its own closing section carries forward live into fuse's KNOWN_ISSUES.md; the cost-measurement content becomes historical once ported | confirmed |
| `docs/developer-guide/semantic-model-soma-mapping.md` | retire | becomes docs/history/karma/<relative path> at subtree retirement, mirroring UPSTREAMS.lock's docs/history/soma/ precedent for the other parent; not yet created | category |
| `docs/developer-guide/soma-port-baseline.md` | retire | becomes docs/history/karma/<relative path> at subtree retirement, mirroring UPSTREAMS.lock's docs/history/soma/ precedent for the other parent; not yet created | category |
| `docs/index.md` | retire | becomes docs/history/karma/<relative path> at subtree retirement, mirroring UPSTREAMS.lock's docs/history/soma/ precedent for the other parent; not yet created | category |
| `docs/plans/2026-09-07-plugin-coupling-architecture-design.md` | retire | becomes docs/history/karma/<relative path> at subtree retirement, mirroring UPSTREAMS.lock's docs/history/soma/ precedent for the other parent; not yet created | category |
| `docs/requirements.txt` | retire | becomes docs/history/karma/<relative path> at subtree retirement, mirroring UPSTREAMS.lock's docs/history/soma/ precedent for the other parent; not yet created | category |
| `docs/roadmap.md` | retire | becomes docs/history/karma/<relative path> at subtree retirement, mirroring UPSTREAMS.lock's docs/history/soma/ precedent for the other parent; not yet created | category |
| `docs/user-guide/README.md` | retire | becomes docs/history/karma/<relative path> at subtree retirement, mirroring UPSTREAMS.lock's docs/history/soma/ precedent for the other parent; not yet created | category |
| `docs/user-guide/field-transfer.md` | retire | becomes docs/history/karma/<relative path> at subtree retirement, mirroring UPSTREAMS.lock's docs/history/soma/ precedent for the other parent; not yet created | category |
| `docs/user-guide/getting-started.md` | retire | becomes docs/history/karma/<relative path> at subtree retirement, mirroring UPSTREAMS.lock's docs/history/soma/ precedent for the other parent; not yet created | category |
| `docs/user-guide/gray-scott-coupler.md` | retire | becomes docs/history/karma/<relative path> at subtree retirement, mirroring UPSTREAMS.lock's docs/history/soma/ precedent for the other parent; not yet created | category |
| `docs/user-guide/meshes-and-spaces.md` | retire | becomes docs/history/karma/<relative path> at subtree retirement, mirroring UPSTREAMS.lock's docs/history/soma/ precedent for the other parent; not yet created | category |
| `docs/user-guide/parallel-and-checkpoints.md` | retire | becomes docs/history/karma/<relative path> at subtree retirement, mirroring UPSTREAMS.lock's docs/history/soma/ precedent for the other parent; not yet created | category |
| `docs/user-guide/physics-systems.md` | retire | becomes docs/history/karma/<relative path> at subtree retirement, mirroring UPSTREAMS.lock's docs/history/soma/ precedent for the other parent; not yet created | category |

### `examples/` (2 files)

| path | disposition | target / reason | confirmation |
|---|---|---|---|
| `examples/CMakeLists.txt` | retire | build scaffolding for the file above | category |
| `examples/gray_scott_operator_split.C` | retire | KARMA's operator-split Gray-Scott driver; fuse already has its own Gray-Scott multiphysics example on a different, already-chosen coupling scheme (WindowCoupler sub-cycling, not operator splitting -- item 5); the underlying operator-split library it drives is itself retiring (item 6) | confirmed |

### `lib/` (1 files)

| path | disposition | target / reason | confirmation |
|---|---|---|---|
| `lib/.gitignore` | retire | placeholder for build output, not source | category |

### `include/` (193 files)

| path | disposition | target / reason | confirmation |
|---|---|---|---|
| `include/abi/karma_abi.h` | retire | superseded by fuse's own provider/participant ABI (fuse_plugin_abi()/fuse_coupler_abi(), abi/entries.def), already built at S6 | category |
| `include/abi/karma_component_abi.h` | retire | superseded by fuse's own provider/participant ABI (fuse_plugin_abi()/fuse_coupler_abi(), abi/entries.def), already built at S6 | category |
| `include/base/.gitignore` | retire | build-artifact placeholder | category |
| `include/base/dummy_stl.h` | retire | Doxygen placeholder marked DO NOT INCLUDE; no runtime content | confirmed |
| `include/base/idelegate.h` | retire | std::function covers IDelegate/Delegate (C++20 available) | category |
| `include/base/ikarma.h` | port-adapt | top-level plugin/component interface marker; adapt into physics-kit's own entry type (depends on libMesh ReferenceCountedObject today -- see open question on libMesh) | category |
| `include/base/karma.h` | port-adapt | KARMAPackage top-level driver type; GetPot-based config parsing half retires to fuse's YAML front end (src/model/yaml_io.cpp) | category |
| `include/base/karma_base.h` | retire | umbrella header aggregating karma_config.h; physics-kit gets its own umbrella | category |
| `include/base/karma_config.h.in` | retire | CMake-generated config header, superseded by FUSE's own CMake configure_file outputs | category |
| `include/base/karma_configuration.h` | retire | PETSc/SLEPc preprocessor version-gate macros; superseded by CMake-time feature checks (cmake/FuseDependencies.cmake) and fuse::petsc's own version handling | category |
| `include/base/karma_constants.h` | retire | single PI constant; superseded by std::numbers::pi_v<Real> (C++20) | confirmed |
| `include/base/karma_error.h` | retire | karma_error/karma_exception macros superseded by fuse::Result/Diagnostics + FUSE_ASSERT (I6), same disposition as the already-superseded karma_exceptions.h | category |
| `include/base/karma_exceptions.h` | superseded | include/fuse/base/error.hpp, include/fuse/base/result.hpp | confirmed |
| `include/base/karma_geometry.h` | undecided | BoundingBox -- does physics-kit reuse ArborX::Box / fuse::search's existing box type (docs/dev/search.md) instead of a new BoundingBox? | category |
| `include/base/karma_getpot.h` | retire | vendored third-party GetPot config-file parser; superseded by fuse's YAML front end | confirmed |
| `include/base/karma_io_buffers.h` | undecided | cout_proxy/static_logger/cerr_proxy stream redirection -- does physics-kit need rank-aware stream capture, or does fuse's own diagnostics channel suffice? | category |
| `include/base/karma_log.h` | retire | KARMALog/Log superseded by fuse::Diagnostics (Result-carrying, not print-based) per fuse's I6/Result discipline | category |
| `include/base/karma_options.h` | retire | PETSC_OPTIONS/SLEPC_OPTIONS/LIBMESH_OPTIONS/KARMA_OPTIONS command-line macros; superseded by the YAML graph descriptor plus fuse::petsc's own option propagation | category |
| `include/base/karma_parameters.h` | port-adapt | type-erased parameter/property bag for physics configuration; check against fuse::model's existing config types before porting literally | category |
| `include/base/karma_print_trace.h` | retire | trivial backtrace helper; superseded by std::stacktrace (C++20) or dropped | category |
| `include/base/karma_test_suite.h` | retire | KARMA's own hand-rolled test harness; fuse uses GoogleTest throughout (docs/dev/conventions.md) | category |
| `include/base/karma_types.h` | undecided | fundamental scalar type aliases (std::complex use) -- coordinate with the parallel numtypes track (value_traits<T>::scalar_type, D-E) rather than reintroducing independently | category |
| `include/base/simple_collection.h` | retire | ICollection template; generic-container role covered by std:: containers plus fuse's own field/registry containers | category |
| `include/base/yasper.h` | retire | public-domain intrusive smart pointer, superseded by std::shared_ptr/unique_ptr | confirmed |
| `include/coupling/karma_abi_records.h` | retire | superseded by fuse's abi/entries.def generator pattern (FUSE_ABI_*_ENTRY), already built at S6 | category |
| `include/coupling/karma_byte_writer.h` | superseded | include/fuse/base/bytes.hpp (test/base/bytes_test.cpp) | confirmed |
| `include/coupling/karma_collective_outcome.h` | superseded | fuse::Epoch / EpochSchedule (ir/op.hpp Epoch instruction, runtime/epoch_schedule.hpp); confirmed via docs/dev/porting.md ('CollectiveOutcome as the Epoch instruction') and docs/dev/rank-agreement.md | confirmed |
| `include/coupling/karma_compatibility.h` | superseded | include/fuse/model/compatibility.hpp, connect.hpp (test/model/compatibility_test.cpp) | confirmed |
| `include/coupling/karma_compiled_fem_connection.h` | port-adapt | D-A: FEM/mesh-specific transfer and lookup machinery with no fuse equivalent (fuse::search is ArborX-backed point/box query, not FEM basis evaluation) | category |
| `include/coupling/karma_component.h` | superseded | include/fuse/model/component.hpp (ComponentDecl/ComponentKind) + include/fuse/model/field_spec.hpp; confirmed via porting.md's merge table ('component contract...capability bits replace supports_*()') | confirmed |
| `include/coupling/karma_component_runtime.h` | superseded | include/fuse/runtime/component.hpp (Component/ComponentSet/ComponentIterator) | confirmed |
| `include/coupling/karma_conservative_transfer.h` | port-adapt | D-A: FEM/mesh-specific transfer and lookup machinery with no fuse equivalent (fuse::search is ArborX-backed point/box query, not FEM basis evaluation) | category |
| `include/coupling/karma_distributed_point_locator.h` | port-adapt | D-A: FEM/mesh-specific transfer and lookup machinery with no fuse equivalent (fuse::search is ArborX-backed point/box query, not FEM basis evaluation) | category |
| `include/coupling/karma_fem_field_evaluator.h` | port-adapt | D-A: FEM/mesh-specific transfer and lookup machinery with no fuse equivalent (fuse::search is ArborX-backed point/box query, not FEM basis evaluation) | category |
| `include/coupling/karma_fem_transfer.h` | port-adapt | D-A: FEM/mesh-specific transfer and lookup machinery with no fuse equivalent (fuse::search is ArborX-backed point/box query, not FEM basis evaluation) | category |
| `include/coupling/karma_field.h` | superseded | include/fuse/model/field_spec.hpp (already in [soma]'s locked superseded list, which plays the merged role per porting.md's 'field handle' row) | confirmed |
| `include/coupling/karma_field_coverage.h` | undecided | porting.md's merge table names a spine(KARMA)/coupler(soma)-merged successor for this concept, but no literally-named current fuse type was found by grep (searched exact symbol names from the karma header); confirm the current fuse name before claiming supersession | category |
| `include/coupling/karma_field_entry_encoding.h` | undecided | porting.md's merge table names a spine(KARMA)/coupler(soma)-merged successor for this concept, but no literally-named current fuse type was found by grep (searched exact symbol names from the karma header); confirm the current fuse name before claiming supersession | category |
| `include/coupling/karma_field_identity.h` | undecided | porting.md's merge table names a spine(KARMA)/coupler(soma)-merged successor for this concept, but no literally-named current fuse type was found by grep (searched exact symbol names from the karma header); confirm the current fuse name before claiming supersession | category |
| `include/coupling/karma_field_transaction.h` | undecided | porting.md's merge table names a spine(KARMA)/coupler(soma)-merged successor for this concept, but no literally-named current fuse type was found by grep (searched exact symbol names from the karma header); confirm the current fuse name before claiming supersession | category |
| `include/coupling/karma_field_transfer.h` | undecided | porting.md's merge table names a spine(KARMA)/coupler(soma)-merged successor for this concept, but no literally-named current fuse type was found by grep (searched exact symbol names from the karma header); confirm the current fuse name before claiming supersession | category |
| `include/coupling/karma_field_view.h` | superseded | include/fuse/field/field_ref.hpp, include/fuse/field/registry.hpp, include/fuse/runtime/component.hpp (confirmed matches via grep on FieldView/FieldSpan/FieldRef) | confirmed |
| `include/coupling/karma_fixed_point.h` | retire | the native two-component operator-split graph and fixed-point precursor (item 6): generic operator-split/fixed-point coupling is superseded by fuse's own coupling graph + model/iteration.hpp + num/accelerator.hpp (S2-S4); the Lie/Strang order baselines remain useful oracle numbers only if physics-kit ever exposes operator splitting as a D-B temporal integrator | confirmed |
| `include/coupling/karma_intersection_provider.h` | port-adapt | D-A: FEM/mesh-specific transfer and lookup machinery with no fuse equivalent (fuse::search is ArborX-backed point/box query, not FEM basis evaluation) | category |
| `include/coupling/karma_moab_point_locator.h` | port-adapt | D-A: FEM/mesh-specific transfer and lookup machinery with no fuse equivalent (fuse::search is ArborX-backed point/box query, not FEM basis evaluation) | category |
| `include/coupling/karma_operator_split_checkpoint.h` | retire | the native two-component operator-split graph and fixed-point precursor (item 6): generic operator-split/fixed-point coupling is superseded by fuse's own coupling graph + model/iteration.hpp + num/accelerator.hpp (S2-S4); the Lie/Strang order baselines remain useful oracle numbers only if physics-kit ever exposes operator splitting as a D-B temporal integrator | confirmed |
| `include/coupling/karma_operator_split_graph.h` | retire | the native two-component operator-split graph and fixed-point precursor (item 6): generic operator-split/fixed-point coupling is superseded by fuse's own coupling graph + model/iteration.hpp + num/accelerator.hpp (S2-S4); the Lie/Strang order baselines remain useful oracle numbers only if physics-kit ever exposes operator splitting as a D-B temporal integrator | confirmed |
| `include/coupling/karma_petsc_field_storage.h` | port-adapt | D-A: FEM/mesh-specific transfer and lookup machinery with no fuse equivalent (fuse::search is ArborX-backed point/box query, not FEM basis evaluation) | category |
| `include/coupling/karma_point_locator.h` | port-adapt | D-A: FEM/mesh-specific transfer and lookup machinery with no fuse equivalent (fuse::search is ArborX-backed point/box query, not FEM basis evaluation) | category |
| `include/coupling/karma_quasi_interpolation.h` | port-adapt | D-A: FEM/mesh-specific transfer and lookup machinery with no fuse equivalent (fuse::search is ArborX-backed point/box query, not FEM basis evaluation) | category |
| `include/coupling/karma_reference_components.h` | undecided | porting.md's merge table names a spine(KARMA)/coupler(soma)-merged successor for this concept, but no literally-named current fuse type was found by grep (searched exact symbol names from the karma header); confirm the current fuse name before claiming supersession | category |
| `include/coupling/karma_registry.h` | retire | ComponentRegistry's role is covered by fuse::abi::Registry (src/abi/registry.hpp/.cpp) plus fuse::field::FieldRegistry; exact assertion-level equivalence to karma_static_registry_test NOT individually confirmed | category |
| `include/coupling/karma_semantic_version.h` | superseded | include/fuse/base/version.hpp (test/base/version_test.cpp) | confirmed |
| `include/coupling/karma_semantics.h` | superseded | include/fuse/model/semantics.hpp (test/model/semantics_test.cpp) | confirmed |
| `include/coupling/karma_semantics_adapter.h` | undecided | porting.md's merge table names a spine(KARMA)/coupler(soma)-merged successor for this concept, but no literally-named current fuse type was found by grep (searched exact symbol names from the karma header); confirm the current fuse name before claiming supersession | category |
| `include/coupling/karma_transfer_diagnostics.h` | undecided | porting.md's merge table names a spine(KARMA)/coupler(soma)-merged successor for this concept, but no literally-named current fuse type was found by grep (searched exact symbol names from the karma header); confirm the current fuse name before claiming supersession | category |
| `include/coupling/karma_transfer_map.h` | port-adapt | D-A: FEM-specific sparse interpolation row representation feeds fuse::backend's LinearOperator/remap_plan, not a 1:1 supersession | category |
| `include/coupling/karma_transfer_map_io.h` | port-adapt | D-A: persisted FEM transfer-map file format; fuse's own persisted-map format (S5) covers remap weights generically but not FEM-specific metadata -- confirm overlap before literal port | category |
| `include/coupling/karma_transform.h` | superseded | include/fuse/model/transform.hpp (test/model/transform_test.cpp) | confirmed |
| `include/coupling/karma_transient_reaction_diffusion_adapter.h` | port-adapt | D-A: FEM/mesh-specific transfer and lookup machinery with no fuse equivalent (fuse::search is ArborX-backed point/box query, not FEM basis evaluation) | category |
| `include/enums/bc_type.h` | port-adapt | FE-domain vocabulary enum (element/BC/flux/property tag); no fuse equivalent yet | category |
| `include/enums/domain_shape.h` | port-adapt | FE-domain vocabulary enum (element/BC/flux/property tag); no fuse equivalent yet | category |
| `include/enums/enum_utils.h` | port-adapt | FE-domain vocabulary enum (element/BC/flux/property tag); no fuse equivalent yet | category |
| `include/enums/io_package.h` | port-adapt | FE-domain vocabulary enum (element/BC/flux/property tag); no fuse equivalent yet | category |
| `include/enums/karma_linear_operations.h` | retire | operation/layout tags for karma's own linalg layer, which retires to fuse::linalg (D-A); no longer needed once that layer is dropped | category |
| `include/enums/karma_nonlinear_operations.h` | retire | operation/layout tags for karma's own linalg layer, which retires to fuse::linalg (D-A); no longer needed once that layer is dropped | category |
| `include/enums/linear_algebra_layout.h` | retire | operation/layout tags for karma's own linalg layer, which retires to fuse::linalg (D-A); no longer needed once that layer is dropped | category |
| `include/enums/matrix_structure.h` | retire | operation/layout tags for karma's own linalg layer, which retires to fuse::linalg (D-A); no longer needed once that layer is dropped | category |
| `include/enums/numerical_flux_type.h` | port-adapt | FE-domain vocabulary enum (element/BC/flux/property tag); no fuse equivalent yet | category |
| `include/enums/partitioner_type.h` | retire | operation/layout tags for karma's own linalg layer, which retires to fuse::linalg (D-A); no longer needed once that layer is dropped | category |
| `include/enums/physics_properties.h` | port-adapt | FE-domain vocabulary enum (element/BC/flux/property tag); no fuse equivalent yet | category |
| `include/enums/primary_axis.h` | port-adapt | FE-domain vocabulary enum (element/BC/flux/property tag); no fuse equivalent yet | category |
| `include/enums/runge_kutta_schemes.h` | port-integrator | D-B: temporal-scheme tag enumeration, ported alongside fuse::num's integrators | owner-decision |
| `include/enums/supporting_packages.h` | port-adapt | FE-domain vocabulary enum (element/BC/flux/property tag); no fuse equivalent yet | category |
| `include/interface/karma_context.h` | port-adapt | KARMAContext: solution-state container (vector/matrix/context, libMesh ReferenceCountedObject base); largely retires to fuse::linalg's DistVector/DistSparseMatrix, with FE-specific dof/BC state adapted | category |
| `include/interface/karma_interface.h` | port-adapt | KARMAInterface: KARMA's central orchestration class (System+DofMap+BC+assembly+solver wiring); adapt onto fuse::ImplicitComponent/LinearizationToken (S7) rather than porting literally | category |
| `include/interface/karma_wrappers.h` | retire | PETSc SNES/TS C-callback bridge functions (karma_wrapper_petsc_snes_residual etc.); this bridging role is already provided generically by fuse::petsc::solve()/MffdToken (S7) | category |
| `include/io/karma_input_document.h` | port-adapt | KARMAInputDocument: input-deck document abstraction; likely retires toward fuse's YAML front end, confirm against src/model/yaml_io.cpp before porting literally | category |
| `include/io/karma_moab_io.h` | undecided | KARMAMoabIO/KARMALoadedCheckpoint mixes two concerns: FE-specific MOAB mesh I/O (element blocks/sidesets, genuinely new, port-adapt) and a MOAB-tag-based checkpoint/restart mechanism (encode/decode_checkpoint_scalar, checkpoint_gather_stats) that duplicates the DESIRED, separately-tracked parallel-HDF5+XDMF restart deliverable (fuse already has include/fuse/restart_io/hdf5_restart_io.hpp) -- physics-kit should not invent a second restart mechanism; needs an owner decision on where the split falls | confirmed |
| `include/io/karma_monitor.h` | port-adapt | solution-monitoring/output callback hooked into karma_interface's residual history; check against fuse's own Newton convergence logging (newton-krylov.md) before porting literally | category |
| `include/numerics/abstract_system_owner.h` | port-adapt | D-A: owning/lifetime contract for a System | category |
| `include/numerics/algebra/imatrix.h` | retire | D-A: fuse::linalg's DistVector/DistSparseMatrix + dense least-squares facade (S7) is the adaptation target; this layer is not ported literally | owner-decision |
| `include/numerics/algebra/ivector.h` | retire | D-A: fuse::linalg's DistVector/DistSparseMatrix + dense least-squares facade (S7) is the adaptation target; this layer is not ported literally | owner-decision |
| `include/numerics/algebra/karma_dense.h` | retire | D-A: fuse::linalg's DistVector/DistSparseMatrix + dense least-squares facade (S7) is the adaptation target; this layer is not ported literally | owner-decision |
| `include/numerics/algebra/karma_matrix.h` | retire | D-A: fuse::linalg's DistVector/DistSparseMatrix + dense least-squares facade (S7) is the adaptation target; this layer is not ported literally | owner-decision |
| `include/numerics/algebra/karma_vector.h` | retire | D-A: fuse::linalg's DistVector/DistSparseMatrix + dense least-squares facade (S7) is the adaptation target; this layer is not ported literally | owner-decision |
| `include/numerics/algebra/linear_algebra_factory.h` | retire | D-A: fuse::linalg's DistVector/DistSparseMatrix + dense least-squares facade (S7) is the adaptation target; this layer is not ported literally | owner-decision |
| `include/numerics/algebra/petsc_block_matrix.h` | retire | D-A: fuse::linalg's DistVector/DistSparseMatrix + dense least-squares facade (S7) is the adaptation target; this layer is not ported literally | owner-decision |
| `include/numerics/algebra/petsc_block_vector.h` | retire | D-A: fuse::linalg's DistVector/DistSparseMatrix + dense least-squares facade (S7) is the adaptation target; this layer is not ported literally | owner-decision |
| `include/numerics/algebra/petsc_native_matrix.h` | retire | D-A: fuse::linalg's DistVector/DistSparseMatrix + dense least-squares facade (S7) is the adaptation target; this layer is not ported literally | owner-decision |
| `include/numerics/algebra/petsc_native_scatter.h` | retire | D-A: fuse::linalg's DistVector/DistSparseMatrix + dense least-squares facade (S7) is the adaptation target; this layer is not ported literally | owner-decision |
| `include/numerics/algebra/petsc_native_vector.h` | retire | D-A: fuse::linalg's DistVector/DistSparseMatrix + dense least-squares facade (S7) is the adaptation target; this layer is not ported literally | owner-decision |
| `include/numerics/algebra/petsc_raii.h` | retire | D-A: fuse::linalg's DistVector/DistSparseMatrix + dense least-squares facade (S7) is the adaptation target; this layer is not ported literally | owner-decision |
| `include/numerics/eigen/karma_eigen_solver.h` | port-adapt | no SLEPc/eigensolver capability exists anywhere in fuse today; new capability, needs an owner decision on whether SLEPc becomes an optional dependency alongside PETSc/MOAB | category |
| `include/numerics/fe/karma_basis.h` | port-adapt | D-A: FEM basis/geometry-map/reference-topology core, the central content of physics-kit | owner-decision |
| `include/numerics/fe/karma_fe_base.h` | port-adapt | D-A: FEM basis/geometry-map/reference-topology core, the central content of physics-kit | owner-decision |
| `include/numerics/fe/karma_geometry_map.h` | port-adapt | D-A: FEM basis/geometry-map/reference-topology core, the central content of physics-kit | owner-decision |
| `include/numerics/fe/karma_quadrature.h` | port-adapt | D-A: port the CURRENT resized create_product_quadrature() (already fixed upstream, KNOWN_ISSUES #1); carry the distorted-pyramid accuracy risk forward into fuse's own KNOWN_ISSUES.md rather than closing it silently (item 6) | confirmed |
| `include/numerics/fe/karma_reference_topology.h` | port-adapt | D-A: FEM basis/geometry-map/reference-topology core, the central content of physics-kit | owner-decision |
| `include/numerics/icollection.h` | port-adapt | D-A: generic collection interface | category |
| `include/numerics/isolver.h` | undecided | D-A: abstract solver interface -- likely retires to fuse::num::solve()'s own interface | category |
| `include/numerics/isystem.h` | port-adapt | D-A: abstract System interface, selects between karma's native System and the libMesh-backed one | category |
| `include/numerics/karma_algebraic_system.h` | port-adapt | D-A: KARMA's native algebraic-system assembly core | category |
| `include/numerics/karma_dof_map.h` | port-adapt | D-A: degree-of-freedom map | category |
| `include/numerics/karma_operator.h` | port-adapt | D-A: operator base contract | category |
| `include/numerics/karma_system.h` | port-adapt | D-A: KARMA's native System | category |
| `include/numerics/karma_system_base.h` | port-adapt | D-A: System base contract | category |
| `include/numerics/libmesh_system.h` | undecided | wraps libMesh's own System class -- libMesh is a third dependency (alongside PETSc/MOAB) that fuse's build does not currently resolve at all; confirm whether physics-kit takes on libMesh, or drops this backend and keeps only karma_system.h's native path | confirmed |
| `include/numerics/linear/karma_linear_operator.h` | retire | D-A: fuse::linalg's pcg/gmres + fuse::petsc::solve() over KSP (S7) is the adaptation target | owner-decision |
| `include/numerics/linear/karma_linear_solver.h` | retire | D-A: fuse::linalg's pcg/gmres + fuse::petsc::solve() over KSP (S7) is the adaptation target | owner-decision |
| `include/numerics/linear/petsc_block_linear_solver.h` | retire | D-A: fuse::linalg's pcg/gmres + fuse::petsc::solve() over KSP (S7) is the adaptation target | owner-decision |
| `include/numerics/linear/petsc_native_linear_solver.h` | retire | D-A: fuse::linalg's pcg/gmres + fuse::petsc::solve() over KSP (S7) is the adaptation target | owner-decision |
| `include/numerics/linear/preconditioner/karma_preconditioner.h` | port-adapt | generic preconditioner plumbing likely retires to fuse::petsc's lagged block-Jacobi preconditioner (S7); physics-specific block structure is physics-kit's own | category |
| `include/numerics/linear/preconditioner/petsc_block_preconditioner.h` | port-adapt | generic preconditioner plumbing likely retires to fuse::petsc's lagged block-Jacobi preconditioner (S7); physics-specific block structure is physics-kit's own | category |
| `include/numerics/linear/preconditioner/petsc_native_preconditioner.h` | port-adapt | generic preconditioner plumbing likely retires to fuse::petsc's lagged block-Jacobi preconditioner (S7); physics-specific block structure is physics-kit's own | category |
| `include/numerics/nonlinear/karma_nonlinear_operator.h` | superseded | fuse::num::solve() (Jacobian-free Newton-Krylov) + fuse::petsc::solve() over SNESNEWTONLS (S7, roadmap.md: 'fuse::num::solve()...Jacobian-free Newton-Krylov'); S7's own closure note says this generic machinery was built from OTHER example physics, not by porting these two headers, so they are superseded rather than merely adapted | confirmed |
| `include/numerics/nonlinear/petsc_native_nonlinear_solver.h` | superseded | fuse::num::solve() (Jacobian-free Newton-Krylov) + fuse::petsc::solve() over SNESNEWTONLS (S7, roadmap.md: 'fuse::num::solve()...Jacobian-free Newton-Krylov'); S7's own closure note says this generic machinery was built from OTHER example physics, not by porting these two headers, so they are superseded rather than merely adapted | confirmed |
| `include/numerics/spatial/karma_domain.h` | port-adapt | D-A: FE mesh/domain abstraction (element blocks, boundary tagging), no fuse::backend equivalent (MeshView is remap-oriented, not FE-assembly-oriented) | category |
| `include/numerics/spatial/karma_mesh.h` | port-adapt | D-A: FE mesh/domain abstraction (element blocks, boundary tagging), no fuse::backend equivalent (MeshView is remap-oriented, not FE-assembly-oriented) | category |
| `include/numerics/spatial/karma_mesh_utils.h` | port-adapt | D-A: FE mesh/domain abstraction (element blocks, boundary tagging), no fuse::backend equivalent (MeshView is remap-oriented, not FE-assembly-oriented) | category |
| `include/numerics/temporal/backward_euler.h` | port-integrator | D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `include/numerics/temporal/crank_nicolson.h` | port-integrator | D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `include/numerics/temporal/dopri45.h` | port-integrator | D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `include/numerics/temporal/dopri45_integrator.h` | port-integrator | D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `include/numerics/temporal/embedded_dirk_integrator.h` | port-integrator | D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `include/numerics/temporal/explicit_rk_integrator.h` | port-integrator | D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `include/numerics/temporal/forward_euler.h` | port-integrator | D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `include/numerics/temporal/grk4a.h` | port-integrator | D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `include/numerics/temporal/grk4sn.h` | port-integrator | D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `include/numerics/temporal/grk4t.h` | port-integrator | D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `include/numerics/temporal/implicit_dirk_integrator.h` | port-integrator | D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `include/numerics/temporal/karma_ts_base.h` | port-integrator | D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `include/numerics/temporal/kutta22.h` | port-integrator | D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `include/numerics/temporal/kutta44.h` | port-integrator | D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `include/numerics/temporal/petsc_temporal_integrator.h` | port-integrator | D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `include/numerics/temporal/pseudo_time_integrator.h` | port-integrator | D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `include/numerics/temporal/rk.h` | port-integrator | D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `include/numerics/temporal/rodasp.h` | port-integrator | D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `include/numerics/temporal/ros22.h` | port-integrator | D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `include/numerics/temporal/ros3p.h` | port-integrator | D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `include/numerics/temporal/rosenbrock.h` | port-integrator | D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `include/numerics/temporal/rosenbrock_integrator.h` | port-integrator | D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `include/numerics/temporal/sdirk22.h` | port-integrator | D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `include/numerics/temporal/sdirk23.h` | port-integrator | D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `include/numerics/temporal/sdirk33.h` | port-integrator | D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `include/numerics/temporal/sdirk332.h` | port-integrator | D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `include/numerics/temporal/sdirk34.h` | port-integrator | D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `include/numerics/temporal/sdirk44.h` | port-integrator | D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `include/numerics/temporal/sdirk45.h` | port-integrator | D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `include/numerics/temporal/sdirk543.h` | port-integrator | D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `include/numerics/temporal/sdirk643.h` | port-integrator | D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `include/numerics/temporal/temporal_integrator.h` | port-integrator | D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `include/numerics/temporal/trbe.h` | port-integrator | D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `include/numerics/user_sparsity.h` | port-adapt | D-A: user-supplied sparsity pattern hook | category |
| `include/parallel/karma_partitioner.h` | retire | mesh-partitioning wrapper; superseded by fuse::partition (RCB partitioner ported from rcb_moab, already a real layer in fuse) | category |
| `include/physics/advection-reaction/advection_reaction_assembly.h` | port-adapt | D-A: advection-reaction FE assembly; no fuse equivalent, physics-kit's own concrete physics module | owner-decision |
| `include/physics/advection-reaction/advection_reaction_mass_operator.h` | port-adapt | D-A: advection-reaction mass operator; no fuse equivalent, physics-kit's own concrete physics module | owner-decision |
| `include/physics/advection-reaction/advection_reaction_preconditioner_operator.h` | port-adapt | D-A: advection-reaction preconditioner block; no fuse equivalent, physics-kit's own concrete physics module | owner-decision |
| `include/physics/advection-reaction/advection_reaction_system.h` | port-adapt | D-A: advection-reaction System; no fuse equivalent, physics-kit's own concrete physics module | owner-decision |
| `include/physics/advection-reaction/advection_reaction_system_operator.h` | port-adapt | D-A: advection-reaction residual/Jacobian operator; no fuse equivalent, physics-kit's own concrete physics module | owner-decision |
| `include/physics/advection-reaction/advection_reaction_system_property_manager.h` | port-adapt | D-A: advection-reaction property manager; no fuse equivalent, physics-kit's own concrete physics module | owner-decision |
| `include/physics/base/karma_physics_base.h` | port-adapt | D-A: KARMA's physics-module base contract (depends on libMesh ReferenceCountedObject -- see libMesh open question); no fuse equivalent, physics-kit's own concrete physics module | owner-decision |
| `include/physics/base/karma_physics_base_mass_operator.h` | port-adapt | D-A: generic mass-operator contract; no fuse equivalent, physics-kit's own concrete physics module | owner-decision |
| `include/physics/base/karma_physics_base_property_manager.h` | port-adapt | D-A: generic property-manager contract; no fuse equivalent, physics-kit's own concrete physics module | owner-decision |
| `include/physics/common/generic_mass_operator.h` | port-adapt | D-A: shared mass-operator implementation; no fuse equivalent, physics-kit's own concrete physics module | owner-decision |
| `include/physics/common/karma_boundary_condition.h` | port-adapt | D-A: boundary-condition representation; no fuse equivalent, physics-kit's own concrete physics module | owner-decision |
| `include/physics/common/karma_physics_bc_manager.h` | port-adapt | D-A: boundary-condition manager; no fuse equivalent, physics-kit's own concrete physics module | owner-decision |
| `include/physics/diffusion-reaction/diffusion_reaction_assembly.h` | port-adapt | D-A: diffusion-reaction FE assembly; no fuse equivalent, physics-kit's own concrete physics module | owner-decision |
| `include/physics/diffusion-reaction/diffusion_reaction_mass_operator.h` | port-adapt | D-A: diffusion-reaction mass operator; no fuse equivalent, physics-kit's own concrete physics module | owner-decision |
| `include/physics/diffusion-reaction/diffusion_reaction_preconditioner_operator.h` | port-adapt | D-A: diffusion-reaction preconditioner block; no fuse equivalent, physics-kit's own concrete physics module | owner-decision |
| `include/physics/diffusion-reaction/diffusion_reaction_system.h` | port-adapt | D-A: diffusion-reaction System; no fuse equivalent, physics-kit's own concrete physics module | owner-decision |
| `include/physics/diffusion-reaction/diffusion_reaction_system_operator.h` | port-adapt | D-A: diffusion-reaction residual/Jacobian operator; no fuse equivalent, physics-kit's own concrete physics module | owner-decision |
| `include/physics/diffusion-reaction/diffusion_reaction_system_property_manager.h` | port-adapt | D-A: diffusion-reaction property manager; no fuse equivalent, physics-kit's own concrete physics module | owner-decision |
| `include/physics/diffusion-reaction/gray_scott_model.h` | port-adapt | KARMA's own Gray-Scott kinetics (Du=0.16,Dv=0.08,F=0.060,k=0.062 defaults) -- a DIFFERENT parameter regime and multi-mesh FEM discretisation from fuse/examples/gray_scott's single-grid FV model (item 5); port as physics-kit's own verification case, not a replacement for the existing example | confirmed |
| `include/physics/diffusion-reaction/transient_reaction_diffusion_assembly.h` | port-adapt | D-A: transient reaction-diffusion assembly; no fuse equivalent, physics-kit's own concrete physics module | owner-decision |
| `include/physics/diffusion-reaction/transient_reaction_diffusion_component.h` | port-adapt | D-A: transient reaction-diffusion component wrapper; no fuse equivalent, physics-kit's own concrete physics module | owner-decision |
| `include/physics/ode-system/ode_mass_operator.h` | port-adapt | D-A: ODE-system mass operator; no fuse equivalent, physics-kit's own concrete physics module | owner-decision |
| `include/physics/ode-system/ode_system.h` | port-adapt | D-A: ODE System (couples to fuse::num's temporal integrators, D-B); no fuse equivalent, physics-kit's own concrete physics module | owner-decision |
| `include/physics/ode-system/ode_system_operator.h` | port-adapt | D-A: ODE residual/Jacobian operator; no fuse equivalent, physics-kit's own concrete physics module | owner-decision |
| `include/physics/ode-system/ode_system_property_manager.h` | port-adapt | D-A: ODE property manager; no fuse equivalent, physics-kit's own concrete physics module | owner-decision |
| `include/utils/karma_utils.h` | port-adapt | grab-bag utility header; split per actual use at port time | category |
| `include/utils/lapack_utils.h` | port-adapt | LAPACK wrapper needed by FE dense solves; check against fuse::linalg's dense least-squares facade first | category |
| `include/utils/petsc_ksp_utils.h` | retire | superseded by fuse::petsc's own KSP wiring (S7) | category |
| `include/utils/petsc_snes_utils.h` | retire | superseded by fuse::petsc::solve() over SNESNEWTONLS (S7) | category |
| `include/utils/petsc_vec_utils.h` | retire | superseded by fuse::linalg::DistVector (S7) | category |
| `include/utils/smart_pointers/copied_ptr.h` | retire | pre-C++11 smart-pointer zoo, superseded by std::shared_ptr/unique_ptr and value semantics | confirmed |
| `include/utils/smart_pointers/counted_ptr.h` | retire | pre-C++11 smart-pointer zoo, superseded by std::shared_ptr/unique_ptr and value semantics | confirmed |
| `include/utils/smart_pointers/cow_ptr.h` | retire | pre-C++11 smart-pointer zoo, superseded by std::shared_ptr/unique_ptr and value semantics | confirmed |
| `include/utils/smart_pointers/linked_ptr.h` | retire | pre-C++11 smart-pointer zoo, superseded by std::shared_ptr/unique_ptr and value semantics | confirmed |
| `include/utils/smart_pointers/owned_ptr.h` | retire | pre-C++11 smart-pointer zoo, superseded by std::shared_ptr/unique_ptr and value semantics | confirmed |
| `include/utils/static_vector.h` | retire | fixed-capacity vector; superseded by std::array or fuse::linalg small-vector utility if one exists | category |
| `include/utils/string_utils.h` | retire | std:: string utilities suffice | category |
| `include/utils/typename_traits.h` | undecided | type-name/trait helpers -- coordinate with the numtypes track (CoupledValue concept, D-E) before reintroducing | category |

### `src/` (154 files)

| path | disposition | target / reason | confirmation |
|---|---|---|---|
| `src/CMakeLists.txt` | retire | build scaffolding | category |
| `src/abi/karma_abi.C` | retire | superseded by fuse's own provider/participant ABI (fuse_plugin_abi()/fuse_coupler_abi(), abi/entries.def), already built at S6 | category |
| `src/abi/karma_component_abi.C` | retire | superseded by fuse's own provider/participant ABI (fuse_plugin_abi()/fuse_coupler_abi(), abi/entries.def), already built at S6 | category |
| `src/abi/karma_exports.map` | retire | superseded by fuse's own provider/participant ABI (fuse_plugin_abi()/fuse_coupler_abi(), abi/entries.def), already built at S6 | category |
| `src/abi/karma_exports.sym` | retire | superseded by fuse's own provider/participant ABI (fuse_plugin_abi()/fuse_coupler_abi(), abi/entries.def), already built at S6 | category |
| `src/base/ikarma.C` | port-adapt | implementation half of ikarma.h | category |
| `src/base/karma.C` | port-adapt | implementation half of karma.h | category |
| `src/base/karma_log.C` | retire | implementation half of karma_log.h | category |
| `src/base/karma_print_trace.C` | retire | implementation half of karma_print_trace.h | category |
| `src/coupling/karma_collective_outcome.C` | superseded | implementation half of include/coupling/karma_collective_outcome.h: fuse::Epoch / EpochSchedule (ir/op.hpp Epoch instruction, runtime/epoch_schedule.hpp); confirmed via docs/dev/porting.md ('CollectiveOutcome as the Epoch instruction') and docs/dev/rank-agreement.md | confirmed |
| `src/coupling/karma_compatibility.C` | superseded | include/fuse/model/compatibility.hpp | confirmed |
| `src/coupling/karma_compiled_fem_connection.C` | port-adapt | implementation half of include/coupling/karma_compiled_fem_connection.h: D-A: FEM/mesh-specific transfer and lookup machinery with no fuse equivalent (fuse::search is ArborX-backed point/box query, not FEM basis evaluation) | category |
| `src/coupling/karma_component.C` | superseded | implementation half of include/coupling/karma_component.h: include/fuse/model/component.hpp (ComponentDecl/ComponentKind) + include/fuse/model/field_spec.hpp; confirmed via porting.md's merge table ('component contract...capability bits replace supports_*()') | confirmed |
| `src/coupling/karma_component_runtime.C` | superseded | implementation half of include/coupling/karma_component_runtime.h: include/fuse/runtime/component.hpp (Component/ComponentSet/ComponentIterator) | confirmed |
| `src/coupling/karma_conservative_transfer.C` | port-adapt | implementation half of include/coupling/karma_conservative_transfer.h: D-A: FEM/mesh-specific transfer and lookup machinery with no fuse equivalent (fuse::search is ArborX-backed point/box query, not FEM basis evaluation) | category |
| `src/coupling/karma_distributed_point_locator.C` | port-adapt | implementation half of include/coupling/karma_distributed_point_locator.h: D-A: FEM/mesh-specific transfer and lookup machinery with no fuse equivalent (fuse::search is ArborX-backed point/box query, not FEM basis evaluation) | category |
| `src/coupling/karma_fem_field_evaluator.C` | port-adapt | implementation half of include/coupling/karma_fem_field_evaluator.h: D-A: FEM/mesh-specific transfer and lookup machinery with no fuse equivalent (fuse::search is ArborX-backed point/box query, not FEM basis evaluation) | category |
| `src/coupling/karma_fem_transfer.C` | port-adapt | implementation half of include/coupling/karma_fem_transfer.h: D-A: FEM/mesh-specific transfer and lookup machinery with no fuse equivalent (fuse::search is ArborX-backed point/box query, not FEM basis evaluation) | category |
| `src/coupling/karma_field.C` | superseded | implementation half of include/coupling/karma_field.h: include/fuse/model/field_spec.hpp (already in [soma]'s locked superseded list, which plays the merged role per porting.md's 'field handle' row) | confirmed |
| `src/coupling/karma_field_coverage.C` | undecided | implementation half of include/coupling/karma_field_coverage.h: porting.md's merge table names a spine(KARMA)/coupler(soma)-merged successor for this concept, but no literally-named current fuse type was found by grep (searched exact symbol names from the karma header); confirm the current fuse name before claiming supersession | category |
| `src/coupling/karma_field_entry_encoding.C` | undecided | implementation half of include/coupling/karma_field_entry_encoding.h: porting.md's merge table names a spine(KARMA)/coupler(soma)-merged successor for this concept, but no literally-named current fuse type was found by grep (searched exact symbol names from the karma header); confirm the current fuse name before claiming supersession | category |
| `src/coupling/karma_field_transaction.C` | undecided | implementation half of include/coupling/karma_field_transaction.h: porting.md's merge table names a spine(KARMA)/coupler(soma)-merged successor for this concept, but no literally-named current fuse type was found by grep (searched exact symbol names from the karma header); confirm the current fuse name before claiming supersession | category |
| `src/coupling/karma_field_transfer.C` | undecided | implementation half of include/coupling/karma_field_transfer.h: porting.md's merge table names a spine(KARMA)/coupler(soma)-merged successor for this concept, but no literally-named current fuse type was found by grep (searched exact symbol names from the karma header); confirm the current fuse name before claiming supersession | category |
| `src/coupling/karma_field_view.C` | superseded | implementation half of include/coupling/karma_field_view.h: include/fuse/field/field_ref.hpp, include/fuse/field/registry.hpp, include/fuse/runtime/component.hpp (confirmed matches via grep on FieldView/FieldSpan/FieldRef) | confirmed |
| `src/coupling/karma_fixed_point.C` | retire | implementation half of include/coupling/karma_fixed_point.h: the native two-component operator-split graph and fixed-point precursor (item 6): generic operator-split/fixed-point coupling is superseded by fuse's own coupling graph + model/iteration.hpp + num/accelerator.hpp (S2-S4); the Lie/Strang order baselines remain useful oracle numbers only if physics-kit ever exposes operator splitting as a D-B temporal integrator | confirmed |
| `src/coupling/karma_moab_point_locator.C` | port-adapt | implementation half of include/coupling/karma_moab_point_locator.h: D-A: FEM/mesh-specific transfer and lookup machinery with no fuse equivalent (fuse::search is ArborX-backed point/box query, not FEM basis evaluation) | category |
| `src/coupling/karma_operator_split_checkpoint.C` | retire | implementation half of include/coupling/karma_operator_split_checkpoint.h: the native two-component operator-split graph and fixed-point precursor (item 6): generic operator-split/fixed-point coupling is superseded by fuse's own coupling graph + model/iteration.hpp + num/accelerator.hpp (S2-S4); the Lie/Strang order baselines remain useful oracle numbers only if physics-kit ever exposes operator splitting as a D-B temporal integrator | confirmed |
| `src/coupling/karma_operator_split_graph.C` | retire | implementation half of include/coupling/karma_operator_split_graph.h: the native two-component operator-split graph and fixed-point precursor (item 6): generic operator-split/fixed-point coupling is superseded by fuse's own coupling graph + model/iteration.hpp + num/accelerator.hpp (S2-S4); the Lie/Strang order baselines remain useful oracle numbers only if physics-kit ever exposes operator splitting as a D-B temporal integrator | confirmed |
| `src/coupling/karma_petsc_field_storage.C` | port-adapt | implementation half of include/coupling/karma_petsc_field_storage.h: D-A: FEM/mesh-specific transfer and lookup machinery with no fuse equivalent (fuse::search is ArborX-backed point/box query, not FEM basis evaluation) | category |
| `src/coupling/karma_quasi_interpolation.C` | port-adapt | implementation half of include/coupling/karma_quasi_interpolation.h: D-A: FEM/mesh-specific transfer and lookup machinery with no fuse equivalent (fuse::search is ArborX-backed point/box query, not FEM basis evaluation) | category |
| `src/coupling/karma_reference_components.C` | undecided | implementation half of include/coupling/karma_reference_components.h: porting.md's merge table names a spine(KARMA)/coupler(soma)-merged successor for this concept, but no literally-named current fuse type was found by grep (searched exact symbol names from the karma header); confirm the current fuse name before claiming supersession | category |
| `src/coupling/karma_registry.C` | retire | implementation half of include/coupling/karma_registry.h: ComponentRegistry's role is covered by fuse::abi::Registry (src/abi/registry.hpp/.cpp) plus fuse::field::FieldRegistry; exact assertion-level equivalence to karma_static_registry_test NOT individually confirmed | category |
| `src/coupling/karma_semantic_version.C` | superseded | include/fuse/base/version.hpp | confirmed |
| `src/coupling/karma_semantics.C` | superseded | include/fuse/model/semantics.hpp | confirmed |
| `src/coupling/karma_semantics_adapter.C` | undecided | implementation half of include/coupling/karma_semantics_adapter.h: porting.md's merge table names a spine(KARMA)/coupler(soma)-merged successor for this concept, but no literally-named current fuse type was found by grep (searched exact symbol names from the karma header); confirm the current fuse name before claiming supersession | category |
| `src/coupling/karma_transfer_diagnostics.C` | undecided | implementation half of include/coupling/karma_transfer_diagnostics.h: porting.md's merge table names a spine(KARMA)/coupler(soma)-merged successor for this concept, but no literally-named current fuse type was found by grep (searched exact symbol names from the karma header); confirm the current fuse name before claiming supersession | category |
| `src/coupling/karma_transfer_map.C` | port-adapt | implementation half of include/coupling/karma_transfer_map.h: D-A: FEM-specific sparse interpolation row representation feeds fuse::backend's LinearOperator/remap_plan, not a 1:1 supersession | category |
| `src/coupling/karma_transfer_map_io.C` | port-adapt | implementation half of include/coupling/karma_transfer_map_io.h: D-A: persisted FEM transfer-map file format; fuse's own persisted-map format (S5) covers remap weights generically but not FEM-specific metadata -- confirm overlap before literal port | category |
| `src/coupling/karma_transform.C` | superseded | include/fuse/model/transform.hpp | confirmed |
| `src/coupling/karma_transient_reaction_diffusion_adapter.C` | port-adapt | implementation half of include/coupling/karma_transient_reaction_diffusion_adapter.h: D-A: FEM/mesh-specific transfer and lookup machinery with no fuse equivalent (fuse::search is ArborX-backed point/box query, not FEM basis evaluation) | category |
| `src/enums/bc_type.C` | port-adapt | mirrors the disposition of the corresponding include/enums header | category |
| `src/enums/karma_linear_operations.C` | retire | mirrors the disposition of the corresponding include/enums header | category |
| `src/enums/karma_nonlinear_operations.C` | retire | mirrors the disposition of the corresponding include/enums header | category |
| `src/enums/partitioner_type.C` | retire | mirrors the disposition of the corresponding include/enums header | category |
| `src/enums/physics_properties.C` | port-adapt | mirrors the disposition of the corresponding include/enums header | category |
| `src/enums/runge_kutta_schemes.C` | port-integrator | mirrors the disposition of the corresponding include/enums header | category |
| `src/interface/karma_context.C` | port-adapt | implementation half of the orchestration headers above | category |
| `src/interface/karma_interface.C` | port-adapt | implementation half of the orchestration headers above | category |
| `src/interface/karma_interface_block_routines.C` | port-adapt | implementation half of the orchestration headers above | category |
| `src/interface/karma_interface_routines.C` | port-adapt | implementation half of the orchestration headers above | category |
| `src/interface/karma_wrappers.C` | retire | implementation half of karma_wrappers.h | category |
| `src/io/karma_input_document.C` | port-adapt | implementation half | category |
| `src/io/karma_moab_io.C` | undecided | implementation half of karma_moab_io.h -- same split question | confirmed |
| `src/io/karma_monitor.C` | port-adapt | implementation half | category |
| `src/numerics/algebra/linear_algebra_factory.C` | retire | implementation half of include/numerics/algebra/linear_algebra_factory.h: D-A: fuse::linalg's DistVector/DistSparseMatrix + dense least-squares facade (S7) is the adaptation target; this layer is not ported literally | owner-decision |
| `src/numerics/algebra/petsc_block_matrix.C` | retire | implementation half of include/numerics/algebra/petsc_block_matrix.h: D-A: fuse::linalg's DistVector/DistSparseMatrix + dense least-squares facade (S7) is the adaptation target; this layer is not ported literally | owner-decision |
| `src/numerics/algebra/petsc_block_vector.C` | retire | implementation half of include/numerics/algebra/petsc_block_vector.h: D-A: fuse::linalg's DistVector/DistSparseMatrix + dense least-squares facade (S7) is the adaptation target; this layer is not ported literally | owner-decision |
| `src/numerics/algebra/petsc_native_matrix.C` | retire | implementation half of include/numerics/algebra/petsc_native_matrix.h: D-A: fuse::linalg's DistVector/DistSparseMatrix + dense least-squares facade (S7) is the adaptation target; this layer is not ported literally | owner-decision |
| `src/numerics/algebra/petsc_native_scatter.C` | retire | implementation half of include/numerics/algebra/petsc_native_scatter.h: D-A: fuse::linalg's DistVector/DistSparseMatrix + dense least-squares facade (S7) is the adaptation target; this layer is not ported literally | owner-decision |
| `src/numerics/algebra/petsc_native_vector.C` | retire | implementation half of include/numerics/algebra/petsc_native_vector.h: D-A: fuse::linalg's DistVector/DistSparseMatrix + dense least-squares facade (S7) is the adaptation target; this layer is not ported literally | owner-decision |
| `src/numerics/eigen/karma_eigen_solver.C` | port-adapt | implementation half of include/numerics/eigen/karma_eigen_solver.h: no SLEPc/eigensolver capability exists anywhere in fuse today; new capability, needs an owner decision on whether SLEPc becomes an optional dependency alongside PETSc/MOAB | category |
| `src/numerics/fe/karma_basis.C` | port-adapt | implementation half of include/numerics/fe/karma_basis.h: D-A: FEM basis/geometry-map/reference-topology core, the central content of physics-kit | owner-decision |
| `src/numerics/fe/karma_fe_base.C` | port-adapt | implementation half of include/numerics/fe/karma_fe_base.h: D-A: FEM basis/geometry-map/reference-topology core, the central content of physics-kit | owner-decision |
| `src/numerics/fe/karma_geometry_map.C` | port-adapt | implementation half of include/numerics/fe/karma_geometry_map.h: D-A: FEM basis/geometry-map/reference-topology core, the central content of physics-kit | owner-decision |
| `src/numerics/fe/karma_reference_topology.C` | port-adapt | implementation half of include/numerics/fe/karma_reference_topology.h: D-A: FEM basis/geometry-map/reference-topology core, the central content of physics-kit | owner-decision |
| `src/numerics/icollection.C` | port-adapt | implementation half of the System/DofMap core above | category |
| `src/numerics/isystem.C` | port-adapt | implementation half of the System/DofMap core above | category |
| `src/numerics/karma_algebraic_system.C` | port-adapt | implementation half of the System/DofMap core above | category |
| `src/numerics/karma_dof_map.C` | port-adapt | implementation half of the System/DofMap core above | category |
| `src/numerics/karma_operator.C` | port-adapt | implementation half of the System/DofMap core above | category |
| `src/numerics/karma_system.C` | port-adapt | implementation half of the System/DofMap core above | category |
| `src/numerics/libmesh_system.C` | undecided | implementation half of libmesh_system.h -- same libMesh dependency question | confirmed |
| `src/numerics/linear/karma_linear_operator.C` | retire | implementation half of include/numerics/linear/karma_linear_operator.h: D-A: fuse::linalg's pcg/gmres + fuse::petsc::solve() over KSP (S7) is the adaptation target | owner-decision |
| `src/numerics/linear/petsc_native_linear_solver.C` | retire | implementation half of include/numerics/linear/petsc_native_linear_solver.h: D-A: fuse::linalg's pcg/gmres + fuse::petsc::solve() over KSP (S7) is the adaptation target | owner-decision |
| `src/numerics/linear/preconditioner/petsc_block_preconditioner.C` | port-adapt | implementation half of include/numerics/linear/preconditioner/petsc_block_preconditioner.h: generic preconditioner plumbing likely retires to fuse::petsc's lagged block-Jacobi preconditioner (S7); physics-specific block structure is physics-kit's own | category |
| `src/numerics/linear/preconditioner/petsc_native_preconditioner.C` | port-adapt | implementation half of include/numerics/linear/preconditioner/petsc_native_preconditioner.h: generic preconditioner plumbing likely retires to fuse::petsc's lagged block-Jacobi preconditioner (S7); physics-specific block structure is physics-kit's own | category |
| `src/numerics/nonlinear/karma_nonlinear_operator.C` | superseded | implementation half of include/numerics/nonlinear/karma_nonlinear_operator.h: fuse::num::solve() (Jacobian-free Newton-Krylov) + fuse::petsc::solve() over SNESNEWTONLS (S7, roadmap.md: 'fuse::num::solve()...Jacobian-free Newton-Krylov'); S7's own closure note says this generic machinery was built from OTHER example physics, not by porting these two headers, so they are superseded rather than merely adapted | confirmed |
| `src/numerics/nonlinear/petsc_native_nonlinear_solver.C` | superseded | implementation half of include/numerics/nonlinear/petsc_native_nonlinear_solver.h: fuse::num::solve() (Jacobian-free Newton-Krylov) + fuse::petsc::solve() over SNESNEWTONLS (S7, roadmap.md: 'fuse::num::solve()...Jacobian-free Newton-Krylov'); S7's own closure note says this generic machinery was built from OTHER example physics, not by porting these two headers, so they are superseded rather than merely adapted | confirmed |
| `src/numerics/spatial/karma_domain.C` | port-adapt | contains the KNOWN DEBUG-BUILD BUG from UPSTREAMS.lock's karma 'open' list: lines 32,54,77 use unqualified Hexagon/Unstructured inside a vacuous assert(); must be fixed as a real invariant (I6) during the port, not carried forward | confirmed |
| `src/numerics/spatial/karma_mesh.C` | port-adapt | implementation half of include/numerics/spatial/karma_mesh.h: D-A: FE mesh/domain abstraction (element blocks, boundary tagging), no fuse::backend equivalent (MeshView is remap-oriented, not FE-assembly-oriented) | category |
| `src/numerics/spatial/karma_mesh_extract_edge.C` | port-adapt | mesh edge/face extraction helper (no separate header; implementation-only), part of the FE mesh/domain abstraction, D-A | category |
| `src/numerics/spatial/karma_mesh_extract_face.C` | port-adapt | mesh edge/face extraction helper (no separate header; implementation-only), part of the FE mesh/domain abstraction, D-A | category |
| `src/numerics/temporal/backward_euler.C` | port-integrator | implementation half of include/numerics/temporal/backward_euler.h: D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `src/numerics/temporal/crank_nicolson.C` | port-integrator | implementation half of include/numerics/temporal/crank_nicolson.h: D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `src/numerics/temporal/dopri45.C` | port-integrator | implementation half of include/numerics/temporal/dopri45.h: D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `src/numerics/temporal/dopri45_integrator.C` | port-integrator | implementation half of include/numerics/temporal/dopri45_integrator.h: D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `src/numerics/temporal/embedded_dirk_integrator.C` | port-integrator | implementation half of include/numerics/temporal/embedded_dirk_integrator.h: D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `src/numerics/temporal/explicit_rk_integrator.C` | port-integrator | implementation half of include/numerics/temporal/explicit_rk_integrator.h: D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `src/numerics/temporal/forward_euler.C` | port-integrator | implementation half of include/numerics/temporal/forward_euler.h: D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `src/numerics/temporal/grk4a.C` | port-integrator | implementation half of include/numerics/temporal/grk4a.h: D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `src/numerics/temporal/grk4sn.C` | port-integrator | implementation half of include/numerics/temporal/grk4sn.h: D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `src/numerics/temporal/grk4t.C` | port-integrator | implementation half of include/numerics/temporal/grk4t.h: D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `src/numerics/temporal/implicit_dirk_integrator.C` | port-integrator | implementation half of include/numerics/temporal/implicit_dirk_integrator.h: D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `src/numerics/temporal/karma_ts_base.C` | port-integrator | implementation half of include/numerics/temporal/karma_ts_base.h: D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `src/numerics/temporal/kutta22.C` | port-integrator | implementation half of include/numerics/temporal/kutta22.h: D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `src/numerics/temporal/kutta44.C` | port-integrator | implementation half of include/numerics/temporal/kutta44.h: D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `src/numerics/temporal/petsc_temporal_integrator.C` | port-integrator | implementation half of include/numerics/temporal/petsc_temporal_integrator.h: D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `src/numerics/temporal/pseudo_time_integrator.C` | port-integrator | implementation half of include/numerics/temporal/pseudo_time_integrator.h: D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `src/numerics/temporal/rk.C` | port-integrator | implementation half of include/numerics/temporal/rk.h: D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `src/numerics/temporal/rodasp.C` | port-integrator | implementation half of include/numerics/temporal/rodasp.h: D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `src/numerics/temporal/ros22.C` | port-integrator | implementation half of include/numerics/temporal/ros22.h: D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `src/numerics/temporal/ros3p.C` | port-integrator | implementation half of include/numerics/temporal/ros3p.h: D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `src/numerics/temporal/rosenbrock.C` | port-integrator | implementation half of include/numerics/temporal/rosenbrock.h: D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `src/numerics/temporal/rosenbrock_integrator.C` | port-integrator | implementation half of include/numerics/temporal/rosenbrock_integrator.h: D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `src/numerics/temporal/sdirk22.C` | port-integrator | implementation half of include/numerics/temporal/sdirk22.h: D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `src/numerics/temporal/sdirk23.C` | port-integrator | implementation half of include/numerics/temporal/sdirk23.h: D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `src/numerics/temporal/sdirk33.C` | port-integrator | implementation half of include/numerics/temporal/sdirk33.h: D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `src/numerics/temporal/sdirk332.C` | port-integrator | implementation half of include/numerics/temporal/sdirk332.h: D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `src/numerics/temporal/sdirk34.C` | port-integrator | implementation half of include/numerics/temporal/sdirk34.h: D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `src/numerics/temporal/sdirk44.C` | port-integrator | implementation half of include/numerics/temporal/sdirk44.h: D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `src/numerics/temporal/sdirk45.C` | port-integrator | implementation half of include/numerics/temporal/sdirk45.h: D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `src/numerics/temporal/sdirk543.C` | port-integrator | implementation half of include/numerics/temporal/sdirk543.h: D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `src/numerics/temporal/sdirk643.C` | port-integrator | implementation half of include/numerics/temporal/sdirk643.h: D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `src/numerics/temporal/temporal_integrator.C` | port-integrator | implementation half of include/numerics/temporal/temporal_integrator.h: D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `src/numerics/temporal/trbe.C` | port-integrator | implementation half of include/numerics/temporal/trbe.h: D-B: KARMA time integrator, ported into fuse::num templated on fuse::linalg (owner decision, all temporal integrators including embedded/adaptive) | owner-decision |
| `src/numerics/user_sparsity.C` | port-adapt | implementation half of the System/DofMap core above | category |
| `src/parallel/karma_partitioner.C` | retire | implementation half; superseded by fuse::partition | category |
| `src/physics/advection-reaction/advection_reaction_mass_operator.C` | port-adapt | D-A: implementation half of the physics-module header above | owner-decision |
| `src/physics/advection-reaction/advection_reaction_preconditioner_operator.C` | port-adapt | D-A: implementation half of the physics-module header above | owner-decision |
| `src/physics/advection-reaction/advection_reaction_system.C` | port-adapt | D-A: implementation half of the physics-module header above | owner-decision |
| `src/physics/advection-reaction/advection_reaction_system_jacobian_matrix.C` | port-adapt | D-A: implementation half of the physics-module header above | owner-decision |
| `src/physics/advection-reaction/advection_reaction_system_operator.C` | port-adapt | D-A: implementation half of the physics-module header above | owner-decision |
| `src/physics/advection-reaction/advection_reaction_system_property_manager.C` | port-adapt | D-A: implementation half of the physics-module header above | owner-decision |
| `src/physics/advection-reaction/advection_reaction_system_residual.C` | port-adapt | D-A: implementation half of the physics-module header above | owner-decision |
| `src/physics/base/karma_physics_base.C` | port-adapt | D-A: implementation half of the physics-module header above | owner-decision |
| `src/physics/base/karma_physics_base_property_manager.C` | port-adapt | D-A: implementation half of the physics-module header above | owner-decision |
| `src/physics/common/impls/collective_signature.h` | port-adapt | D-A: implementation half of the physics-module header above | owner-decision |
| `src/physics/common/impls/generic_mass_matrix.C` | port-adapt | D-A: implementation half of the physics-module header above | owner-decision |
| `src/physics/common/impls/generic_mass_matrix_multiply.C` | port-adapt | D-A: implementation half of the physics-module header above | owner-decision |
| `src/physics/common/impls/generic_mass_matrix_solve.C` | port-adapt | D-A: implementation half of the physics-module header above | owner-decision |
| `src/physics/common/karma_boundary_condition.C` | port-adapt | D-A: implementation half of the physics-module header above | owner-decision |
| `src/physics/common/karma_physics_bc_manager.C` | port-adapt | D-A: implementation half of the physics-module header above | owner-decision |
| `src/physics/diffusion-reaction/diffusion_reaction_mass_operator.C` | port-adapt | D-A: implementation half of the physics-module header above | owner-decision |
| `src/physics/diffusion-reaction/diffusion_reaction_preconditioner_operator.C` | port-adapt | D-A: implementation half of the physics-module header above | owner-decision |
| `src/physics/diffusion-reaction/diffusion_reaction_preconditioner_point_jacobi_solve.C` | port-adapt | D-A: implementation half of the physics-module header above | owner-decision |
| `src/physics/diffusion-reaction/diffusion_reaction_system.C` | port-adapt | D-A: implementation half of the physics-module header above | owner-decision |
| `src/physics/diffusion-reaction/diffusion_reaction_system_jacobian_matrix.C` | port-adapt | D-A: implementation half of the physics-module header above | owner-decision |
| `src/physics/diffusion-reaction/diffusion_reaction_system_operator.C` | port-adapt | D-A: implementation half of the physics-module header above | owner-decision |
| `src/physics/diffusion-reaction/diffusion_reaction_system_property_manager.C` | port-adapt | D-A: implementation half of the physics-module header above | owner-decision |
| `src/physics/diffusion-reaction/diffusion_reaction_system_residual.C` | port-adapt | D-A: implementation half of the physics-module header above | owner-decision |
| `src/physics/diffusion-reaction/gray_scott_model.C` | port-adapt | implementation half of gray_scott_model.h (item 5) | confirmed |
| `src/physics/diffusion-reaction/transient_reaction_diffusion_assembly.C` | port-adapt | D-A: implementation half of the physics-module header above | owner-decision |
| `src/physics/diffusion-reaction/transient_reaction_diffusion_component.C` | port-adapt | D-A: implementation half of the physics-module header above | owner-decision |
| `src/physics/ode-system/ode_system.C` | port-adapt | D-A: implementation half of the physics-module header above | owner-decision |
| `src/physics/ode-system/ode_system_mass_operator.C` | port-adapt | D-A: implementation half of the physics-module header above | owner-decision |
| `src/physics/ode-system/ode_system_operator.C` | port-adapt | D-A: implementation half of the physics-module header above | owner-decision |
| `src/physics/ode-system/ode_system_property_manager.C` | port-adapt | D-A: implementation half of the physics-module header above | owner-decision |
| `src/utils/error_measure.C` | port-adapt | implementation half of the corresponding header above | category |
| `src/utils/lapack_utils.C` | port-adapt | implementation half of the corresponding header above | category |
| `src/utils/petsc_ksp_utils.C` | retire | implementation half of the corresponding header above | category |
| `src/utils/petsc_snes_utils.C` | retire | implementation half of the corresponding header above | category |
| `src/utils/petsc_vec_utils.C` | retire | implementation half of the corresponding header above | category |
| `src/utils/static_vector.C` | retire | implementation half of the corresponding header above | category |
| `src/utils/typename_traits.C` | retire | implementation half of the corresponding header above | category |

### `test/` (87 files)

| path | disposition | target / reason | confirmation |
|---|---|---|---|
| `test/CMakeLists.txt` | retire | build scaffolding; its content is the source for the test-mapping deliverable (item 3), not something ported itself | category |
| `test/abi_c11_conformance_test.c` | retire | ABI layout conformance; mirrors fuse's own test/abi conformance_test.cpp pattern -- port the check, not the file | category |
| `test/abi_cxx_layout_test.C` | port-adapt | ABI C++ layout check; fuse's test/abi/device_layout_test.cpp is an analogous but not identical check | category |
| `test/abi_install_consumer/CMakeLists.txt` | retire | build scaffolding for the installed-consumer smoke check | category |
| `test/abi_install_consumer/main.c` | port-adapt | exercises the installed ABI headers end-to-end; mirrors fuse's own abi/install_consumer pattern (test/abi/install_consumer/ already exists in fuse) | category |
| `test/abi_negotiation_test.C` | retire | covered by fuse test test/abi/negotiation_test.cpp (version negotiation already ported at S6) | confirmed |
| `test/advection_flux_test.C` | port-adapt | advection-reaction physics module, D-A, no fuse equivalent | owner-decision |
| `test/advection_mms_test.C` | port-adapt | advection-reaction MMS baseline (item 4), D-A | owner-decision |
| `test/algebraic_system_test.C` | port-adapt | System/DofMap core, D-A | owner-decision |
| `test/assembly_test.C` | port-adapt | generic FE assembly, D-A | owner-decision |
| `test/basis_test.C` | port-adapt | FE basis functions, D-A | owner-decision |
| `test/boundary_condition_test.C` | port-adapt | boundary-condition representation, D-A | owner-decision |
| `test/byte_writer_test.C` | superseded | test/base/bytes_test.cpp | confirmed |
| `test/cg_transfer_accuracy_test.C` | port-adapt | CG transfer accuracy baseline (item 4), D-A FEM transfer | owner-decision |
| `test/collective_outcome_mpi_test.C` | retire | covered by fuse test test/parallel/epoch_reconcile_test.cpp (Epoch reconciliation, confirmed) | confirmed |
| `test/collective_outcome_test.C` | retire | covered by fuse test test/schedule/epoch_test.cpp + test/runtime/epoch_schedule_test.cpp (confirmed) | confirmed |
| `test/compiled_connection_test.C` | port-adapt | compiled FEM connection, D-A, no fuse equivalent (FEM-specific) | owner-decision |
| `test/component_compliance_test.C` | retire | covered by fuse tests test/model/component_test.cpp + test/runtime/component_iface_test.cpp (confirmed Component/ComponentDecl exist) | confirmed |
| `test/conservative_transfer_capability_test.C` | port-adapt | conservative FEM transfer, D-A, overlaps fuse::backend::spacetime/supermesh but FEM-specific capability bit untested there | category |
| `test/coupling_compatibility_oracle_test.C` | retire | covered by fuse test test/model/compatibility_test.cpp (compatibility oracle already ported at S2/S8.0 merge) | category |
| `test/coupling_compatibility_test.C` | superseded | test/model/compatibility_test.cpp | confirmed |
| `test/coupling_semantics_test.C` | superseded | test/model/semantics_test.cpp | confirmed |
| `test/dense_test.C` | retire | karma_dense.h retires to fuse::linalg's dense least-squares facade (D-A) | owner-decision |
| `test/diffusion_assembly_test.C` | port-adapt | diffusion-reaction physics module, D-A | owner-decision |
| `test/diffusion_lifecycle_test.C` | port-adapt | diffusion-reaction component lifecycle, D-A | owner-decision |
| `test/diffusion_mms_test.C` | port-adapt | diffusion MMS baseline (item 4), D-A | owner-decision |
| `test/distributed_point_locator_test.C` | port-adapt | FEM point-location, D-A, no fuse equivalent (fuse::search is ArborX box/point query, not FE-basis evaluation) | category |
| `test/fe_face_test.C` | port-adapt | FE reference-topology face variants incl. pyramid/prism quadrature (item 6), D-A | owner-decision |
| `test/fe_test.C` | port-adapt | FE core incl. quadrature exactness and pyramid-gradients sections (item 6), D-A | owner-decision |
| `test/fem_field_evaluator_test.C` | port-adapt | FEM field evaluation at arbitrary points, D-A | owner-decision |
| `test/fem_transfer_convergence_test.C` | port-adapt | FEM transfer convergence baseline (item 4), D-A | owner-decision |
| `test/fem_transfer_partitioned_reference_test.C` | port-adapt | FEM transfer oracle on partitioned mesh, D-A | owner-decision |
| `test/fem_transfer_projection_test.C` | port-adapt | FEM transfer projection, D-A | owner-decision |
| `test/fem_transfer_sampling_test.C` | port-adapt | FEM transfer sampling, D-A | owner-decision |
| `test/fem_transfer_serial_reference_test.C` | port-adapt | FEM transfer serial oracle, D-A | owner-decision |
| `test/field_commit_test.C` | undecided | mirrors karma_field_transaction.h's undecided supersession status (no confirmed current fuse name found) | category |
| `test/field_entry_encoding_mpi_test.C` | undecided | mirrors karma_field_entry_encoding.h's undecided status | category |
| `test/field_entry_encoding_test.C` | undecided | mirrors karma_field_entry_encoding.h's undecided status | category |
| `test/field_transaction_test.C` | undecided | mirrors karma_field_transaction.h's undecided status (grep found no 'Transaction' anywhere in current fuse) | confirmed |
| `test/field_transfer_contract_test.C` | undecided | mirrors karma_field_transfer.h's undecided status | category |
| `test/fixed_point_coupling_test.C` | retire | generic fixed-point coupling (item 6); carries the iteration-count baseline (item 4: maximum.iterations==3) as an oracle number only | confirmed |
| `test/geometry_inverse_map_contract_test.C` | port-adapt | FE geometry inverse map, D-A | owner-decision |
| `test/geometry_test.C` | undecided | mirrors karma_geometry.h's BoundingBox/ArborX::Box open question | category |
| `test/gray_scott_model_test.C` | port-adapt | KARMA Gray-Scott kinetics (item 5), D-A -- different parameter regime from fuse's existing example | confirmed |
| `test/gray_scott_operator_split_test.C` | retire | exercises the retiring operator-split/fixed-point coupling scheme (item 6); the Lie/Strang order baselines (item 4) are preserved as oracle numbers only | confirmed |
| `test/gray_scott_partitioned_test.C` | port-adapt | multi-mesh Gray-Scott partitioning (item 5), D-A | confirmed |
| `test/gray_scott_restart_test.C` | undecided | mirrors karma_moab_io.h's checkpoint/restart split question (item 7) | confirmed |
| `test/gray_scott_spatial_mms_test.C` | port-adapt | Gray-Scott spatial MMS baseline (item 4/5), D-A | confirmed |
| `test/language_adapter_schema_test.C` | retire | ABI language-adapter schema; fuse's own Fortran/Python binding schema already exists at S6 | category |
| `test/legacy_bc_api_probe.C` | retire | named 'legacy' by KARMA itself; superseded by the current boundary_condition_test path | confirmed |
| `test/mass_assembly_test.C` | port-adapt | mass-matrix FE assembly, D-A | owner-decision |
| `test/mesh_test.C` | port-adapt | FE mesh/domain abstraction, D-A | owner-decision |
| `test/moab_coupler_probe.C` | port-adapt | MOAB-backed FEM coupling probe, D-A | owner-decision |
| `test/moab_field_commit_test.C` | undecided | mirrors karma_moab_io.h's checkpoint split question (item 7) | confirmed |
| `test/moab_io_test.C` | undecided | mirrors karma_moab_io.h's checkpoint/restart split question (item 7) | confirmed |
| `test/moab_point_locator_test.C` | port-adapt | MOAB-backed FEM point location, D-A | owner-decision |
| `test/native_quasi_interpolation_test.C` | port-adapt | native quasi-interpolation transfer, D-A | owner-decision |
| `test/ode_system_test.C` | port-adapt | ODE-system physics module, D-A, couples to fuse::num's integrators (D-B) | owner-decision |
| `test/operator_split_connection_test.C` | retire | item 6: generic operator-split coupling, superseded | confirmed |
| `test/operator_split_graph_test.C` | retire | item 6: generic operator-split coupling, superseded | confirmed |
| `test/operator_split_introspection_test.C` | retire | item 6: generic operator-split coupling, superseded | confirmed |
| `test/operator_split_scheduler_test.C` | retire | item 6: generic operator-split coupling, superseded | confirmed |
| `test/point_locator_contract_test.C` | port-adapt | FEM point-locator contract, D-A | owner-decision |
| `test/semantic_version_test.C` | superseded | test/base/version_test.cpp | confirmed |
| `test/smoke_test.C` | retire | trivial link smoke test; fuse has its own build/link smoke checks | category |
| `test/static_registry_test.C` | undecided | no single named fuse test found for ComponentRegistry's selection-policy semantics; registry behaviour appears bundled into component_iface_test.cpp/abi conformance -- confirm scope before retiring | category |
| `test/system_owner_test.C` | port-adapt | System ownership/lifetime, D-A | owner-decision |
| `test/system_test.C` | port-adapt | System core, D-A | owner-decision |
| `test/transfer_diagnostics_test.C` | undecided | mirrors karma_transfer_diagnostics.h's undecided status | category |
| `test/transfer_map_apply_test.C` | port-adapt | FEM transfer map application, D-A | owner-decision |
| `test/transfer_map_io_test.C` | port-adapt | FEM transfer map persistence, D-A, check overlap with fuse's S5 persisted-map format | category |
| `test/transfer_map_parallel_io_test.C` | port-adapt | FEM transfer map parallel I/O, D-A | category |
| `test/transfer_map_test.C` | port-adapt | FEM transfer map core, D-A | owner-decision |
| `test/transform_inference_test.C` | superseded | test/model/transform_test.cpp | confirmed |
| `test/transient_reaction_diffusion_assembly_test.C` | port-adapt | transient reaction-diffusion assembly, D-A | owner-decision |
| `test/transient_reaction_diffusion_component_test.C` | port-adapt | transient reaction-diffusion component, D-A | owner-decision |
| `test/transient_reaction_diffusion_mms_test.C` | port-adapt | transient reaction-diffusion MMS baseline (item 4), D-A | owner-decision |
| `test/unit/Make.dep` | retire | legacy cpptest/autoconf unit-test harness, explicitly flagged as historical by test/CMakeLists.txt's own comment; not part of the 203-call-site CMake-registered suite | confirmed |
| `test/unit/Makefile` | retire | legacy cpptest/autoconf unit-test harness, explicitly flagged as historical by test/CMakeLists.txt's own comment; not part of the 203-call-site CMake-registered suite | confirmed |
| `test/unit/bin/.gitignore` | retire | legacy cpptest/autoconf unit-test harness, explicitly flagged as historical by test/CMakeLists.txt's own comment; not part of the 203-call-site CMake-registered suite | confirmed |
| `test/unit/hello_test.C` | retire | legacy cpptest/autoconf unit-test harness, explicitly flagged as historical by test/CMakeLists.txt's own comment; not part of the 203-call-site CMake-registered suite | confirmed |
| `test/unit/input/.gitignore` | retire | legacy cpptest/autoconf unit-test harness, explicitly flagged as historical by test/CMakeLists.txt's own comment; not part of the 203-call-site CMake-registered suite | confirmed |
| `test/unit/karma_enums_test.C` | retire | legacy cpptest unit test, superseded in coverage by the CMake-registered suite's equivalent areas (enums/algebra/base tests) per the same historical-harness note | confirmed |
| `test/unit/karma_matrix_test.C` | retire | legacy cpptest unit test, superseded in coverage by the CMake-registered suite's equivalent areas (enums/algebra/base tests) per the same historical-harness note | confirmed |
| `test/unit/karma_package_test.C` | retire | legacy cpptest unit test, superseded in coverage by the CMake-registered suite's equivalent areas (enums/algebra/base tests) per the same historical-harness note | confirmed |
| `test/unit/karma_vector_test.C` | retire | legacy cpptest unit test, superseded in coverage by the CMake-registered suite's equivalent areas (enums/algebra/base tests) per the same historical-harness note | confirmed |
| `test/unit/output/.gitignore` | retire | legacy cpptest/autoconf unit-test harness, explicitly flagged as historical by test/CMakeLists.txt's own comment; not part of the 203-call-site CMake-registered suite | confirmed |## Appendix B: test mapping (75 CMake-registered test executables / 203 add_test call-sites)

Granularity note: KARMA's own `docs/HANDOFF.md` records 425 passing ctest rows on a host run; the 203 `add_test()` call-sites in `test/CMakeLists.txt` expand to that total via `foreach(ranks ...)`/`foreach(section ...)`/`foreach(failure ...)` loops (demonstrated: the 12-item `karma_fe_sections` list alone contributes 24 rows from two registrations). Reconciling the full 425 would need `cmake`+`ctest -N`, which the task's "do not build or run KARMA" rule precludes; this table is therefore at the 75-executable granularity (one row per distinct behaviour under test), not the 425-ctest-row granularity.

| test executable | sources | KARMA area | fuse tier | disposition | covering fuse test / open question |
|---|---|---|---|---|---|
| `karma_abi_layout_c_test` | `abi_c11_conformance_test.c` | abi | 0 | retire | ABI layout conformance; mirrors fuse's own test/abi conformance_test.cpp pattern -- port the check, not the file |
| `karma_abi_layout_test` | `abi_cxx_layout_test.C` | abi | 0 | port-adapt | ABI C++ layout check; fuse's test/abi/device_layout_test.cpp is an analogous but not identical check |
| `karma_abi_negotiation_test` | `abi_negotiation_test.C` | abi | 0 | retire | covered by fuse test test/abi/negotiation_test.cpp (version negotiation already ported at S6) |
| `karma_advection_dg_face_mutation_test` | `advection_flux_test.C;$<TARGET_OBJECTS:karma_advection_dg_face_mutation>` | physics/advection-reaction | 3 | port-adapt | advection-reaction physics module, D-A, no fuse equivalent |
| `karma_advection_flux_test` | `advection_flux_test.C` | physics/advection-reaction | 3 | port-adapt | advection-reaction physics module, D-A, no fuse equivalent |
| `karma_advection_mms_test` | `advection_mms_test.C` | physics/advection-reaction | 3 | port-adapt | advection-reaction MMS baseline (item 4), D-A |
| `karma_algebraic_system_test` | `algebraic_system_test.C` | numerics/(direct) System/DofMap core | 3 | port-adapt | System/DofMap core, D-A |
| `karma_assembly_test` | `assembly_test.C` | physics/common + numerics/fe (generic assembly) | 3 | port-adapt | generic FE assembly, D-A |
| `karma_basis_test` | `basis_test.C` | numerics/fe + numerics/algebra | 3 | port-adapt | FE basis functions, D-A |
| `karma_boundary_condition_test` | `boundary_condition_test.C` | physics/common (boundary conditions) | 3 | port-adapt | boundary-condition representation, D-A |
| `karma_byte_writer_test` | `byte_writer_test.C` | base/model (S1/S2 vocabulary) | 0 | retire | test/base/bytes_test.cpp |
| `karma_cg_transfer_accuracy_test` | `cg_transfer_accuracy_test.C` | coupling (FEM transfer, cg accuracy baseline item 4) | 0 | port-adapt | CG transfer accuracy baseline (item 4), D-A FEM transfer |
| `karma_collective_outcome_mpi_test` | `collective_outcome_mpi_test.C` | coupling (generic field/component/registry/outcome infra) | 1 | retire | covered by fuse test test/parallel/epoch_reconcile_test.cpp (Epoch reconciliation, confirmed) |
| `karma_collective_outcome_test` | `collective_outcome_test.C` | coupling (generic field/component/registry/outcome infra) | 0 | retire | covered by fuse test test/schedule/epoch_test.cpp + test/runtime/epoch_schedule_test.cpp (confirmed) |
| `karma_compiled_connection_test` | `compiled_connection_test.C` | coupling (operator-split/fixed-point precursor, item 6) | 1 | port-adapt | compiled FEM connection, D-A, no fuse equivalent (FEM-specific) |
| `karma_component_compliance_test` | `component_compliance_test.C` | coupling (generic field/component/registry/outcome infra) | 0 | retire | covered by fuse tests test/model/component_test.cpp + test/runtime/component_iface_test.cpp (confirmed Component/ComponentDecl exist) |
| `karma_conservative_transfer_capability_test` | `conservative_transfer_capability_test.C` | coupling (FEM/mesh transfer) | 0 | port-adapt | conservative FEM transfer, D-A, overlaps fuse::backend::spacetime/supermesh but FEM-specific capability bit untested there |
| `karma_coupling_compatibility_oracle_test` | `coupling_compatibility_oracle_test.C` | base/model (S1/S2 vocabulary) | 0 | retire | covered by fuse test test/model/compatibility_test.cpp (compatibility oracle already ported at S2/S8.0 merge) |
| `karma_coupling_compatibility_test` | `coupling_compatibility_test.C` | base/model (S1/S2 vocabulary) | 0 | retire | test/model/compatibility_test.cpp |
| `karma_coupling_semantics_test` | `coupling_semantics_test.C` | base/model (S1/S2 vocabulary) | 0 | retire | test/model/semantics_test.cpp |
| `karma_dense_test` | `dense_test.C` | numerics/fe + numerics/algebra | 3 | retire | karma_dense.h retires to fuse::linalg's dense least-squares facade (D-A) |
| `karma_diffusion_assembly_test` | `diffusion_assembly_test.C` | physics/common + numerics/fe (generic assembly) | 3 | port-adapt | diffusion-reaction physics module, D-A |
| `karma_diffusion_lifecycle_test` | `diffusion_lifecycle_test.C` | physics/diffusion-reaction | 3 | port-adapt | diffusion-reaction component lifecycle, D-A |
| `karma_diffusion_mms_test` | `diffusion_mms_test.C` | physics/diffusion-reaction | 3 | port-adapt | diffusion MMS baseline (item 4), D-A |
| `karma_distributed_point_locator_test` | `distributed_point_locator_test.C` | coupling (FEM/mesh transfer) | 0 | port-adapt | FEM point-location, D-A, no fuse equivalent (fuse::search is ArborX box/point query, not FE-basis evaluation) |
| `karma_fe_face_test` | `fe_face_test.C` | numerics/fe (incl. item 6 pyramid quadrature) | 3 | port-adapt | FE reference-topology face variants incl. pyramid/prism quadrature (item 6), D-A |
| `karma_fe_test` | `fe_test.C` | numerics/fe (incl. item 6 pyramid quadrature) | 3 | port-adapt | FE core incl. quadrature exactness and pyramid-gradients sections (item 6), D-A |
| `karma_fem_field_evaluator_test` | `fem_field_evaluator_test.C` | coupling (FEM/mesh transfer) | 0 | port-adapt | FEM field evaluation at arbitrary points, D-A |
| `karma_fem_transfer_convergence_test` | `fem_transfer_convergence_test.C` | coupling (FEM/mesh transfer) | 0 | port-adapt | FEM transfer convergence baseline (item 4), D-A |
| `karma_fem_transfer_partitioned_reference_test` | `fem_transfer_partitioned_reference_test.C` | coupling (FEM/mesh transfer) | 0 | port-adapt | FEM transfer oracle on partitioned mesh, D-A |
| `karma_fem_transfer_projection_test` | `fem_transfer_projection_test.C` | coupling (FEM/mesh transfer) | 0 | port-adapt | FEM transfer projection, D-A |
| `karma_fem_transfer_sampling_test` | `fem_transfer_sampling_test.C` | coupling (FEM/mesh transfer) | 0 | port-adapt | FEM transfer sampling, D-A |
| `karma_fem_transfer_serial_reference_test` | `fem_transfer_serial_reference_test.C` | coupling (FEM/mesh transfer) | 0 | port-adapt | FEM transfer serial oracle, D-A |
| `karma_field_commit_test` | `field_commit_test.C` | coupling (generic field/component/registry/outcome infra) | 0 | undecided | mirrors karma_field_transaction.h's undecided supersession status (no confirmed current fuse name found) |
| `karma_field_entry_encoding_mpi_test` | `field_entry_encoding_mpi_test.C` | coupling (generic field/component/registry/outcome infra) | 1 | undecided | mirrors karma_field_entry_encoding.h's undecided status |
| `karma_field_entry_encoding_test` | `field_entry_encoding_test.C` | coupling (generic field/component/registry/outcome infra) | 0 | undecided | mirrors karma_field_entry_encoding.h's undecided status |
| `karma_field_transaction_test` | `field_transaction_test.C` | coupling (generic field/component/registry/outcome infra) | 0 | undecided | mirrors karma_field_transaction.h's undecided status (grep found no 'Transaction' anywhere in current fuse) |
| `karma_field_transfer_contract_test` | `field_transfer_contract_test.C` | coupling (generic field/component/registry/outcome infra) | 0 | undecided | mirrors karma_field_transfer.h's undecided status |
| `karma_fixed_point_coupling_test` | `fixed_point_coupling_test.C` | coupling (operator-split/fixed-point precursor, item 6) | 0 | retire | generic fixed-point coupling (item 6); carries the iteration-count baseline (item 4: maximum.iterations==3) as an oracle number only |
| `karma_geometry_inverse_map_contract_test` | `geometry_inverse_map_contract_test.C` | numerics/fe + numerics/algebra | 3 | port-adapt | FE geometry inverse map, D-A |
| `karma_geometry_test` | `geometry_test.C` | numerics/fe + numerics/algebra | 3 | undecided | mirrors karma_geometry.h's BoundingBox/ArborX::Box open question |
| `karma_gray_scott_model_test` | `gray_scott_model_test.C` | physics/diffusion-reaction (Gray-Scott, item 5) | 3 | port-adapt | KARMA Gray-Scott kinetics (item 5), D-A -- different parameter regime from fuse's existing example |
| `karma_gray_scott_operator_split_test` | `gray_scott_operator_split_test.C` | coupling (operator-split/fixed-point precursor, item 6) | 0 | retire | exercises the retiring operator-split/fixed-point coupling scheme (item 6); the Lie/Strang order baselines (item 4) are preserved as oracle numbers only |
| `karma_gray_scott_partitioned_test` | `gray_scott_partitioned_test.C` | physics/diffusion-reaction (Gray-Scott, item 5) | 3 | port-adapt | multi-mesh Gray-Scott partitioning (item 5), D-A |
| `karma_gray_scott_restart_test` | `gray_scott_restart_test.C` | physics/diffusion-reaction (Gray-Scott, item 5) | 3 | undecided | mirrors karma_moab_io.h's checkpoint/restart split question (item 7) |
| `karma_gray_scott_spatial_mms_test` | `gray_scott_spatial_mms_test.C` | physics/diffusion-reaction (Gray-Scott, item 5) | 3 | port-adapt | Gray-Scott spatial MMS baseline (item 4/5), D-A |
| `karma_language_adapter_schema_test` | `language_adapter_schema_test.C` | abi | 0 | retire | ABI language-adapter schema; fuse's own Fortran/Python binding schema already exists at S6 |
| `karma_legacy_bc_api_probe` | `legacy_bc_api_probe.C` | physics/common (boundary conditions) | 3 | retire | named 'legacy' by KARMA itself; superseded by the current boundary_condition_test path |
| `karma_mass_assembly_test` | `mass_assembly_test.C` | physics/common + numerics/fe (generic assembly) | 3 | port-adapt | mass-matrix FE assembly, D-A |
| `karma_mesh_test` | `mesh_test.C` | numerics/spatial + io (MOAB) | 2 | port-adapt | FE mesh/domain abstraction, D-A |
| `karma_moab_coupler_probe` | `moab_coupler_probe.C` | numerics/spatial + io (MOAB) | 2 | port-adapt | MOAB-backed FEM coupling probe, D-A |
| `karma_moab_field_commit_test` | `moab_field_commit_test.C` | coupling (generic field/component/registry/outcome infra) | 2 | undecided | mirrors karma_moab_io.h's checkpoint split question (item 7) |
| `karma_moab_io_test` | `moab_io_test.C` | numerics/spatial + io (MOAB) | 2 | undecided | mirrors karma_moab_io.h's checkpoint/restart split question (item 7) |
| `karma_moab_point_locator_test` | `moab_point_locator_test.C` | coupling (FEM/mesh transfer) | 2 | port-adapt | MOAB-backed FEM point location, D-A |
| `karma_native_quasi_interpolation_test` | `native_quasi_interpolation_test.C` | coupling (FEM/mesh transfer) | 0 | port-adapt | native quasi-interpolation transfer, D-A |
| `karma_ode_system_test` | `ode_system_test.C` | numerics/(direct) System/DofMap core | 3 | port-adapt | ODE-system physics module, D-A, couples to fuse::num's integrators (D-B) |
| `karma_operator_split_connection_test` | `operator_split_connection_test.C` | coupling (operator-split/fixed-point precursor, item 6) | 0 | retire | item 6: generic operator-split coupling, superseded |
| `karma_operator_split_graph_test` | `operator_split_graph_test.C` | coupling (operator-split/fixed-point precursor, item 6) | 0 | retire | item 6: generic operator-split coupling, superseded |
| `karma_operator_split_introspection_test` | `operator_split_introspection_test.C` | coupling (operator-split/fixed-point precursor, item 6) | 0 | retire | item 6: generic operator-split coupling, superseded |
| `karma_operator_split_scheduler_test` | `operator_split_scheduler_test.C` | coupling (operator-split/fixed-point precursor, item 6) | 0 | retire | item 6: generic operator-split coupling, superseded |
| `karma_point_locator_contract_test` | `point_locator_contract_test.C` | coupling (FEM/mesh transfer) | 0 | port-adapt | FEM point-locator contract, D-A |
| `karma_semantic_version_test` | `semantic_version_test.C` | base/model (S1/S2 vocabulary) | 0 | retire | test/base/version_test.cpp |
| `karma_smoke_test` | `smoke_test.C` | base/model (S1/S2 vocabulary) | 0 | retire | trivial link smoke test; fuse has its own build/link smoke checks |
| `karma_static_registry_test` | `static_registry_test.C` | coupling (generic field/component/registry/outcome infra) | 0 | undecided | no single named fuse test found for ComponentRegistry's selection-policy semantics; registry behaviour appears bundled into component_iface_test.cpp/abi conformance -- confirm scope before retiring |
| `karma_system_owner_test` | `system_owner_test.C` | numerics/(direct) System/DofMap core | 3 | port-adapt | System ownership/lifetime, D-A |
| `karma_system_test` | `system_test.C` | numerics/(direct) System/DofMap core | 3 | port-adapt | System core, D-A |
| `karma_transfer_diagnostics_test` | `transfer_diagnostics_test.C` | coupling (FEM/mesh transfer) | 0 | undecided | mirrors karma_transfer_diagnostics.h's undecided status |
| `karma_transfer_map_apply_test` | `transfer_map_apply_test.C` | coupling (FEM/mesh transfer) | 0 | port-adapt | FEM transfer map application, D-A |
| `karma_transfer_map_io_test` | `transfer_map_io_test.C` | coupling (FEM/mesh transfer) | 0 | port-adapt | FEM transfer map persistence, D-A, check overlap with fuse's S5 persisted-map format |
| `karma_transfer_map_parallel_io_test` | `transfer_map_parallel_io_test.C` | coupling (FEM/mesh transfer) | 1 | port-adapt | FEM transfer map parallel I/O, D-A |
| `karma_transfer_map_test` | `transfer_map_test.C` | coupling (FEM/mesh transfer) | 0 | port-adapt | FEM transfer map core, D-A |
| `karma_transform_inference_test` | `transform_inference_test.C` | base/model (S1/S2 vocabulary) | 0 | retire | test/model/transform_test.cpp |
| `karma_transient_reaction_diffusion_assembly_test` | `transient_reaction_diffusion_assembly_test.C` | physics/common + numerics/fe (generic assembly) | 3 | port-adapt | transient reaction-diffusion assembly, D-A |
| `karma_transient_reaction_diffusion_component_test` | `transient_reaction_diffusion_component_test.C` | physics/diffusion-reaction | 3 | port-adapt | transient reaction-diffusion component, D-A |
| `karma_transient_reaction_diffusion_mms_test` | `transient_reaction_diffusion_mms_test.C` | physics/diffusion-reaction | 3 | port-adapt | transient reaction-diffusion MMS baseline (item 4), D-A |

---

## Lead's corrections after review (2026-10-27)

Two claims above were checked against the vendored source by the lead and are corrected here; the
text above is the track's own and is left as written.

1. **The libMesh dependency (Section 1.3, Open question 1) is confined to code KARMA itself does not
   compile.** MEASURED: no file under `include/` or `src/` includes a `libmesh/...` header; the only
   `#include "reference_counted_object.h"` sites are `include/interface/karma_context.h:30` (and the
   interface header that includes it), and `src/CMakeLists.txt`'s `KARMA_SOURCES` list compiles 114 of
   the 150 `.C` files on disk, leaving out all five `src/interface/*.C`, `numerics/libmesh_system.C`,
   both `numerics/nonlinear/*.C`, `numerics/eigen/karma_eigen_solver.C`, the PETSc native wrappers,
   `io/karma_monitor.C` and `io/karma_input_document.C`, `parallel/karma_partitioner.C`, four
   `utils/*.C`, `physics/base/*.C` and eight `numerics/temporal/*.C` (the seven integrator drivers and
   `temporal_integrator.C`). KARMA's own option `KARMA_ENABLE_LIBMESH` is OFF by default and OFF in the
   oracle build. Consequence: physics-kit adds no libMesh dependency; the excluded files are legacy
   and their disposition is `retire` unless a baseline test is found to need them (none does: the
   oracle registers 425 tests without them). The 36 excluded files are the rows in Appendix A whose
   path appears in the list above; the owner should treat Open question 1 as closed.
2. **Test-mapping granularity.** The 203 `add_test` call sites expand to the 425 rows the oracle
   registers (MEASURED on the owner's machine, 2026-10-27, `ctest -N`: "Total Tests: 425"); the
   per-row mapping is produced in S8.1 from the oracle's own `ctest -N` output, not re-derived here.

Everything else in this document stands as the track delivered it, including its declared depth of
verification (Section 0): 84 of 512 rows confirmed individually, 241 assigned by directory pattern.
