# Restart-to-disk and field output: design (Stage A) and Slice 1 implementation record (Stage B)

STATUS: Sections 0-8 and the citation table below are the Stage A design as the owner accepted it
(design only at that point; HEAD `35229b28a86d065b49c09be21458a2795e2a64c9`), unchanged except for
the correction already noted inline in Section 2.3. Section 9, appended below the citation table,
is the Stage B Slice 1 implementation record: what Slice 1 actually built against the owner's five
decisions, measured findings Stage A did not anticipate, and the scope reductions declared for
this round. Every claim in both parts is a citation to `file:line` in the export it was written
against, not an assumption about what the code "should" do.

Owner decisions already made, not reopened here: the restart format is a custom HDF5 schema;
the mesh is written once and state is written per restart point; field data is keyed by global
id so a run can restart on a different rank count; component state is carried as an opaque,
versioned record; an XDMF (`.xmf`) sidecar lets ParaView/VisIt read the restart file directly;
a CF/UGRID NetCDF-4 writer is a second, output-only deliverable of this same stage.

## 0. Terminology

This document uses `docs/user/glossary.md`'s vocabulary throughout: *coupling window*,
*waveform*, *straddle*, *accelerator*, *rollback*. The one new term this document introduces is
**restart-to-disk**, used throughout to mean "write a run's full in-memory state as bytes a
later process reads back," deliberately distinct from the existing *rollback* checkpoint
(`Component::save_state()`/`restore_state()`, `include/fuse/runtime/component.hpp:374,385`),
which is an in-memory, single-slot checkpoint for re-advancing one iteration-region sweep and
has no file-format or cross-rank-count meaning. Conflating the two is the single most likely
naming mistake in an implementation of this design, because they share a verb ("checkpoint") in
casual speech; this document always qualifies which one it means.

---

## 1. State inventory

Read from the Runner, the iteration machinery, the accelerators and the four worked examples
the task names, not assumed. Each row: owner (class/member), distribution, whether a global-id
index exists for its index space TODAY, and size order.

### 1.1 The Runner's clock and schedule position

| Item | Owner | Distribution | Global id today? | Size |
|---|---|---|---|---|
| Period counter | `Runner::Impl::period_index` (`src/runtime/runner_impl.hpp:1045`, `std::int64_t period_index = 0;`) | Replicated (every rank keeps its own, required to agree -- `RunnerConfig::periods`, `include/fuse/runtime/runner.hpp:137`, is identical input on every rank) | N/A -- a scalar, not an index space | 8 bytes |
| Collective-trace ordinal | `Runner::Impl::collective_trace_ordinal` (`src/runtime/runner_impl.hpp:1046-1053`) | Per-rank, diagnostic only | N/A | 8 bytes |
| Position within the compiled instruction stream, for an externally-driven (ABI) participant | A suspended C++20 coroutine frame (`src/runtime/resume.hpp:1-45`; `Runner::resume_period()`, `include/fuse/runtime/runner.hpp:411`) | Per-rank, per-component | N/A -- not a value, a return address plus locals | unserializable in its current form |

**The schedule-position item is the one entry in this table that is not data at all.**
`resume_period()`'s own doc comment (`include/fuse/runtime/runner.hpp:389-404`) states it
"resumes the current period's instruction stream from wherever it last suspended," and
`src/runtime/resume.hpp`'s header comment explains why: an externally-driven component
(`Component::is_externally_driven()`, `include/fuse/runtime/component.hpp:474`) suspends a
hand-written coroutine mid-period at exactly the point its `advance()` would run, and resumes
later from `fuse_begin_step()`/`fuse_end_step()` (`src/abi/participant.cpp`). A coroutine frame
is a stack of local variables and a resume address inside a specific compiled binary; it is not
an object `advance()` has, and nothing in `fuse/model/descriptor.hpp` or anywhere else gives it
a serializable representation. **Consequence for this design: a restart point can only be taken
at a period boundary** (`Runner::run_period()` having returned, `include/fuse/runtime/runner.hpp:386`),
never while any rank has a suspended `resume_period()` coroutine outstanding. This must be a
collective refusal (own rule: "A collective path must never return on a locally computed
condition") checked and agreed before the write begins, not discovered mid-write.

### 1.2 Coupling-window state (`fuse::detail::WindowMemberState`)

Source: `include/fuse/runtime/window_coupler.hpp:182-196`.

| Member | Distribution | Global id today? | Size order |
|---|---|---|---|
| `clock` (Rational) -- "Simulated time this member has actually reached" | Per window member, replicated on the ranks that host that member (`WindowMemberConfig::hosted`, `src/runtime/runner_impl.hpp:1098-1104`) | N/A, scalar | 16 bytes |
| `history` -- one `WindowBoundaryHistory` (`std::vector<std::pair<Rational,double>>`) per element, "seeded with the initial condition" (`window_coupler.hpp:176-178`) | Per member, per element | **No.** Indexed by position in the member's own interface-vector element list, not by a mesh global id | O(elements x windows-to-date) |
| `straddle_pending`, `straddle_window_start`, `straddle_window_end` | Per member | N/A, scalars | 24 bytes |
| `last_window_waveform` -- one `std::unique_ptr<Waveform>` per element, `TemporalBasis::Linear` (`window_coupler.hpp:188-193`) | Per member, per element | **No**, same element-position indexing as `history` | O(elements x samples-per-window) |

`Waveform` itself (`include/fuse/num/waveform.hpp:133` ff.) owns its own sample buffer sized by
`capacity`/`basis`/`degree` at construction (`waveform.hpp:152-199`); its state is exactly the
ring of `(t, value)` samples the constructor's own table documents
(`waveform.hpp:81-85`), nothing more.

### 1.3 Accelerator state (Aitken, Steffensen, IQN-ILS/IQN-IMVJ)

Source: `include/fuse/num/accelerator.hpp` (interface) and `src/num/accel_relaxation.cpp`,
`src/num/accel_quasi_newton.cpp` (implementations).

The base interface is explicit that this state is per-iteration-region and stateful across
sweeps: `Accelerator::reset()` "Clears the per-window secant history" (`accelerator.hpp:56-58`),
and `columns()` "Retained secant columns; 0 for the relaxation methods, and 0 immediately after
`reset()`" (`accelerator.hpp:154-156`).

| Class | Member | Role | Global id today? | Size order |
|---|---|---|---|---|
| `AitkenAccel` | `omega0_`, `omega_` (`src/num/accel_relaxation.cpp:404-405`) | scalar dynamic relaxation factor | N/A | 16 bytes |
| `AitkenAccel` | `r_`, `r_prev_` (`accel_relaxation.cpp:407-408`) | previous/current residual, **one entry per stacked interface-vector position** | **No** -- position in the stacked vector `run_region_newton_window()` builds (`RegionBlock::newton_offset`, `src/runtime/runner_impl.hpp:549`), not a mesh global id | O(n) |
| `SteffensenAccel` | `omega0_`, `r_`, `r_prev_`, `x_old_prev_` (`accel_relaxation.cpp:489-493`) | per-component diagonal secant | same as above | O(n) |
| `QuasiNewtonAccel` (IQN-ILS/IQN-IMVJ) | `v_`, `w_` -- "n x cols_, column major, oldest column first" (`src/num/accel_quasi_newton.cpp:766-767`) | the retained secant pairs (V, W) | **No** -- row index is the stacked interface-vector position, column index is reuse-window age; neither is a global id | O(n x reuse) |
| `QuasiNewtonAccel` | `x_new_prev_`, `r_prev_` (`accel_quasi_newton.cpp:768-769`) | previous iterate/residual | same | O(n) |
| `QuasiNewtonAccel` | `blocks_` -- `std::vector<JacobianBlock>`, each holding `z`, `v` ("n x cols, column major," `accel_quasi_newton.cpp:92-94`) | IQN-IMVJ's folded Jacobian history across windows (`Accelerator::reset()`'s own comment: "IQN_IMVJ folds it [into blocks_] instead," `accelerator.hpp:56-58`) | **No**, same convention | O(n x reuse x blocks) |
| `QuasiNewtonAccel` | `alpha_`, `ls_a_`, `ls_b_`, `rho_`, `jrho_`, `probe_`, `jac_a_`, `jac_b_`, `jac_beta_` (`accel_quasi_newton.cpp:783-793`) | per-solve scratch, re-derived every call from the members above | N/A | O(n) each |

**This is the state inventory's central negative result, asked for explicitly by the task
brief: every accelerator buffer is indexed by position in the stacked interface vector
(`RegionBlock::offset`/`length`, `src/runtime/runner_impl.hpp`), which is an artifact of how
the current rank's own `RegionRuntime` concatenates its hosted blocks -- it has no relationship
to a mesh global id and is not even stable across a rank-count change for the SAME graph**,
because the set of blocks a rank hosts, and therefore the stacked vector's own layout, is a
function of `WindowMemberConfig::hosted`/`consumer_hosted` (`src/runtime/runner_impl.hpp:1098,
1127` -- the D19 fix this file's own comment narrates at length), which depends on the
rank-to-component placement. A different rank count placing the same components differently
produces a stacked vector with different length and different block order on every rank. An
accelerator's `v_`/`w_`/`blocks_` columns from a 2-rank run have **no element-wise
correspondence** to the columns a 3-rank run would have built -- the IQN history is a function
of the iteration path, which a re-partitioned run does not reproduce anyway (see Section 7.ii).
This is why a rank-count-independent restart cannot and need not attempt to carry this history
across a rank-count change -- see Section 3's Option B discussion and Section 7.ii's exact
statement of what can be compared.

### 1.4 Newton state

Source: `include/fuse/num/newton_krylov.hpp`, `src/runtime/iteration.cpp`.

**Measured, not assumed: the "lagged" block-Jacobi preconditioner and the Eisenstat-Walker
forcing-term history do NOT persist across periods in the current implementation.** Both call
sites that build a `BlockJacobiPreconditioner` (`src/runtime/iteration.cpp:2519` and `:3016`,
`std::optional<num::BlockJacobiPreconditioner> bjp;`) declare it as a **local stack variable**
inside the region-advance function (`run_region_newton()`/`run_region_newton_window()`), rebuilt
from the components' current `LinearizationToken`s (`iteration.cpp:2489-2520`) every time that
function runs -- once per region per period. `RegionRuntime` (`src/runtime/runner_impl.hpp:555`
ff.) holds only setup-time, graph-derived data for a Newton region (`newton_n`, `newton_scale`,
`newton_direct_members`, ..., `runner_impl.hpp:677-813`) -- diagnostics and layout, not a live
solver object. The "lag" (`preconditioner_lag`, rebuilt "every `preconditioner_lag` Newton
steps," `newton_krylov.hpp:28-30`) counts Newton steps *within one call*, starting over at zero
every period. **Consequence: there is nothing here for restart to serialize.** A restart that
resumes at a period boundary (the only kind this design supports, Section 1.1) finds the next
period's Newton solve building its own fresh preconditioner from the components' current
linearization, exactly as an uninterrupted run would at that same period. The Eisenstat-Walker
forcing function, `eisenstat_walker_eta(opt, k, ...)` (`newton_krylov.hpp:251`), takes the step
index `k` and the current/previous residual norms as *arguments*, not members -- it is a pure
function called from within the same per-period solve, with no state surviving the call that
built it.

### 1.5 Registered field storage

Owner: each `Component`, bound once via `Component::register_fields(FieldRegistry&, const
CouplingGraph&)` (`include/fuse/runtime/component.hpp:315`). `FieldRegistry::bind()`
(`include/fuse/field/registry.hpp:28-69`) stores a `FieldRef` -- a non-owning pointer plus a
`Layout` (`include/fuse/field/field_ref.hpp:63-67`: `stride`, `component_stride`, `count`,
`components`) -- never the data itself. Distribution and global-id availability are properties
of **what the component registered the pointer over**, not of the registry: see Section 2.

### 1.6 Transport / remap state

`fuse::Transport` instances (`graph_transports`, `src/runtime/runner_impl.hpp`, one per
`OpKind::Transport` instruction, "Never resized after setup") and the per-plan device/host
scratch buffers (`transport_in_scratch`/`transport_out_scratch`) are **derived, not carried
state**: they are rebuilt from the coupling graph and the current field layouts at `setup()`
(`runner_impl.hpp`'s own comment: "built once in setup() from the channels of the coupling
graph"). A remap weight matrix (`fuse::backend::RemapPlan`, intersection weights from
`intx_weights.cpp`/`tempest_weights.cpp`) is recomputable from the mesh alone and from the
remap kind/order recorded in the coupling graph -- it does **not** need to be in the restart
file, provided the mesh (Section 4) and the graph descriptor (Section 4's provenance block) that
produced it are both restored first and the weights are deterministically rebuilt from them (not
read from a stale cache). This is the "which is recomputable, which is not" the task brief asks
for: remap/transport state is recomputable; **the mesh and the field values are not**.

### 1.7 Plugin / provider state (ABI)

`fuse_provider_abi_v1::save_state`/`::restore_state` (`include/fuse/abi/provider.h:54,57`) carry
the identical `(self, in, out)` shape as the C++ `Component` methods, with `in`/`out` typed
`fuse_null` -- i.e. the ABI's own save/restore is the SAME no-argument, in-memory rollback
primitive as `Component::save_state()`, not a separate restart-to-disk hook. `FUSE_CAP_CHECKPOINT`
(`provider.h:59`) is the capability bit a plugin host already checks before trusting these are
real. **A plugin's internal state is therefore exactly as opaque to restart as a C++
component's own private state** (Section 1.8) -- restart can only ask it to serialize itself
through whatever new entry point this design adds (Section 3), never read it directly.

### 1.8 Component-private ("hidden") state

`Component::capabilities()`'s own doc comment names this directly: `kCapabilityHiddenState`
marks "any state `advance()` mutates outside what `register_fields()` bound"
(`include/fuse/runtime/component.hpp:442-446`, bit meaning documented at `:401-417`). The R16
caveat on `save_state()`/`restore_state()` (`component.hpp:355-367`) is explicit that FUSE's
rollback verification is "a `memcmp` against REGISTERED field storage only... It cannot see, and
therefore cannot verify" anything a component never registered. **Every one of the four worked
examples has such state** -- e.g. `species_core.hpp`'s internal grid/flux buffers behind
`UComponent`/`VComponent` (`examples/brusselator/species_component.hpp:104,430`, each calling
`core_.save_state()`/`core_.restore_state()` at `:179-186,513-520`), `DirichletSlab::saved_q_`
and `NeumannSlab::saved_u_` (`examples/dn_heat/dn_component.hpp:383,522`),
`GrayScottSpeciesComponent::state_`/`saved_` (`examples/gray_scott/species_component.hpp:278-280`),
and `BeamComponent`/`FluidComponent`'s own internal solver state behind `beam_`/`fluid_`
(`examples/fsi_channel/fsi_components.hpp:157-349`). **This is the measured starting fact's own
point, restated per-example: none of this is serializable today**, because `save_state()`
returns `Result<void>`, not bytes.

### 1.9 Python components

`python/src/component_bindings.cpp` is the pybind11 trampoline for `fuse::Component`; a Python
component's `save_state()`/`restore_state()` overrides are ordinary Python-side method calls
through that trampoline, with the identical no-argument, in-memory-rollback shape as the C++
base (Section 1.8's caveat applies without modification -- a Python component's own instance
attributes are exactly as invisible to FUSE as a C++ component's private members). **Not
separately measured beyond this** -- I did not find a Python example component with
restart-relevant state beyond what Section 1.8 already covers; flagged as an open item if the
owner has one in mind (open_questions).

### 1.10 Summary -- what has NO global-id index today

Explicitly, per the task's instruction to say so: window waveform/history element position
(Section 1.2), every accelerator buffer (Aitken/Steffensen `r_`/`r_prev_`, IQN
`v_`/`w_`/`blocks_`, Section 1.3), and the plain `std::vector<double>` arrays every one of the
four worked examples registers (Section 2.3) carry **no** global id today. Only entities read
through a `fuse::MeshView` (native or MOAB backend) carry one (Section 2.1-2.2) -- and of the
four named examples, only the Brusselator actually goes through a `MeshView` (Section 2.3).

---

## 2. Global ids

### 2.1 Where they come from, and the contract they carry

`fuse::MeshView::global_id(int dim, std::size_t i)` (`include/fuse/backend/mesh_view.hpp:139`)
is the one interface every mesh-backed field in FUSE can call. Its own contract
(`mesh_view.hpp:39-42`) states it explicitly: "`global_id(dim, i)` is NOT an index. Global ids
are typically sparse after partitioning, may exceed `entity_count(dim)` by orders of magnitude,
and must never be used to subscript anything. They are for matching entities across ranks and
across meshes." A local index `i` "carries no ordering guarantee across dimensions, ranks, or
reloads" (`mesh_view.hpp:48-51`). This is exactly the "point hits are never a global id"
discipline `GhostPattern::remote_local_index` documents for a different structure
(`include/fuse/linalg/comm.hpp:79-83`) -- the codebase is consistent that a "local index" and a
"global id" are never interchangeable.

### 2.2 Per-backend uniqueness/contiguity

**Native mesh backend** (`fuse::native::NativeMeshView`, `src/backend/native_mesh.cpp:770-773`,
`global_id()` returns `vertex_gid_[i]`/`cell_gid_[i]` directly). These ids are supplied by the
CALLER at construction -- `NativeMeshBuilder::add_vertex(std::int64_t gid, ...)` and
`::add_cell(..., std::int64_t gid, ...)` (`src/backend/native_mesh.cpp:13,25`) -- the builder
does not assign or validate them itself. Uniqueness and contiguity are therefore a property of
whichever generator built the mesh, not a guarantee `NativeMeshBuilder` enforces. The generator
FUSE ships for the Brusselator (`fuse::native::NativeMeshBackend`, "generated:quad:NxM") assigns
`cell_gid = ix + iy*nx` (`examples/brusselator/brusselator_physics.hpp:20-22`, citing
`src/backend/native_generators.cpp`'s own `Grid::cell_gid`) -- **contiguous and unique by
construction** over the whole (un-partitioned) logical grid, independent of how many ranks the
mesh is later sliced across.

**MOAB backend** (`fuse::MoabMeshView`, `src/backend/moab_backend.cpp:381` ff.). `DimData`
(`moab_backend.cpp:364-379`) keeps `global_ids` parallel to the MOAB entity handles per
dimension, but explicitly tracks `gid_error`: a dimension can have **no usable global
numbering** -- the comment gives a concrete, measured cause: "a partitioned MOAB read routinely
manufactures auxiliary entities that no one numbered -- `PARALLEL_RESOLVE_SHARED_ENTS` creates
interface edges with `GLOBAL_ID` left at MOAB's -1 default" (`moab_backend.cpp:370-378`).
`global_id()` throws `InvariantViolation` rather than return the unusable value
(`moab_backend.cpp:447-456`) -- "Returning the raw value, or 0, would let two entities match that
are not the same entity." **Consequence for restart: a MOAB-backed field over a dimension with
`gid_error` set cannot be restarted rank-count-independently (Option A, Section 3) until that
dimension is numbered** -- this is a refusal the restart writer must make by calling
`global_id()` and letting it throw (or checking the equivalent query if one is added), never by
silently writing MOAB's internal handle instead.

### 2.3 Per-field-family answer, as asked

| Family | Global ids from | Unique/contiguous? | How a reader on a different rank count finds its rows |
|---|---|---|---|
| Native mesh fields (any component built over `fuse::native::NativeMeshView`) | `NativeMeshBuilder::add_vertex`/`add_cell`'s caller-supplied `gid` (`native_mesh.cpp:13,25`) | Only as strong as the generator; FUSE's own quad generator is contiguous+unique (Section 2.2) | Read by global id: hyperslab/point selection against the HDF5 dataset's own global-id-indexed layout (Section 4), independent of partition |
| MOAB-backed fields | MOAB's `GLOBAL_ID` tag, surfaced via `DimData::global_ids` (`moab_backend.cpp:364-379`) | Per-dimension; refuse (Section 2.2) when `gid_error` is set | Same read pattern, gated on the refusal above |
| `fuse::linalg::DistVector` | **None.** `DistVector`'s own header states its layout is "owned block first" in "its own local order" (`include/fuse/linalg/dist_vector.hpp:7-8,24-26`); `GhostPattern::remote_local_index` is explicitly "never a global id" (`comm.hpp:79-83`) | N/A -- no id space exists | **Cannot**, today. A `DistVector` built directly (not through a `MeshView`-backed field) has no way for a reader at a different rank count to know which logical row its position 0 corresponds to. This is the gap the owner's tension statement (measured starting fact) is really about, concretely: it is not a hypothetical "a component only offering an opaque blob," it is `DistVector` itself when used without a mesh behind it (e.g. a hand-assembled linear system in a Newton region) |
| Window waveform samples (`WindowMemberState::history`/`last_window_waveform`) | None (Section 1.2) -- indexed by element position in the member's own interface vector | N/A | Not re-partitionable by global id today; restarts at the SAME rank count only, unless the member's element ordering is itself tied to a global id the component chooses to expose (open question, see open_questions) |
| Brusselator example | Native mesh, `cell_gid = ix + iy*nx` (Section 2.2); `NativeGrid::make` recovers the local-index<->(i,j) correspondence by reading `MeshView::centroid` back (`examples/brusselator/brusselator_physics.hpp:23-25`) | Yes | Reader computes `(ix, iy)` from the stored global id, reads by hyperslab into whichever row band this rank's own partition owns |
| dn_heat example | **None.** `DirichletSlab`/`NeumannSlab` are placed one-component-per-rank (`examples/dn_heat/main.cpp:176,179`: `ranks = {0,1}` and `{1,1}` -- i.e. the whole component lives on exactly one rank each), and each component's own `u_in_`/`q_`/`u_`/`q_in_` (`examples/dn_heat/dn_component.hpp:380-382,519-521`) is a plain, fully-owned `std::vector<double>` -- not partitioned across ranks at all | N/A | Not applicable: there is no sub-component partitioning to re-partition. A restart that moves this component to a DIFFERENT rank still reads the whole array whole; "rank-count independence" for this example means only "the run as a whole may use a different number of ranks," never "this component's own array is split differently" |
| Gray-Scott example | Same pattern as dn_heat: `GrayScottSpeciesComponent::state_` (`examples/gray_scott/species_component.hpp:278`) is a single-rank-owned array, no `MeshView` in this example (confirmed: no `MeshView`/`NativeMeshView` reference anywhere under `examples/gray_scott/`) | N/A | Same as dn_heat |

**This table is the direct, measured answer to the owner's tension statement.** Only the
Brusselator, of the four named examples, exercises genuine sub-component spatial partitioning
with a real global id; dn_heat, Gray-Scott and fsi_channel place each component wholly on one
rank. CORRECTED (an earlier draft of this document misread the bracket literal): `RankSet`'s
two-argument constructor is `RankSet(int first, int count)` (`include/fuse/model/component.hpp:68`),
constructing the contiguous range `[first, first + count)` (`component.hpp:52`;
`ComponentDecl::ranks` is declared `RankSet` at `component.hpp:211`), so a
designated-initializer literal like
`.ranks = {0, 1}` list-initializes via this constructor, not via the separate, `explicit`
arbitrary-set constructor at `component.hpp:103`, which cannot be reached by brace-init with two
plain `int`s). So `{0, 1}` means `first=0, count=1`, i.e. **rank 0 only** -- the identical
single-rank placement dn_heat's own `{0, 1}`/`{1, 1}` pair already uses (rank 0, rank 1
respectively, `examples/dn_heat/main.cpp:176,179`). fsi_channel's `beam` and `fluid`
(`examples/fsi_channel/main.cpp:120,122`) are BOTH declared `.ranks = {0, 1}` and are therefore
BOTH placed on rank 0 only -- the same single rank as each other, not two different ranks and
not "replicated across both ranks 0 and 1" as an earlier draft of this section claimed. The
correction changes the mechanism (fsi_channel's two components share one rank, rather than each
spanning two) but not the conclusion: there is still no sub-component spatial partitioning
inside `BeamComponent`/`FluidComponent` for any rank count to re-partition, so "restart on a
different rank count" for dn_heat, Gray-Scott and fsi_channel alike means only "the run's
rank-to-component MAPPING changed," never "re-partition this component's own array by global
id" -- there is no such array to re-partition. The rank-count-independent slice of this design
(Section 8) is therefore properly verified on the Brusselator, not on dn_heat, Gray-Scott or
fsi_channel, none of which can exercise the re-partition path at all regardless of how the
restart reader is written.

---

## 3. Component interface options

All three options below extend `fuse::Component` (`include/fuse/runtime/component.hpp`) with
NEW virtual methods, defaulted to no-ops exactly as `save_state()`/`restore_state()` are today
(`component.hpp:374,385`) -- an existing component that does not override them is simply not
restartable, the same posture the existing rollback default already takes for
non-iterated components.

### Option A -- a writer/reader object the component calls, keyed by the component's own global ids

```cpp
// New, in component.hpp's own style (Result<void>, no exceptions across this boundary).
class RestartWriter {
 public:
  // `ids.size() == values.size() / components`; `ids` are THIS component's own global ids
  // for the index space `values` is defined over (mesh entity ids from MeshView::global_id(),
  // or a component-chosen numbering for a non-mesh index space -- see open_questions).
  virtual Result<void> put_field(std::string_view name, std::span<const std::int64_t> ids,
                                  std::span<const double> values, int components) = 0;
  // Scalars with no spatial index -- replicated, written once (first-writer-wins or
  // collectively-agreed value; see refusal rules, Section 4.4).
  virtual Result<void> put_scalar(std::string_view name, double value) = 0;
  virtual Result<void> put_scalar(std::string_view name, const Rational& value) = 0;
};
class RestartReader {
 public:
  // Returns only the rows THIS rank's own ids ask for; the reader resolves the hyperslab
  // or point selection (Section 4.3), the component never sees file offsets.
  virtual Result<void> get_field(std::string_view name, std::span<const std::int64_t> ids,
                                  std::span<double> out, int components) = 0;
  virtual Result<double> get_scalar(std::string_view name) = 0;
};

class Component {
  // ... existing members unchanged ...
  virtual Result<void> write_restart(RestartWriter&) { return Result<void>::success(); }
  virtual Result<void> read_restart(RestartReader&) { return Result<void>::success(); }
  // Paired with a new capability bit, kCapabilityRestart, read the same way
  // supports_rollback() reads kCapabilityRollback (component.hpp:420-429) -- NOT folded into
  // the same bit, because a component can roll back (iteration-region) without being
  // restartable-to-disk, and vice versa.
};
```

- **C/Fortran ABI shape**, following `docs/dev/abi-design.md`'s append-only rule (every member
  appended, never inserted; `struct_size`-gated reads; an appended member must grow `sizeof`,
  `abi-design.md:62-78`): two new table entries in `fuse_provider_abi_v1`
  (`include/fuse/abi/provider.h:27`), appended after `restore_state` --
  `write_restart`/`read_restart`, same `(self, in, out)` shape as every other entry, with new
  `fuse_restart_writer`/`fuse_restart_reader` opaque handle types in `types.h` carrying the
  `put_field`/`get_field` operations as further table entries (the "narrow waist" the ABI
  design page already uses for the field table, `provider.h`'s own `describe`/`bind` pattern).
  Generated from `include/fuse/abi/entries.def`, per the existing golden-file discipline
  (`abi-design.md:40-41`).
- **Python binding impact**: `python/src/component_bindings.cpp`'s pybind11 trampoline gains two
  more overridable methods, mirroring how `save_state`/`restore_state` are already exposed;
  `RestartWriter`/`RestartReader` need pybind11 wrappers exposing `put_field`/`get_field` over
  NumPy arrays (ids as `int64`, values as `float64`), a binding of the same shape
  `field_registry.hpp`/`field_bindings.cpp` already use for `FieldRegistry`.
- **Effect on existing components**: every one of the four worked examples needs a
  `write_restart`/`read_restart` override, because every one carries hidden state
  (Section 1.8) that `register_fields()` never exposed. For dn_heat/Gray-Scott/fsi_channel
  (Section 2.3, no sub-component partitioning), the "global ids" argument degenerates to the
  trivial case of one id per array element numbered `0..n-1` -- correct and sufficient, because
  there is nothing to re-partition, but it means Option A's generality is doing no useful work
  for those three; see Section 8's staging argument for using dn_heat as the same-rank-count
  pilot precisely because of this simplicity.
- **What FUSE can verify vs. must trust**: FUSE can verify that `ids` has no duplicate within one
  component's own call (an O(n log n) check at write time, cheap next to the I/O it guards) and
  that every id the file claims for a field was actually written by SOME rank (a collective
  count-and-compare, not a local one). FUSE **cannot** verify that the values are a faithful
  serialization of the component's full state -- exactly the R16 caveat already stated for
  rollback (`component.hpp:355-367`) applied to a new context: "FUSE verifies restart found
  everything it was told to expect" is a different, strictly weaker claim than "FUSE verifies
  the restart is correct," and documentation for this method must say so as plainly as the R16
  comment already does for rollback.

### Option B -- an opaque, versioned byte record per rank, refused by name on a different rank count

```cpp
class Component {
  virtual Result<std::vector<std::byte>> save_restart_blob() { return std::vector<std::byte>{}; }
  virtual Result<void> load_restart_blob(std::span<const std::byte> bytes,
                                         std::uint32_t written_by_version) {
    return Result<void>::success();
  }
  // Declares the component's own serialization version, so a reader can refuse an
  // incompatible blob by name rather than by a corrupted read -- the same posture
  // kGraphVersion already takes (descriptor.hpp) for the coupling graph itself.
  [[nodiscard]] virtual std::uint32_t restart_blob_version() const { return 0; }
};
```

- Same decomposition only, by construction: the blob is opaque bytes with no id structure FUSE
  can read, so a reader at a different rank count has no way to know which bytes belong to which
  logical row even in principle. The refusal is **by name, at restart-open time, collectively**:
  every rank reads the restart file's own recorded rank count (a scalar in the provenance block,
  Section 4.1) and compares it to `MPI_Comm_size()` of the restart communicator BEFORE any blob
  is read; a mismatch is a `Failure` returned identically on every rank from the SAME locally
  computed comparison (every rank computes the identical comparison from identical inputs -- the
  restart file's own rank count and this run's own `MPI_Comm_size()` -- so the "agree first"
  requirement is satisfied by construction, not by an extra reduction: see Section 7.iii's test
  of this exact refusal).
- **ABI shape**: two more `fuse_provider_abi_v1` entries, `save_restart_blob`/`load_restart_blob`,
  taking a `fuse_byte_span` in/out struct (new type, `types.h`, following the existing
  `struct_size`-first convention) plus the `written_by_version` scalar.
- **Effect on existing components**: trivial to add -- `memcpy` the private members into a byte
  buffer -- compared to Option A's requirement to enumerate every element by id. This is Option
  B's whole appeal for the three single-rank-per-component examples (Section 2.3): there is
  nothing to gain from Option A's generality there, and Option B is strictly less code.
- **What FUSE can verify**: the blob's own declared version against what the reader expects
  (refuse on mismatch, Section 4.4), and that a blob exists for every component the graph names
  (a missing component record, Section 7.iii). FUSE cannot verify anything about the blob's
  internal structure -- it is bytes.

### Option C -- both, with A required for rank-count-independent restart

The recommended default, following the owner's already-made decision that field data is keyed
by global id generally: implement both interfaces, document Option A as the one a component
MUST implement to be restarted on a different rank count (checked, not trusted -- a component
that implements only Option B is tagged in the schema's component-record as "same-decomposition
only," Section 4, and the reader refuses it collectively if the restart's rank count differs
from the write's, exactly as a pure-Option-B component would be refused). This matches
Section 2.3's finding exactly: dn_heat/Gray-Scott/fsi_channel are honest Option-B candidates (no
partitioning to lose), Brusselator is the one example for which Option A is load-bearing, not
decorative.

**Open question for the owner (also listed in open_questions):** does a component get to
declare "I offer Option B only" as a first-class, intentional choice (e.g. a legacy or
third-party physics code the owner cannot modify), or is Option A eventually mandatory for every
component the graph admits into a restartable run? The trade-off: mandatory-A is simpler to
reason about (one restart contract) but forces work on components that, per Section 2.3, gain
nothing from it; optional-B-only is pragmatic but means "is this run restartable at this rank
count" is a property the owner must track per-component, not a single yes/no for the graph.

---

## 4. HDF5 schema

### 4.1 Group/dataset/attribute layout

```
/                                   (root)
  @format_name          = "fuse-restart"     (string attribute)
  @format_version       = <uint32>           (bumped per Section 4.4's rule)
  @fuse_library_version = {major, minor, patch}   (mirrors fuse_semantic_version, types.h:55-59)
  @descriptor_hash       = <uint64>          (fnv1a over serialize(graph), see below)
  @written_by_rank_count = <uint32>          (MPI_Comm_size() at write time; Option-B refusal key, Section 3)

  /mesh                              (written ONCE per file; never rewritten by a later restart point)
    @mesh_backend        = "native" | "moab"
    @dim_count
    /dim0/global_ids      [N0]  int64            (vertices; MeshView::global_id(0, i), mesh_view.hpp:139)
    /dim0/coordinates     [N0,3] double
    /dimD/global_ids      [ND]  int64            (D = topological dim; cells)
    /dimD/connectivity    [ragged -- offsets + flat vertex-gid list, mirroring
                            native_mesh_io.cpp's own VTK-cell-type convention, see Section 5]
    /dimD/cell_type       [ND]  uint8            (vtk_cell_type() encoding, native_mesh_io.cpp:23-37,
                                                   reused rather than reinvented)

  /restart_points/<k>/              (k = 0, 1, 2, ... in write order)
    @time_numerator, @time_denominator   (fuse_exact_time's own two int64 fields, types.h:50-53 --
                                           written exactly, never only as the lossy double below)
    @time_double                          (double, for a human/ParaView reading the sidecar only;
                                            never read back by the restart reader itself)
    @period_index                         (int64, Runner::Impl::period_index, runner_impl.hpp:1045)
    @checksum              = <uint64>     (fnv1a over the restart point's own payload bytes,
                                            the same primitive transfer_map_io.cpp:153,195 already
                                            uses for its own file's payload)
    /fields/<field_name>/
      global_ids    [n_written]  int64    (every id ANY rank wrote this point; monotonically
                                            increasing on write so a hyperslab reader can binary-
                                            search its own range)
      values        [n_written, components]  double
    /components/<component_name>/
      @restart_blob_version   (Option B, Section 3)
      @interface  = "A" | "B" | "A+B"      (which of Section 3's methods this component actually
                                             used when THIS point was written -- read back before
                                             restore_restart() is called, so a reader never
                                             guesses)
      blob          [bytes]                (Option B; opaque)
      scalars/<name>  (Option A scalars; one dataset or attribute per declared scalar)
    /regions/<region_name>/
      @converged_iterations                (diagnostic, not required for correctness)
    /accelerators/<region_name>/
      @kind           = "None"|"Constant"|"Aitken"|"Steffensen"|"IQN_ILS"|"IQN_IMVJ"
      omega                                 (Aitken/Steffensen scalar state, Section 1.3)
      r_prev          [n]                   (Section 1.3; n = THIS rank's OWN stacked-interface
                                              -vector length at write time -- see the refusal below)
      v, w            [n, cols]             (IQN; Section 1.3)
      @cols
      @written_by_stacked_vector_layout_hash   (a hash of the (region, rank, block-order) tuple
                                                 that produced the stacked vector this point's
                                                 `r_prev`/`v`/`w` are indexed by -- Section 7.ii's
                                                 own refusal key: a restart reader that finds this
                                                 hash does not match ITS OWN freshly-computed
                                                 layout for the same region refuses to load the
                                                 accelerator arrays and instead calls
                                                 Accelerator::reset(), exactly as a window
                                                 boundary would (accelerator.hpp:56-58) --
                                                 because, per Section 1.3, there is no
                                                 element-wise correspondence across a layout
                                                 change, so loading anyway would silently
                                                 mis-assign a residual component to the wrong
                                                 physical quantity)
    /windows/<region_name>/<member_name>/
      clock_numerator, clock_denominator
      history/<element>        [k, 2]        (t, value pairs, Section 1.2)
      last_window_waveform/<element>/samples [k2]   (Waveform's own sample buffer, Section 1.2)
      straddle_pending, straddle_window_start_{num,den}, straddle_window_end_{num,den}
```

**Provenance.** The `@descriptor_hash` is `fnv1a(serialize(graph))` --
`fuse::serialize(const CouplingGraph&)` already exists (`include/fuse/model/descriptor.hpp:293`)
and three OTHER call sites already hash their own serialized bytes with this exact function
(`src/backend/weights_operator.cpp:176`, `src/model/semantics.cpp:343`,
`src/model/transform.cpp:125`) -- this design reuses that established pattern rather than
inventing a new hash, and the graph's own version gate (`kGraphVersion`,
`descriptor.hpp:486-489`) already refuses a version mismatch inside `deserialize()` before this
design's own hash comparison would even run. **One measured caveat the schema inherits
unavoidably:** `descriptor.hpp`'s own `Writer::pod`/`Reader::pod` are host-endian, not
little-endian like every other canonical encoding in `fuse/base/bytes.hpp` -- recorded as OPEN in
`KNOWN_ISSUES.md` (D62, "The graph descriptor is host-endian while `bytes.hpp`'s canonical rules
are little-endian"). **Consequence: `@descriptor_hash` is currently only comparable between
hosts of the same endianness**, which this design does not fix (D62 is owned elsewhere) but must
document at the point the hash is used, so a cross-architecture restart failure is correctly
diagnosed as D62, not as a new restart defect.

### 4.2 Why global-id-keyed datasets, not one dataset per rank

Keying `/restart_points/<k>/fields/<name>/{global_ids,values}` by global id rather than by
"rank 0's chunk, rank 1's chunk, ..." is what makes Option A's promise real: a reader at a
DIFFERENT rank count opens the same two flat datasets and computes its OWN hyperslab or point
selection from the global ids it now owns, with no dependency on how many ranks wrote the file.
This is the direct consequence of `MeshView::global_id()`'s own contract (Section 2.1) -- ids are
for matching, not for position -- applied to a file format instead of to in-memory matching.

### 4.3 Parallel I/O pattern

- **Collective MPI-IO** for every per-rank-varying dataset (`global_ids`, `values` in each field
  group): `H5Dwrite`/`H5Dread` with `H5FD_MPIO_COLLECTIVE` set via the dataset transfer property
  list, each rank contributing a hyperslab at the offset its own prefix-summed local count
  determines (an `MPI_Exscan` over "how many ids do I own," a standard parallel-HDF5 idiom, not
  a new primitive this codebase needs to invent).
- **Chunking**: one chunk per rank's own contiguous write region at write time is the natural
  choice and keeps the write collective-but-independent per rank; HDF5's own chunk cache sizing
  is a tuning question deferred past Stage A (robustness-over-efficiency: get a correct
  collective write first, per the owner's standing priority).
- **Who writes replicated data** (the mesh, Section 4.1's `/mesh` group; provenance attributes):
  rank 0 writes, every other rank's corresponding collective call participates with a zero-size
  selection -- the ordinary parallel-HDF5 pattern for "logically global, physically replicated"
  data, so the write is still one collective call every rank enters (never a rank-0-only
  `H5Dwrite` that leaves other ranks not calling the collective operation at all, which would be
  exactly the "collective path returns on a locally computed condition" mistake the owner's
  standing rule forbids).

### 4.4 Atomicity and consistency

- **Write to a temporary name, rename on success.** `/path/restart.h5.tmp-<pid>` (or a
  per-restart-point temp group, see trade-off below) written and `H5Fclose`d successfully, then
  `rename()`d to the real name -- collectively agreed: every rank calls `rename()` only after a
  collective `MPI_Allreduce(MIN, ok)` on "did my own close succeed," so a crash partway through
  one rank's write can never leave a file at the real path that looks complete. This mirrors
  `transfer_map_io.cpp`'s own atomicity discipline at a smaller scale (it checks a magic number
  and a trailing `fnv1a` checksum, `transfer_map_io.cpp:153,195`, before trusting a read) scaled
  up to "the whole file."
- **Trade-off for multi-restart-point files vs. one-file-per-point:** a single HDF5 file growing
  one `/restart_points/<k>/` group per checkpoint needs an append-safe write (HDF5 supports
  extending a file, but the rename-on-success atomicity above protects the WHOLE file, so a
  crash mid-append to an otherwise-valid existing file is a harder case -- the new point's data
  must be written to a staging location and the file extended only after that succeeds, or each
  restart point is simply its own file, trading one atomic rename per checkpoint for N small
  files instead of one growing one). This decision belongs in open_questions.
- **Checksum**: `@checksum` on each restart point (fnv1a over the point's own payload bytes,
  Section 4.1), checked by the reader BEFORE any `restore_restart()`/`read_restart()` call runs --
  a mismatch is `Failure`, collectively agreed the same way the rank-count mismatch is (Section 3,
  Option B), never a partial restore.
- **Version gate**: `@format_version` checked first, before any other read; an unknown or
  unsupported version is `Failure` with no attempt to interpret the rest of the file --
  `descriptor.hpp`'s own version-gate rationale (`descriptor.hpp:90-152`, "the version check
  turns [a silent misread] into a refusal") applies verbatim here, for the same reason.

---

## 5. XDMF sidecar

- **What it references**: an XDMF `.xmf` file is XML that points AT the HDF5 datasets by path
  (`<DataItem Format="HDF">restart.h5:/mesh/dim2/coordinates</DataItem>`-style references) --
  it duplicates no field data, only describes the HDF5 layout's topology/geometry/attribute
  structure for a reader (ParaView, VisIt) that speaks XDMF natively. One `.xmf` `<Grid>` per
  restart point, referencing that point's `/restart_points/<k>/fields/*` datasets as
  `<Attribute>` elements over the SAME `/mesh` geometry/topology referenced by every grid (the
  "mesh written once" decision, Section 4.1, is exactly what makes a single shared
  `<Geometry>`/`<Topology>` block legal XDMF across every `<Grid>` in a `<Grid
  CollectionType="Temporal">`).
- **Which mesh kinds it is expressible for**: XDMF's own `<Topology TopologyType="...">`
  vocabulary covers exactly the cell types `native_mesh_io.cpp`'s own `vtk_cell_type()` already
  enumerates (`native_mesh_io.cpp:23-37`: Vertex, Polyline/segment, Triangle, Quad, Polygon,
  Tetrahedron, Pyramid, Wedge/Prism, Hexahedron, Polyhedron) -- reusing that mapping rather than
  inventing a second one means the sidecar writer needs only the existing VTK-code table plus
  XDMF's own name for each code, not new per-cell-type logic. A **structured** native mesh
  (regular Nx x Ny quad grid, as the Brusselator generator produces) can ALSO be expressed as
  XDMF's more compact `<Topology TopologyType="2DCoRectMesh">` with an `<Origin>`/`<Spacing>`
  geometry instead of an explicit connectivity list -- an optimization, not required for Stage A
  correctness, and deferred (robustness first, per the owner's standing priority) unless the
  owner wants it in the first slice.
- **What it cannot show**: component-private state (Section 1.8 -- a plugin's opaque blob,
  Section 3 Option B) has no spatial location XDMF can attach an `<Attribute>` to, so it is
  invisible in ParaView/VisIt by construction, not by an XDMF-writer omission -- the sidecar can
  only expose what Option A's global-id-keyed fields put into `/restart_points/<k>/fields/`.
  Accelerator and window-waveform state (Section 1.2, Section 1.3) are likewise not mesh-attached
  quantities and are not represented in the sidecar at all; they exist only in the HDF5 file for
  the restart reader.

---

## 6. NetCDF-UGRID output

- **Conventions**: CF-1.x for variable attributes (`units`, `standard_name`/`long_name`,
  `_FillValue`) and UGRID-1.0 for the mesh topology variable (`mesh_topology:cf_role =
  "mesh_topology"`, `face_node_connectivity`, `face_dimension`, etc.) -- the two conventions
  compose in NetCDF-4 by design (UGRID defines the mesh topology variable and leaves CF's own
  variable-attribute rules untouched for the data variables that reference it via
  `:mesh = "<topology_var_name>"` and `:location = "face"|"node"|"edge"`).
- **Which of FUSE's meshes map to UGRID**: UGRID-1.0's core topology (face-node connectivity) is
  a 2-D surface-mesh convention; FUSE's **2-D face meshes** (the Brusselator's quad generator,
  or a MOAB-loaded 2-D unstructured mesh) map onto it directly -- `face_node_connectivity` from
  the same cell/vertex connectivity the HDF5 schema's `/mesh` group already stores (Section 4.1),
  so this writer is a second serialization of the SAME mesh, not a second source of truth for
  it. A genuinely **3-D volume mesh** (tetrahedra/hexahedra with volume cells) is **outside**
  UGRID-1.0's core convention, which has no standard volume-cell connectivity; the UGRID
  community's own layered/volume-mesh extensions exist but are not part of the stable 1.0
  convention most tools (ParaView's UGRID reader) implement -- a volume-mesh FUSE field is
  written as plain CF data (dimensions + coordinates, no UGRID topology variable) rather than
  claiming UGRID conformance it cannot deliver. This split belongs in open_questions: does the
  owner want volume output at all in this stage, given the honest answer is "CF, not UGRID, for
  a volume mesh"?
- **Parallel write path**: NetCDF-4/HDF5 collective write through the same `/opt/netcdf`
  (built parallel on `/opt/hdf5`, per host facts) the HDF5 restart writer already targets --
  `nc_var_par_access(..., NC_COLLECTIVE)` before a collective `nc_put_vara_double`, the standard
  parallel-NetCDF-4 idiom; no PnetCDF classic-format path is needed since NetCDF-4 is the
  explicit target.
- **Why output only, not restart**: UGRID/CF has no concept of component-private state
  (Section 1.8), accelerator history (Section 1.3), or window waveforms (Section 1.2) -- it is a
  conventions layer for describing PHYSICAL field data on a mesh for analysis tools, not a
  restart format, and was never asked to be one. A NetCDF writer that only ever sees
  `register_fields()`-bound storage (Section 1.5) by construction cannot restart a run,
  independent of any implementation choice this design makes.

---

## 7. Verification plan (tests before code)

### (i) Same rank count

Run N periods uninterrupted; separately, run k < N periods, write a restart-to-disk point,
start a NEW process at the same rank count, restore, run N-k more periods. Compare: every
registered field (bit-identical `memcmp`, the same standard `run_region`'s own rollback
verification already uses, `include/fuse/runtime/component.hpp:360-367`'s R16 comment names the
mechanism), every component's private state surfaced via Option A/B's own value (bit-identical,
since nothing about re-partitioning enters at the same rank count), the Runner's own
`period_index` and `RunStats` (`include/fuse/runtime/runner.hpp` -- `stats()`), and stdout/log
content where the run emits deterministic diagnostics. This test needs `FUSE_FP_REPRODUCIBLE=ON`
(`cmake/FuseFpPolicy.cmake:48`, default ON) for the comparison to be bit-identical rather than
ULP-close, per that flag's own measured rationale (`FuseFpPolicy.cmake:1-30`: without it, a
reduction loop's vectorized-vs-scalar code shape can differ by one ULP between two otherwise
identical runs).

### (ii) Different rank count

Restored state compared BY GLOBAL ID (not by rank-local position) to the written state.
**Precisely what can and cannot be bit-identical, stated as the task asks:**

- A field's own VALUE at a given global id: bit-identical, because the value itself does not
  depend on which rank currently owns that id -- it is a number attached to a mesh entity, and
  re-partitioning changes who holds it, never what it is.
- The accelerator's own internal history (Section 1.3): **cannot** be identical, and this design
  does not attempt to carry it across a rank-count change at all (Section 4.1's own refusal:
  `@written_by_stacked_vector_layout_hash` mismatch means `Accelerator::reset()`, not a loaded
  history) -- because, per Section 1.3, a different rank count produces a different
  stacked-vector layout with no element-wise correspondence to the old one. The strongest
  comparison that is still exact: the restored run's FIRST post-restart iteration-region sweep
  starts from `reset()` (zero history, `accelerator.hpp:56-58`'s own stated post-reset state)
  and the test asserts it reaches the SAME converged fixed point the uninterrupted run's
  corresponding window reached -- a weaker, but genuinely exact, claim: not "the iteration path
  is identical" (it provably is not, different accelerator starting state) but "the fixed point
  is identical," checked to the derived tolerance the owner's standing convention requires (a
  tolerance from a stated theorem -- the fixed-point residual's own convergence tolerance
  already in `RunnerConfig`, `include/fuse/runtime/runner.hpp`, is the correct instrument here,
  not a new ad hoc one).
- A reduction whose order changes with the decomposition (any `Reducer::sum`/`norm2` over a
  `DistVector`, `include/fuse/linalg/dist_vector.hpp:40-48`): **cannot** be bit-identical across
  a rank-count change, because floating-point addition is not associative and a different
  partition sums in a different order. The exact, derivable comparison: the tolerance convention
  already on record for this project -- `c*u*(sum|x_i y_i|)` from Higham's recursive-summation
  bound -- bounds exactly this discrepancy, and this design reuses that bound rather than
  widening it or inventing a new one; the test's own tolerance must cite which theorem it
  derives from, per the owner's standing tolerance convention.

### (iii) Every refusal made to fire

| Refusal | Trigger | Collective? |
|---|---|---|
| Version mismatch | `@format_version` set to an unsupported value in a hand-edited test fixture | Every rank reads the identical attribute, so yes by construction (no reduction needed, identical input) |
| Descriptor-hash mismatch | Restart a graph compiled with one field added/removed against a file written by the graph without it | Same -- every rank computes the hash from its own in-memory graph, identical on every rank by `validate()`'s own precondition |
| Opaque (Option B) record restored at a different rank count | Write at 2 ranks, attempt restore at 3 | Collective, per Section 3's own stated mechanism (every rank compares the file's `@written_by_rank_count` to its own `MPI_Comm_size()`) |
| Missing component record | Delete one component's `/components/<name>/` group from a test fixture before restoring | Every rank's reader enumerates the SAME expected component list from the (already-restored) graph, so the "is it present" check is locally computed from identical inputs on every rank -- collective by construction, same pattern as the version check |
| Truncated/temporary file | Kill the write process after `H5Fcreate` but before the collective close-and-rename (Section 4.4) completes; attempt to open `restart.h5.tmp-*` directly, or the ordinary path when the rename never happened | File-open failure or missing-group failure, caught at `@format_version`-read time, before any payload is trusted |
| MOAB dimension with `gid_error` passed to Option A | A test fixture MOAB mesh lacking `GLOBAL_ID` on interface edges (`moab_backend.cpp:370-378`'s own scenario) | `MeshView::global_id()` throws `InvariantViolation` (`moab_backend.cpp:454`) -- the restart writer must catch this at `put_field()` time and turn it into a named `Failure`, not let the exception escape across the ABI boundary unhandled |
| Accelerator layout-hash mismatch at a rank-count change | Restore at a different rank count than written; assert `Accelerator::reset()` fired, not a stale load | Locally computed from the file's `@written_by_stacked_vector_layout_hash` vs. the freshly-resolved layout -- identical inputs on every rank, collective by construction |

### (iv) Mutation rows

Every refusal above is a `requires = "mpi"` candidate ONLY where the refusal is genuinely
rank-count-dependent (the opaque-record-at-different-rank-count and accelerator-layout-hash
rows); the version/hash/missing-component/truncated-file refusals are checkable at a single
rank and belong at tier 0, following the existing convention `test/runtime/`'s own tests already
use (`docs/dev/testing.md`'s tier-0 description). A host run adds exactly the two
rank-count-dependent rows plus the same-rank-count and different-rank-count round trips
themselves (Section 7.i, Section 7.ii) -- nothing else in this plan needs more than one process.

**What can run at tier 0 (no MPI) vs. what needs the host, stated explicitly:** tier 0 covers
the HDF5 schema's own structural correctness (write-then-read round trip of the mesh, a single
restart point, every refusal in the table above except the two rank-count-dependent rows) at
ONE rank -- HDF5 itself needs no MPI to read/write a file serially, and `/opt/hdf5` being a
parallel build does not require every program linking it to use `MPI_COMM_WORLD`
(`H5Fcreate`/`H5Dwrite` with the default, non-collective property list works at one rank without
`MPI_Init` at all). The host is needed for: the collective MPI-IO write/read path itself
(Section 4.3), both round trips in Section 7.i/Section 7.ii at more than one rank, and the two
rank-count-dependent refusals.

---

## 8. Staging

### Slice 1 -- same-rank-count restart of dn_heat

**Why dn_heat, not Brusselator, for the first slice**: dn_heat already carries a derived
agreement-bound tolerance (`examples/dn_heat/agreement_bound.hpp`) and has no sub-component
partitioning to get right at the same time as the restart mechanics (Section 2.3) -- the
smallest possible surface that still exercises every non-partitioning-dependent piece of this
design: the HDF5 schema (Section 4), the XDMF sidecar (Section 5), Option B for
`DirichletSlab`/`NeumannSlab`'s own private state (Section 3; trivial here per Section 3's own
observation), the Runner's clock/period-index restore (Section 1.1), and refusals (i), (ii),
(iv), (v) of Section 7.iii (every one except the two rank-count-dependent rows, which genuinely
need Brusselator).

- **Files touched** (new, under the owner's live checkout once Stage A is approved -- none
  touched in THIS stage): `include/fuse/runtime/restart.hpp` (new `RestartWriter`/`RestartReader`
  interfaces plus the Option-B blob methods, added to `component.hpp`), `src/runtime/restart_io.cpp`
  (HDF5 writer/reader implementation), `src/runtime/restart_xdmf.cpp` (sidecar), a new
  `cmake/FuseDependencies.cmake` option block, `examples/dn_heat/dn_component.hpp` (Option-B
  overrides), `examples/dn_heat/main.cpp` (checkpoint-interval/restart CLI options), new tests
  under `test/runtime/restart_*.cpp`.
- **CMake options**: a NEW `FUSE_ENABLE_HDF5` option (default OFF, following the
  `FUSE_ENABLE_MOAB`/`FUSE_ENABLE_PETSC` naming convention, `cmake/FuseDependencies.cmake:119-122`),
  **not** folded into `FUSE_ENABLE_LOWFIVE` -- LowFive's own documentation states plainly that its
  interception is in-memory only, "no file ever reaches disk" (`docs/user/lowfive.md:6-7`), so
  reusing `FUSE_ENABLE_LOWFIVE`'s HDF5 *discovery* (the `find_package(HDF5 REQUIRED COMPONENTS
  C)` call, `cmake/FuseDependencies.cmake:1054`) is correct and saves duplicated `find_package`
  logic, but the restart writer must call `H5Fcreate`/`H5Dwrite` with the DEFAULT (non-VOL)
  property list, never LowFive's `H5VOLProperty`, or the restart write would silently become
  in-memory-only too. **This interaction needs the SAME two-HDF5-image hazard check
  (`tools/probe-hdf5-images.sh`, `cmake/FuseDependencies.cmake:1107-1217`) already gates
  `FUSE_ENABLE_LOWFIVE` + `FUSE_ENABLE_MOAB` on**, extended to cover `FUSE_ENABLE_HDF5` +
  `FUSE_ENABLE_LOWFIVE` together, since both would then link the SAME `/opt/hdf5` into one
  process for two different purposes (genuine disk I/O vs. VOL interception) -- this is a new
  configure-time check, not a reuse of the existing one as-is, and belongs in open_questions
  as a build-system design point the owner should see before it is written.
- **KNOWN_ISSUES.md entry to open**: `D63` (next available number, after `D62`) -- "Restart-to-disk
  cannot capture a period mid-suspension (an externally-driven ABI participant's coroutine
  frame, `src/runtime/resume.hpp`) -- restart points are period-boundary only, by construction,
  not a gap to close later" -- recorded as a DESIGN CONSTRAINT (Section 1.1), not a defect,
  following the file's own convention of recording known, deliberate limitations next to open
  defects.
- **What would make me stop and ask the owner**: discovering that `DirichletSlab`/`NeumannSlab`'s
  own private state (the `k1_eff`/`beta_` nonlinearity parameters and `q_`/`u_in_` buffers,
  `examples/dn_heat/dn_component.hpp`) includes anything this document's state inventory
  (Section 1) did not already name -- a sign the inventory itself is incomplete and needs the
  owner's review before more examples are touched.

### Slice 2 -- rank-count-independent restart (Brusselator)

Built on Slice 1's schema and writer/reader, adding: Option A's `put_field`/`get_field` over
`MeshView::global_id()` for `UComponent`/`VComponent` (`examples/brusselator/species_component.hpp`),
the accelerator-layout-hash refusal (Section 4.1, Section 7.iii), and the different-rank-count
round trip test (Section 7.ii) at 2 and 3 ranks on the host. Files touched:
`examples/brusselator/species_component.hpp` (Option A overrides), `examples/brusselator/main.cpp`
(restart CLI options), new `test/parallel/restart_rank_change_mpi_test.cpp`. CMake: no new
option, reuses Slice 1's `FUSE_ENABLE_HDF5`.

### Slice 3 -- XDMF sidecar

Adds `restart_xdmf.cpp`'s own writer (if not already built in Slice 1 -- listed separately here
because it is independently useful once ANY restart file exists, and the owner may want to see
it against dn_heat's structured grid before Brusselator's unstructured-generator output is
ready) and a manual-inspection acceptance step (open ParaView/VisIt against the sidecar -- not
automatable in this sandbox, host-only, by the owner).

### Slice 4 -- NetCDF-UGRID output

Independent of restart entirely (Section 6: output-only). New `FUSE_ENABLE_NETCDF` option
(following the same naming convention), `src/backend/netcdf_ugrid_writer.cpp`, wired to
whichever of dn_heat (CF only, no UGRID -- it has no mesh) or Brusselator (CF+UGRID, 2-D face
mesh) the owner wants demonstrated first; Brusselator is the natural choice since it is the only
one of the four examples with a mesh to put a UGRID topology variable over at all (Section 2.3).

---

## Summary table of file:line citations used above

(For the reviewer's convenience; every citation above is repeated here with no new claims.)

- `include/fuse/runtime/component.hpp:315,342,374,385,401-417,420-429,442-446,474` -- Component
  interface, save/restore_state, capabilities, is_externally_driven.
- `include/fuse/runtime/runner.hpp:137,386,389-404,411` -- RunnerConfig::periods, run_period,
  resume_period.
- `src/runtime/runner_impl.hpp:549,555,677-813,1045-1053,1098-1104,1127` -- period_index,
  RegionRuntime, WindowMemberConfig::hosted/consumer_hosted.
- `src/runtime/resume.hpp:1-45` -- coroutine frame mechanism for externally-driven advance.
- `include/fuse/runtime/window_coupler.hpp:140-258` -- WindowSubstep, WindowMemberState,
  WindowMember, WindowChannel.
- `include/fuse/num/waveform.hpp:81-85,133-199` -- Waveform sample/basis contract.
- `include/fuse/num/accelerator.hpp:35-58,154-156` -- Accelerator interface, reset/columns.
- `src/num/accel_relaxation.cpp:404-408,489-493` -- AitkenAccel/SteffensenAccel members.
- `src/num/accel_quasi_newton.cpp:92-94,766-793` -- QuasiNewtonAccel/JacobianBlock members.
- `src/runtime/iteration.cpp:2489-2522,3016-3019` -- local BlockJacobiPreconditioner.
- `include/fuse/num/newton_krylov.hpp:28-30,251` -- lagged preconditioner, Eisenstat-Walker.
- `include/fuse/field/registry.hpp:28-69` -- FieldRegistry::bind.
- `include/fuse/field/field_ref.hpp:45-67` -- Layout.
- `include/fuse/backend/mesh_view.hpp:39-51,139` -- MeshView::global_id contract.
- `src/backend/native_mesh.cpp:13,25,770-773` -- NativeMeshBuilder, NativeMeshView::global_id.
- `src/backend/moab_backend.cpp:364-379,447-456,689-692` -- MoabMeshView global ids, gid_error.
- `include/fuse/linalg/dist_vector.hpp:7-8,24-48` -- DistVector layout, no global id.
- `include/fuse/linalg/comm.hpp:79-83` -- GhostPattern::remote_local_index, not a global id.
- `examples/brusselator/brusselator_physics.hpp:16-25` -- Brusselator mesh/cell_gid.
- `examples/brusselator/species_component.hpp:104,141-186,430,463-520` -- UComponent/VComponent.
- `examples/dn_heat/dn_component.hpp:229-397,380-383,397-522` -- DirichletSlab/NeumannSlab.
- `examples/dn_heat/main.cpp:176,179,260,263` -- single-rank-per-component placement.
- `examples/gray_scott/species_component.hpp:56-280` -- GrayScottSpeciesComponent.
- `examples/fsi_channel/fsi_components.hpp:61-349` -- BeamComponent/FluidComponent.
- `examples/fsi_channel/main.cpp:120,122` -- beam/fluid both on rank 0 only (`RankSet(0,1)`, `component.hpp:68`), not two ranks.
- `include/fuse/model/descriptor.hpp:90-152,163,196,293,486-489` -- Writer/Reader, serialize,
  kGraphVersion, version-gate rationale.
- `include/fuse/base/bytes.hpp:133` -- fnv1a.
- `src/backend/weights_operator.cpp:176`, `src/model/semantics.cpp:343`,
  `src/model/transform.cpp:125` -- existing fnv1a(serialize(...)) precedent.
- `src/backend/transfer_map_io.cpp:153,195` -- existing checksum-then-trust discipline.
- `src/backend/native_mesh_io.cpp:23-37` -- vtk_cell_type table, reused for XDMF topology types.
- `include/fuse/abi/provider.h:27,54,57,59` -- fuse_provider_abi_v1 save_state/restore_state.
- `include/fuse/abi/types.h:50-59` -- fuse_exact_time, fuse_semantic_version.
- `docs/dev/abi-design.md:22-41,62-78` -- one-symbol ABI, append-only struct growth rules.
- `docs/user/lowfive.md:6-7` -- LowFive interception is in-memory only, no file reaches disk.
- `cmake/FuseDependencies.cmake:119-122,1050-1217` -- option naming convention, LowFive+HDF5
  discovery, two-HDF5-image hazard probe.
- `cmake/FuseFpPolicy.cmake:1-30,48` -- FUSE_FP_REPRODUCIBLE.
- `KNOWN_ISSUES.md` D14 (closed, two-HDF5-image hazard), D58 (probe limitation), D62 (descriptor
  endianness) -- context for the provenance hash's own caveat.

---

## 9. Stage B Slice 1 -- implementation record (2026-10-15)

The owner accepted this design with five decisions, reproduced here against what Slice 1 actually
built, each cited to the file that carries it.

1. **Form A is primary; Form B is same-rank-count only; a graph refuses a rank-count change
   collectively if any component offers only Form B.** Implemented exactly as a capability-bit
   pair, `kCapabilityRestartFormA`/`kCapabilityRestartFormB` (`include/fuse/model/
   capabilities.hpp`, bits 6-7), checked by `fuse::check_restart_rank_count()` (`src/runtime/
   restart.cpp`) -- `test/runtime/restart_test.cpp`'s `FormBOnlyComponentOnDifferentRankCountRefuses`
   proves the refusal fires and that equal rank counts never trigger it. All four examples named in
   Stage A were not all touched this slice (dn_heat only, per the staging plan); `examples/dn_heat/
   dn_component.hpp`'s `DirichletSlab`/`NeumannSlab` both set `kCapabilityRestartFormA` and
   implement `write_restart()`/`read_restart()`, confirming the owner's own claim that a
   single-rank component's numbering is trivial (`0..n-1`).

2. **FUSE verifies uniqueness and coverage of the declared global size for ALL arrays of a restart
   point in ONE collective, never one per array.** Implemented as `RestartWriter::put_field()`'s
   `declared_global_size` parameter plus `finish_point()`'s own combined sort over every field's
   `(name, id)` pairs at once (`include/fuse/model/restart.hpp`'s own header comment states the
   contract; `src/restart_io/hdf5_restart_io.cpp`'s `Hdf5GroupWriter::finish_point()` is the
   concrete combined sort). At a single rank (this slice; MPI cannot run in this sandbox) the
   combined sort re-confirms what `put_field()`'s own local check already found rather than
   discovering anything new across ranks -- `test/runtime/restart_test.cpp`'s
   `BadGlobalIdNumberingRefuses`/`DeclaredGlobalSizeMismatchRefuses` exercise the LOCAL half of
   this; the genuinely cross-rank half is Slice 2's own host-run deliverable (see host_pending).

3. **Both file layouts, selectable at run time; `GrowingFile` is copy-append-rename, never
   in-place append.** `fuse::restart_io::FileLayout::{SeparateMeshAndPoints,GrowingFile}`
   (`include/fuse/restart_io/hdf5_restart_io.hpp`); `create_restart_point()`'s own implementation
   copies the existing file to a temp name before opening it read-write for the new point, and
   `finish()` is the one atomic rename in either layout (`src/restart_io/hdf5_restart_io.cpp`).
   `test/runtime/restart_test.cpp`'s `GrowingLayoutAppendsASecondPointWithoutDamagingTheFirst`
   writes two points and confirms the first is still readable afterward; the cost (one file copy
   per point) is exactly what the owner's own brief named and is not hidden anywhere.

4. **Accelerator and window-waveform histories are keyed by (field, global id) so they MOVE
   across a rank-count change; a row with no global id refuses by name; an explicit, off-by-default
   run option allows a reset instead, reported at restart.** Implemented for the accelerator half
   -- `Accelerator::export_restart_state()`/`import_restart_state()` (`include/fuse/num/
   accelerator.hpp`), with real bodies for `AitkenAccel` and `SteffensenAccel`
   (`src/num/accel_relaxation.cpp`); `QuasiNewtonAccel` (IQN-ILS/IQN-IMVJ) keeps the inherited
   no-op default this slice, a scope reduction declared below. `Runner::write_restart_fields()`/
   `restore_restart_fields()` (`src/runtime/runner.cpp`) resolve each `RegionBlock`'s own rows to
   the OWNING component's Form-A ids via `build_field_owner_map()`; `allow_accelerator_reset_on_
   mismatch`/`any_accelerator_reset` are threaded through exactly as named.
   **Window-waveform history is NOT implemented this slice** -- dn_heat has no coupling window
   (declared below; Slice 2, alongside Brusselator, is where a window genuinely exists to restart).
   **Same rank count: everything restored bit for bit, including accelerator memory** --
   `test/runtime/restart_test.cpp`'s `SameRankRoundTripSeparateLayout`/`SameRankRoundTripGrowingLayout`
   prove this directly: run 5 periods uninterrupted vs. run 3, write, a NEW `Runner` restores, runs
   2 more, compared `EXPECT_EQ` (not a tolerance) on every field element, both components' private
   `time_`/`history_`, and the LAST period's own `IterationStats::iterations` (sweep count) -- the
   Aitken accelerator's own dynamic relaxation factor and residual history are part of what makes
   that sweep count match, so this is a genuine test of decision 4, not only of decision 1.

5. **NetCDF/MPAS-style general unstructured output; can a UGRID-1.0 `mesh_topology` variable
   reference the MPAS arrays.** Design only this round (Slice 4, unchanged from Stage A's own
   Section 6 deferral) -- **answered, checked against the UGRID-1.0 conventions text rather than
   assumed**: UGRID-1.0's `face_node_connectivity` variable carries its own `start_index` attribute
   (declaring 0- or 1-based indexing) and supports a `_FillValue` for a ragged/padded face-node
   array whose rows have fewer nodes than the array's own width -- exactly MPAS's own
   `verticesOnCell(nCells, maxEdges)` layout (1-based Fortran indexing, padded to `maxEdges` with a
   fill value where `nEdgesOnCell(c) < maxEdges`). **Yes**: a UGRID-1.0 `mesh_topology` variable can
   reference MPAS's own `verticesOnCell` AS-IS, declaring `start_index` and `_FillValue` to match
   MPAS's own convention, with no data rewritten -- the two conventions were designed to compose for
   exactly this padded-polygon case, not merely compatible by coincidence.

### Measured findings this slice surfaced that Stage A's design did not anticipate

- **Accelerator rows from DIFFERENT components collide if concatenated into one flat id space.**
  `dirichlet_neumann`'s own two blocks (`u_wall_in`, owned by `dirichlet`; `q_wall_in`, owned by
  `neumann`) both report Form-A ids `0..n-1` -- correct and sufficient for each field ALONE, but
  writing one flat `r_prev` field over the WHOLE accelerated vector put both components' id-0 rows
  at the same key, refused by `E-RESTART-DUPLICATE-ID`. Fixed with `combine_field_and_local_id()`
  (`src/runtime/runner.cpp`), a hash-combined `(qualified field name, local id)` key -- correct for
  this slice's graphs, FLAGGED rather than assumed general: a genuinely collision-proof design
  writes one NAMED sub-field per block instead of one flat field per region, listed in
  `open_questions` below as Slice 2 work.
- **A component's Form-A field name must equal the graph's own `FieldSpec::name`, not an
  arbitrary label, for any field an accelerated channel feeds.** `RegionBlock::target_field` is
  literally `graph.field(channel.target).name` (`src/runtime/iteration.cpp:1627`); the accelerator
  row resolution looks the owning field up BY THAT STRING, so `dn_component.hpp`'s first draft
  (field names `"q"`/`"u_in"` rather than the graph's own `"q_wall"`/`"u_wall_in"`) refused with
  `E-RESTART-MISSING-FIELD` at every restore. Fixed by renaming; documented as a convention in
  `docs/user/restart.md` and in `dn_component.hpp`'s own comment, so the next component author does
  not rediscover it by the same failing test.
- **The Runner's own clock is not restored by `restore_restart_fields()` and needed a new public
  setter.** `ctx.t_begin = Rational{period_index} * plan.period() + tick.time` (`src/runtime/
  runner.cpp`) is computed from `Impl::period_index` at call time, which is NOT a `Component` or an
  `Accelerator` and therefore outside what `restore_restart_fields()` touches. Added `Runner::
  set_current_period_index()` (`include/fuse/runtime/runner.hpp`, `src/runtime/runner.cpp`),
  called from `fuse::restore_restart_point()` (`src/runtime/restart.cpp`) immediately after fields
  succeed. Without it, the FIRST version of the round-trip test showed every restored field correct
  immediately after restore (a passing diagnostic assertion proved this) but WRONG after the next
  `run_period()` call, because the Runner recomputed `ctx.t_end` from period 0 instead of the
  restored period -- a measured, not hypothetical, bug this test caught before it reached the
  owner.

### Scope reductions declared for this round

- **QuasiNewtonAccel (IQN-ILS/IQN-IMVJ) restart state is not implemented** -- `export_restart_
  state()`/`import_restart_state()` keep the inherited no-op default, so an IQN-accelerated
  region's `v_`/`w_`/`blocks_` are NOT restored (a restore after an IQN-accelerated run resets that
  accelerator silently rather than refusing, because the no-op default reports no mismatch at all
  -- flagged explicitly in open_questions as a gap the owner should see, not a silent limitation).
  Aitken and Steffensen are implemented and tested; `AccelKind::None`/`Constant` need nothing (no
  learned state).
- **Window-waveform history (the second half of owner decision 4) is designed, not implemented.**
  No test in this slice exercises a coupling window (dn_heat has none); the API shape would follow
  the identical `(field, global id)` keying the accelerator path already uses, but writing and
  testing it needs a window-scoped region to exist, which is Slice 2's own Brusselator work.
- **The combined-collective verification of owner decision 2 is implemented and tested only at a
  single rank.** The genuinely cross-rank half (an `Allgatherv` over the combined id buffer) is
  unwritten; MPI cannot start in this sandbox (standing rule), so it could not have been tested here
  regardless of whether it were written.
- **The C/Fortran ABI entries for restart (item B's own "every generated file and golden updated")
  are NOT implemented this round.** `include/fuse/abi/entries.def`'s append-only X-macro generator
  (`tools/gen-abi.py`, 799 lines) and its own golden-file discipline
  (`test/abi/entries_golden_test.cpp`) were read but not extended -- a correct extension needs
  the SAME scrutiny this slice spent on the C++ API (measured bugs found by running real tests,
  twice), and attempting it without that scrutiny risked shipping a plausible-looking but unverified
  ABI surface, which is a worse outcome than declaring it undone. The C++ `fuse::RestartWriter`/
  `RestartReader`/`Component::write_restart()` etc. are complete and tested; only the C-callable
  table entries and the Fortran shim are missing.
- **Python binding impact: not assessed this round.** Listed for completeness, not measured.

## 10. Stage B Slice 1 -- review round 2 record (2026-10-15)

Round 1's review found two MUST-FIX defects and left the brief's ABI item open; this section
records what round 2 changed, file:line cited, against each point the review raised.

### MUST-FIX 1 -- no silent reset

Round 1's base `Accelerator::import_restart_state()` returned success on a no-op default, so an
accelerator that never implemented restart state (QuasiNewtonAccel) silently reset itself on every
restore with no refusal and no report -- a correct-looking restore that was actually wrong. Fixed
with a new virtual, `Accelerator::restart_state_implemented()` (`include/fuse/num/accelerator.hpp:213`,
default `false`); the base `import_restart_state()` now unconditionally
returns `Failure{"E-RESTART-NOT-IMPLEMENTED", ...}` (`accelerator.hpp:272`) and documents that this
default must never actually run, because the caller checks `restart_state_implemented()` first
(`accelerator.hpp:263,278`). `Runner::restore_restart_fields()` (`src/runtime/runner.cpp:4845`)
checks the flag BEFORE calling `import_restart_state()` at all: `false` and no opt-in refuses with
`E-RESTART-ACCEL-NOT-IMPLEMENTED` (`runner.cpp:4848`); opted in via `allow_accelerator_reset_on_
mismatch`, it calls `Accelerator::reset()` directly and sets `any_accelerator_reset = true`
(`runner.cpp:4859`) -- the no-op default is never reached either way. This is collective by
construction, not by a runtime reduction: every rank's graph assigns the same `AccelKind` to the
same region (the graph itself is agreed before any rank runs), so `restart_state_implemented()`
evaluates to the identical boolean on every rank from identical input, and the comment at
`runner.cpp:4839` states this explicitly rather than leaving it to be rediscovered.
`NoneAccel`/`ConstantAccel` override the flag to `true` (trivially correct -- no learned state);
`AitkenAccel`/`SteffensenAccel` override to `true` with real bodies. `QuasiNewtonAccel` (IQN-ILS/
IQN-IMVJ) keeps the inherited `false` -- used deliberately as this round's test subject for the
refusal path (a real, already-wired production `AccelKind`, not a new test-only seam), per the
scope reduction below. Two new tests, `AccelNotImplementedRefusesWithoutOptIn` and
`AccelNotImplementedWithAllowResetSucceedsAndReports` (`test/runtime/restart_test.cpp`), prove both
branches; a mutation row (`restart-accel-not-implemented-gate-disabled`, `build-evidence/mutations/
restart.toml`) disabling the gate kills exactly these two tests, confirmed and reverted.

### MUST-FIX 2 -- exact keys, no hash

Round 1's `combine_field_and_local_id()` (`src/runtime/runner.cpp`, removed this round) hash-
combined `(qualified field name, local id)` into one 64-bit key shared across every accelerated
block in a region -- collision-FREE for this slice's own graphs, but not collision-PROOF, which is
what the review asked for. Fixed by keying exactly instead of combining at all:
`fuse::RestartRowBlock` (`include/fuse/model/restart.hpp:121`, `{field_name, ids}`) replaces the
flat id span in `Accelerator::export_restart_state()`/`import_restart_state()`'s own signature
(`accelerator.hpp`), and `detail::put_field_per_block()`/`get_field_per_block()`
(`accelerator.hpp:328,339`, defined in `src/num/accel_relaxation.cpp`) write and read ONE
`put_field`/`get_field` call PER BLOCK, under the dataset name `name_prefix + "/" +
block.field_name` -- a different HDF5 dataset per field, not a shared id space, so two blocks with
IDENTICAL local ids cannot collide by construction, not merely by this slice's own graphs
happening not to exercise it. `AitkenAccel`/`SteffensenAccel` are rewired onto these helpers;
`Runner::write_restart_fields()`/`restore_restart_fields()` build a `std::vector<RestartRowBlock>`
(one per `RegionBlock`) instead of a flat combined-id vector. A new direct unit test,
`PerBlockKeysNeverCollideOnIdenticalLocalIds` (`test/runtime/restart_test.cpp`), drives two
`RestartRowBlock`s with the SAME local ids `{0,1,2}` and different field names
(`"dirichlet/q_wall"`/`"neumann/q_wall_in"`) and confirms no cross-contamination; a mutation row
(`restart-per-block-key-collapsed-to-id-alone`, `restart.toml`) collapsing the qualified name back
to `name_prefix` alone kills this test plus every multi-block-region test, confirmed and reverted.

### MUST-FIX 3 -- `FUSE_FP_REPRODUCIBLE` status, and why bit-identity holds regardless

`FUSE_FP_REPRODUCIBLE` is **ON by default** (confirmed from this round's own cmake configure
output: "Reproducible floating point: -ffp-contract=off on every target"; the option's default is
set in `cmake/FuseFpPolicy.cmake:49`). Restart's own round-trip tests hold bit-for-bit regardless of
this setting, and the reason is in the policy file's own header, not asserted independently of it:
`FuseFpPolicy.cmake:1-19` explains that the ULP divergence `-ffp-contract=off` guards against comes
from comparing TWO DIFFERENT CODE SHAPES over the same data -- a vectorized loop's main body against
its own scalar remainder tail, or a Release build's auto-vectorized loop against a Debug build's
fully-unrolled one (`FuseFpPolicy.cmake:9-15`: "a runtime-trip-count loop gets split into a
vectorized main body ... plus a scalar remainder tail"). `SameRankRoundTripSeparateLayout`/
`SameRankRoundTripGrowingLayout` (`test/runtime/restart_test.cpp`) compare ONE compiled test
binary's two executions -- an uninterrupted 5-period reference run, and a 3-period run, restore,
plus 2 more periods -- through the IDENTICAL compiled `advance()`/accelerator functions on both
sides; there is only ever ONE code shape in play, never two. Whatever FMA contraction that single
compiled binary applies (fused or not, `FUSE_FP_REPRODUCIBLE` ON or OFF) is applied identically to
both the reference run and the restored run, because both run the same instructions on the same
hardware from the same build -- there is no second code shape for the first one to disagree with.
The round-trip tests' own `EXPECT_EQ` (not a tolerance) on every field element is therefore a valid
assertion independent of this build option, and would remain valid even if a future build turned
`FUSE_FP_REPRODUCIBLE` OFF -- restart's bit-identity guarantee rests on "same binary, same inputs,
twice", not on the contraction policy.

### Item 4 -- ABI entries for restart (brief item, now done)

Read `test/abi/entries_golden_test.cpp` and `test/abi/fortran/sizes_test.f90` first, as instructed:
`types.h`/`participant.h`/`provider.h` and the Fortran modules are ALL generated from
`include/fuse/abi/entries.def` via `tools/gen-abi.py`; the two named test files are themselves
hand-written (`sizes_test.f90` keeps exactly one hand-written mirror type because Fortran cannot
dereference the participant table's function pointers; the provider table IS generated and used
directly, since a provider plugin never dereferences it). Added, following `entries.def`'s own
append-only convention (new entries always appended last):

- **Provider-side Form B**: `save_restart_blob`/`load_restart_blob` (`entries.def:332-333`), using
  `fuse_restart_blob_sink`/`fuse_restart_blob_source` (`entries.def:162-175`) -- query-then-fill,
  reusing `fuse_graph_sink`'s own idiom rather than inventing a new one. New capability bit
  `FUSE_CAP_RESTART_FORM_B` (`entries.def:108`, value 64). `ProviderAdapter::save_restart_blob()`/
  `load_restart_blob()` (`src/abi/provider_adapter.cpp:392,435`) implement the caller side of the
  SAME query-then-fill pattern `impl_save_graph` already uses as the callee side, with a cached
  `last_blob_version_` (`provider_adapter.hpp:84`) backing `restart_blob_version()`.
- **Participant-side**: `request_restart_point` (`entries.def:286`, `fuse_null`/`fuse_null`) --
  a flag, not a trigger: the ABI's fixed `(ctx, in, out)` shape cannot carry a file path or an
  HDF5 layout choice, so `Runner::request_restart_point()`/`take_restart_point_request()`
  (`include/fuse/runtime/runner.hpp:704,711`, `src/runtime/runner.cpp:4704,4708`, backed by
  `Impl::restart_point_requested`, `src/runtime/runner_impl.hpp:1049`) is consume-not-peek: the
  embedding host polls after every `run_period()`/`resume_period()` and decides whether, and how,
  to act. Idempotent -- two requests before a poll leave exactly one pending, proven by
  `AbiParticipant.RequestRestartPointSetsTheRunnersFlagIdempotently` (`test/abi/participant_test.cpp`)
  via a new test-only seam, `fuse::abi_test::runner_of(ctx)` (`src/abi/test_support.hpp/.cpp`).
- Ran `tools/gen-abi.py --write` then `--check` (passed, "6 current, 0 stale, of 6") to regenerate
  all 6 generated files, which caught a REAL latent bug: `gen_fortran_provider_module()`
  (`tools/gen-abi.py:636`) hardcoded its `use ffuse, only: ...` list instead of computing it from
  `spec.provider_entries`' own struct types -- gfortran failed to compile the regenerated
  `ffuse_provider.f90` with "Cannot IMPORT 'fuse_restart_blob_sink' ... does not exist" until this
  was fixed to compute the needed-structs set dynamically, with line-wrapping for Fortran's
  ~132-column free-form limit.
- Golden files updated: `test/abi/struct_sizes.golden` (+`fuse_restart_blob_sink` 40,
  +`fuse_restart_blob_source` 32, `fuse_participant_abi_v1` 144->152, `fuse_provider_abi_v1`
  80->96), `entries_golden_test.cpp`'s measured-struct map and `FUSE_NO_TRAILING_PADDING` checks,
  `negotiation_test.cpp`'s `FakeV2Table` (folded `request_restart_point` into its real-prefix, the
  same convention prior entries already used).
- Two new C-consumer tests at tier 0: `ConformanceTest.ProviderSaveAndLoadRestartBlobRoundTrips`
  (`test/abi/conformance_test.cpp`, a full round trip through a `dlopen`'d plugin ->
  `ProviderAdapter` -> `Component`) and `AbiParticipant.RequestRestartPointSetsTheRunnersFlag
  Idempotently` (above). `test/abi/plugins/linear_provider.c` (shared across every provider-plugin
  test variant) gained real `p_save_restart_blob`/`p_load_restart_blob` bodies and sets
  `FUSE_CAP_RESTART_FORM_B` unconditionally in `p_capabilities`.
- **Not implemented this round, declared as a scope reduction**: Form A's full writer/reader ABI
  surface. `entries.def`'s own header states the one rule every entry in this file follows -- an
  identical flat `(ctx, in, out)` shape -- and there is no existing precedent in this codebase for
  a nested-table or opaque-handle ABI object; inventing one without dedicated review risked the same
  "plausible-looking but unverified" outcome round 1 declined for the whole ABI item. Only Form B
  (same-decomposition-only, opaque bytes) fits the existing shape and is implemented.

### Item 5 -- the cross-rank collective half of decision 2 (brief item, now done, compiled only)

New header `include/fuse/restart_io/collective_check_mpi.hpp` declares
`collective_duplicate_and_coverage_check()` (`collective_check_mpi.hpp:82`) over an `int`
(`MPI_Comm_c2f` handle), not `MPI_Comm` -- following the EXACT convention `fuse/linalg/comm.hpp`
already uses, because `tools/check-layering.py`'s `FORBIDDEN_PUBLIC` rule rejects `mpi.h` in any
public header and `restart_io`'s own `ALLOWED_PUBLIC_THIRD_PARTY` entry is `set()` (verified:
`check-layering.py` reports 0 violations with this header in the tree). The implementation
(`src/restart_io/collective_check_mpi.cpp`, compiled only under `FUSE_ENABLE_MPI`) serializes
`(name, id)` pairs and `(name, declared_size)` pairs as length-prefixed byte buffers and calls
exactly TWO `MPI_Allgatherv`s total -- never one per field, per the owner's decision 2 -- after
which every rank performs an IDENTICAL pure local check over the now-identical gathered data
(duplicate id, declared-size agreement, coverage), so every rank reaches the same verdict with no
further reduction. Five new tests (`test/parallel/restart_collective_mpi_test.cpp`), registered via
`fuse_add_mpi_test(parallel restart_collective_mpi ... RANKS 2 3 LABELS restart_io)`, scale with
`fuse::testing::size()` rather than hardcoding rank counts. **Deliberately not wired into
`Hdf5GroupWriter`/`create_restart_point()` this round** -- Slice 1's own tested tier-0 surface stays
untouched; this is a standalone utility for a future round's parallel-aware writer, matching
`finish_point()`'s own pre-existing comment anticipating exactly this extension. **Compiled, never
run** -- MPI cannot start in this sandbox (standing rule); see `host_pending` for the exact commands
the lead runs on a real multi-rank host.

### Item 6 -- Python binding impact (brief item, read-only assessment, not implemented)

Read `python/src/component_bindings.cpp`, `python/src/field_registry.hpp` +
`field_bindings.cpp` (the `FieldRegistry` wrapper precedent), `python/src/module.cpp`, and
`docs/dev/python-surface.md` (confirmed the doc predates the restart track entirely -- no mention
of restart-to-disk in its own "What this does not expose" section, a genuinely new gap, not an
already-tracked one). `ForwardingComponent` currently infers `kCapabilityRollback` from
`hasattr(obj_, "save_state")`/`hasattr(obj_, "restore_state")` (`component_bindings.cpp:87-88`),
ORed into its declared capabilities at `component_bindings.cpp:177`. Restart-to-disk needs the same
hasattr-gated pattern, not yet written:

- New pybind11 wrapper classes for `RestartWriter`/`RestartReader` exposing `put_field`/`get_field`/
  `put_scalar`/etc. over NumPy arrays (ids as `int64`, values as `float64`) -- sized similarly to
  the `FieldRegistry` wrapper precedent (`field_registry.hpp`, 208 lines; `field_bindings.cpp`, 32
  lines), so roughly 150-250 combined lines.
- `ForwardingComponent` additions: `has_write_restart_`/`has_read_restart_`/
  `has_save_restart_blob_`/etc., hasattr-gated overrides for `write_restart`/`read_restart`/
  `save_restart_blob`/`load_restart_blob`/`restart_blob_version`, plus capability-bit inference
  extensions mirroring `kCapabilityRollback`'s own pattern -- roughly 40-60 lines.
- `module.cpp` registration of the new wrapper classes.
- **No binding work needed for `Accelerator::export_restart_state()`/`import_restart_state()`**:
  accelerators are never subclassed in Python -- a graph configures one only via the `AccelKind`
  enum (`component_bindings.cpp` has no accelerator-subclassing path), so a Python-driven run
  restarts accelerators through the same native C++ implementations regardless of which language
  the components are written in.

`FUSE_ENABLE_PYTHON` was OFF in every build this round; no binding code was written or built,
consistent with the "read-only, do not implement" instruction.

### Item 7 -- sandbox Fortran-plugin conformance failures (brief item, now fixed; MAJOR CORRECTION to round 1)

Round 1 reported "this sandbox has no Fortran compiler" and configured `FUSE_ENABLE_FORTRAN=OFF`
as a workaround. That diagnosis was **wrong**: `gfortran` (GNU 16.2.0) is installed at
`/opt/homebrew/bin/gfortran`, simply not on `PATH` (the identical situation as this sandbox's
`cmake`/`ninja`, which also need an explicit full path). Reconfiguring with
`-DFUSE_ENABLE_FORTRAN=ON -DCMAKE_Fortran_COMPILER=/opt/homebrew/bin/gfortran` makes
`abi.conformance`'s two previously-failing tests (`LagOneGraphAllFourKindsMatchNative`,
`SubcycledMeanGraphAllFourKindsMatchNative`) PASS, because the Fortran plugin dylib those tests
load now actually gets built. With this configuration (HDF5 ON, Fortran ON via the explicit
compiler path, MPI/MOAB/PETSc/LowFive OFF, Debug, `FUSE_FP_REPRODUCIBLE` ON by default): **219/219
tier-0 tests pass (100%)**, up from 210 under round 1's Fortran-OFF workaround (Fortran adds
`fuse_abi_fortran_sizes_test` plus the Fortran-linked verification tests). A separate build with
`-DFUSE_ENABLE_HDF5=OFF` (otherwise identical) confirmed nothing else breaks without HDF5:
**218/218 tests pass (100%)**; that build tree was deleted immediately after recording this result,
per the two-build-tree limit.

### Full tier-0 run, this round's final state

With every round-2 change applied (MUST-FIX 1/2/3, items 4/5/6/7) and `build2` configured exactly
as above (`-DFUSE_ENABLE_HDF5=ON -DFUSE_ENABLE_FORTRAN=ON
-DCMAKE_Fortran_COMPILER=/opt/homebrew/bin/gfortran -DFUSE_ENABLE_MPI=OFF`), a full `ctest`
run reports **100% tests passed, 0 tests failed out of 219**. The five gates (`check-layering.py`,
`check-invariants.py`, `verify-import.py`, `mutate.py --check-anchors`, and `sphinx-build -W
--keep-going`) were run against this same tree; the four static gates report 0 violations/problems
and 357 valid/0 stale anchors (two new rows added to `build-evidence/mutations/restart.toml` this
round, following its own hand-applied/restored/`cmp`-verified convention, not the source-anchor
system `--check-anchors` itself verifies).

### Scope reductions declared this round (in addition to round 1's, unresolved)

- **QuasiNewtonAccel (IQN-ILS/IQN-IMVJ) restart state is still not implemented** -- now with a
  tested, correct refusal (MUST-FIX 1) instead of round 1's silent reset. See `KNOWN_ISSUES.md`
  D64 for exactly what a future round would need to export/import, keyed by field.
- **Window-waveform restart history is still not implemented** (carried from round 1 unchanged --
  dn_heat has no coupling window; Slice 2's Brusselator work is where one exists to restart).
- **Form A's full writer/reader ABI surface is not implemented** (this round's own item 4 scope
  reduction, above) -- only Form B (opaque, same-decomposition-only bytes).
- **The cross-rank collective (item 5) is compiled and unit-tested but not wired into
  `Hdf5GroupWriter`/`create_restart_point()`** -- a standalone utility a future round's writer
  calls.
- **No Python binding code was written** (item 6, read-only by instruction).
- **No fault-injection provider-plugin variant exercises `save_restart_blob`/`load_restart_blob`
  returning an error** -- `linear_provider.c`'s new bodies always succeed; a variant that reports
  `FUSE_ERR_IO` or a version mismatch was not added this round, matching the level of fault
  coverage the ORIGINAL (pre-restart) provider entries this round extended already have in this
  test file.


## 11. Stage B Slice 1b -- typed fields, implementation record (2026-10-15)

The owner reviewed Slice 1's design (Section 9-10) and required restart
fields to be templated on element type before Slice 2 proceeds. This
section records what was built, what the real HDF5 and compiler behaviour
turned out to be (several measured facts this slice's own design review
did not anticipate), and the mutation/tier-0 evidence.

### Design implemented (owner's ten numbered items)

1. **`fuse::ElementType`** (`include/fuse/model/element_type.hpp:54-81`) is a
   closed enum: `Int8`..`Int64`, `UInt8`..`UInt64`, `Binary16`, `BFloat16`,
   `Binary32`, `Binary64`, `LongDouble`, `Bool`. `ElementTypeOf<T>`
   (`element_type.hpp:167-240`) resolves a C++ type to its enumerator at
   compile time; any `T` outside the supported set fails a `static_assert`
   in the unspecialised primary template.
2. **`fuse::bfloat16`** (`include/fuse/model/bfloat16.hpp`) is a FUSE
   storage type: 1 sign, 8 exponent (bias 127, same as `Binary32`), 7
   explicit significand bits. `from_float()` rounds to nearest, ties to
   even, quiet-NaN-preserving (`bfloat16.hpp:63-76`). `test/model/
   bfloat16_test.cpp`'s `ExhaustiveBitToFloatToBitIdentityOverEveryNonNan
   Pattern` round-trips all 65536 bit patterns (254 correctly identified as
   NaN). Stored in HDF5 as a user-defined float type built the same way as
   every other non-native float `ElementType` -- see item 5 below.
3. **`Bool`** is stored as one `UInt8` byte; `validate_bool_bytes()`
   (`src/restart_io/hdf5_restart_io.cpp:482-495`) refuses
   (`E-RESTART-BAD-BOOL-BYTE`) any byte that is not `0` or `1`.
4. **Templated interface.** `put_field<T>()`/`get_field<T>()` and
   `put_scalar<T>()`/`get_scalar<T>()` (`include/fuse/model/restart.hpp:
   194-200, 224-229, 305-311, 326-334`) are non-virtual templates over ONE
   virtual byte-based pair each (`put_field_bytes`/`get_field_bytes`,
   `put_scalar_bytes`/`get_scalar_bytes`). This round MIGRATED every caller
   to the templated interface rather than keeping a double-only legacy
   overload (the "say which" the owner asked for -- migrate, not
   dual-maintain): `src/num/accel_relaxation.cpp`, `src/runtime/
   restart.cpp` (`period_index` now `std::int64_t` through the typed path,
   never cast through `double`), `examples/dn_heat/dn_component.hpp`, and
   `test/runtime/restart_test.cpp`'s existing call sites. `get_scalar<T>()`
   takes an out-parameter `T&` rather than round 1's `Result<double>`
   return (`restart.hpp:314-325`'s comment records why: a value and a
   report cannot both occupy one `Result<>` slot without inventing a pair
   type this project has no other use for). `DirichletSlab` gained a
   `std::int32_t sweep_count_` and a `bool parity_flag_`
   (`examples/dn_heat/dn_component.hpp`), round-tripped through the Runner
   end to end -- `test/runtime/restart_test.cpp`'s `RunResult::
   dirichlet_sweep_count`/`dirichlet_parity_flag`, verified by a real
   mutation (disabling `sweep_count_` restore fails `SameRankRoundTrip*`,
   reverted and confirmed clean).
5. **Per-field metadata.** `write_type_metadata()`/`read_and_verify_type_
   metadata()` (`hdf5_restart_io.cpp:327-472`) write/check `element_type`,
   `byte_size`, `significand_bits`, `exponent_bits`, `exponent_bias`,
   `byte_order`, `is_signed` as HDF5 attributes alongside each dataset.
   `read_and_verify_type_metadata()` was REWRITTEN mid-round (see "Design
   bugs found" below): it returns a fresh `ElementTypeInfo` built from the
   FILE's own attributes, checked only for internal self-consistency
   against the dataset's own actual HDF5 datatype
   (`H5Tget_fields`/`H5Tget_ebias`/`H5Tget_norm`/`H5Tget_class`/
   `H5Tget_sign`) -- never against this build's global `element_type_info()`
   table. Any attribute/dataset-type disagreement refuses
   (`E-RESTART-TYPE-METADATA-MISMATCH`); an unrecognised `element_type`
   name refuses by name (`E-RESTART-UNKNOWN-ELEMENT-TYPE`).
6. **`ReadConversion::Exact`/`Widening`/`AllowNarrowing`**
   (`element_type.hpp:113`) -- `convert_one_element()`
   (`hdf5_restart_io.cpp:613-712`) is the one dispatch point: identical-type
   fast path, `Exact` refusal (`E-RESTART-TYPE-MISMATCH`), `Widening`'s
   exactness check via `fuse::detail::is_exact_widening()`
   (`element_type.cpp:183`, two overloads -- the `ElementTypeInfo`-based
   primary used here, and a convenience `ElementType`-based wrapper for
   same-platform callers), float narrowing via `fuse::detail::
   narrow_float()` (`float_bits.cpp:250`), integer narrowing via
   `encode_int_checked()` (`hdf5_restart_io.cpp:540-595`, refuses
   `E-RESTART-INTEGER-OVERFLOW` on overflow even under `AllowNarrowing`,
   and `E-RESTART-SIGNED-TO-UNSIGNED` unconditionally). All float
   conversion is done in FUSE code built and tested from scratch
   (`include/fuse/model/float_bits.hpp`/`.cpp`: `decode_float_bits()`,
   `encode_float_bits()`, `round_significand()`, `narrow_float()`) --
   **never HDF5's own soft-conversion machinery**; see "HDF5's own
   conversion limitation" below for why. The owner's exact mandated
   scenario -- a hand-built x86-64-style 80-bit `long double` dataset,
   mantissa `0x8000000000000001`, biased exponent `16383` (unbiased `0`,
   value `1 + 2^-63`), explicit leading bit (`H5T_NORM_MSBSET`, per
   `H5Tpublic.h`) -- is `test/runtime/typed_restart_hdf5_test.cpp`'s
   `TypedRestartHdf5.HandBuiltX86LongDoubleExactAndWideningRefuseAllow
   NarrowingReportsDelta`: `Exact` and `Widening` both refuse
   (`E-RESTART-TYPE-MISMATCH`), `AllowNarrowing` succeeds with
   `max_absolute_change` bit-for-bit equal to an independently computed
   `std::ldexp(1.0, -63)`.
7. **Accelerator/waveform histories stay `Binary64`.**
   `src/num/accel_relaxation.cpp`'s per-block helpers and Aitken's omega
   read/write exclusively `double` -- unchanged from Stage B Slice 1 Round
   2; this round only migrated their call sites to the typed
   `put_scalar<double>`/`get_scalar<double>` spelling, never the stored
   type.
8. **Collective global-id check** (`fuse/restart_io/collective_check_mpi.
   hpp`/`.cpp`, int64 ids) is unchanged from Round 2; typed fields do not
   touch it.
9. **ABI.** No Form-A entry exists yet (Round 2 Item 4's own scope
   reduction stands), so no ABI change was made. A future Form-A struct
   that exposes a field's element type would add one `std::uint8_t`-sized
   tag immediately after the existing `byte_size`/`components` pair in
   whatever field-descriptor struct Form A eventually defines, matching
   `ElementType`'s own `std::uint8_t` underlying type -- append-only,
   never inserted before an existing member, the same rule Round 2's Item
   4 note already states for Form A generally.
10. **Python.** Read-only note, not implemented: `fuse::bfloat16` has no
    standard numpy dtype; a future binding would need either `ml_dtypes`
    (adds a third-party dependency solely for its `bfloat16`/`float8_*`
    numpy extension types) or a FUSE-side `uint16` view with a documented
    reinterpretation contract. `Binary16` maps to numpy's own `float16`
    when the platform provides it; `LongDouble` maps to numpy's
    `longdouble` ONLY when the two platforms' extended-precision formats
    agree (this project's own measured Apple-Silicon-vs-x86-64 difference,
    item 2 below, means that mapping is platform-conditional, not
    universal). No implementation; `python/` is untouched this round.

### Measured findings this round surfaced that the design review did not anticipate

- **HDF5's own soft type-conversion refuses `H5T_NORM_MSBSET`.** A
  standalone test program (`/tmp/h5test3.c`, HDF5 1.14.1-2 at `/opt/hdf5`)
  built an 80-bit x86-extended-style `long double` HDF5 type (explicit
  leading significand bit) and asked `H5Dread()` to convert it to a plain
  `double`. HDF5's own error stack: `H5Tconv.c` line 4602 in
  `H5T__conv_f_f()`: "normalization method not implemented yet". This
  means FUSE cannot trust `H5Dread`'s type-conversion machinery for ANY
  conversion touching an explicit-leading-bit format (x86 extended
  precision is the only one this project's measured platforms produce,
  but the limitation is in HDF5's own soft converter, not specific to one
  bit pattern) -- motivating item 6's from-scratch bit-level pipeline.
- **`H5Tset_fields()`/`H5Tset_precision()`/`H5Tset_size()` ordering is
  direction-dependent**, found and fixed twice while building
  `make_derived_float_type()` (`hdf5_restart_io.cpp:236-295`): for a
  PRECISION DECREASE (e.g. `Binary32` -> `BFloat16`/`Binary16`,
  shrinking from 32 to 16 bits), `H5Tset_fields()` must run BEFORE
  `H5Tset_precision()`/`H5Tset_size()` shrink the type, or HDF5 reports
  "exponent bit field size/location is invalid" (the old field positions
  no longer fit the new, smaller size). For a PRECISION INCREASE (e.g.
  `Binary64` -> an 80-bit extended type), size/precision must grow FIRST,
  then `H5Tset_fields()` positions the fields in the new, larger size, or
  HDF5 reports "adjust sign, mantissa, and exponent fields first" (the old
  field positions do not fit the type being asked to shrink TO before it
  has grown). Both exact messages were captured from the real HDF5 error
  stack, not inferred.
- **The original `make_hdf5_type_for()` dispatched `LongDouble` to
  `H5T_NATIVE_LDOUBLE` unconditionally -- wrong when the `ElementTypeInfo`
  comes from a FOREIGN file's own attributes.** On this platform
  `H5T_NATIVE_LDOUBLE` is a 53-bit type; using it as the memory type while
  reading a file whose actual on-disk type is a hand-built MSBSET 80-bit
  type triggered the identical "normalization method not implemented"
  failure as the untrusted-soft-conversion path above, because HDF5 still
  has to soft-convert between the file type and whatever memory type is
  requested. **Fix:** `make_hdf5_type_for()` now dispatches EVERY float
  `ElementType` (`Binary16`/`BFloat16`/`Binary32`/`Binary64`/`LongDouble`)
  through ONE fully general `make_derived_float_type(const
  ElementTypeInfo&)` (`hdf5_restart_io.cpp:236-295`) that builds the HDF5
  type purely from the `ElementTypeInfo`'s OWN fields -- no native-type
  shortcut for any float type, including the common case where a platform's
  own `long double`/`double` happen to match a native type exactly.
  Within that rewrite, the precision formula was also corrected: `stored_
  mantissa_bits + exponent_bits + 1`, NOT `significand_bits + exponent_
  bits + 1` -- `significand_bits` already INCLUDES any implicit leading
  bit (`element_type.hpp`'s own `ElementTypeInfo::significand_bits` doc
  comment), so adding `+1` again overcounted by one bit for every
  IMPLIED-norm type. This was masked for every PRECISION-DECREASE case
  by HDF5's own auto-shrink-on-size-decrease behaviour (documented under
  `H5Tset_size()`), but produced a wrong, non-matching type for the
  INCREASE case (the x86-long-double scenario), which is how it was
  caught -- a real test failure (`E-RESTART-HDF5: H5Dread scalars/x86ld`),
  not a code-review catch.
- **A design flaw in the original `read_and_verify_type_metadata()`
  would have made it structurally IMPOSSIBLE to read a genuinely
  foreign-platform file under `AllowNarrowing`,** contradicting item 6's
  own requirement. The original version compared a file's `element_type`
  attributes against `fuse::element_type_info(enum)` -- THIS build's own
  canonical/global metadata table -- so any file whose actual recorded
  metadata did not match this platform's own table entry (exactly the
  x86-long-double scenario) would refuse before `AllowNarrowing` ever got
  a chance to run. Caught by test DESIGN (recognising the contradiction
  with item 6 while writing the hand-built-file test), fixed before
  running the broken version: the function now builds a FRESH
  `ElementTypeInfo` from the file's own attributes, verified only for
  internal self-consistency against the dataset's own actual HDF5 type.
  `is_exact_widening()` was refactored to take `const ElementTypeInfo&` on
  both sides for this reason (`element_type.hpp:264`, with a convenience
  `ElementType`-based overload at `element_type.hpp:272` for same-platform
  callers), and `convert_one_element()` takes the stored side as `const
  ElementTypeInfo&` directly rather than re-deriving it from an
  `ElementType` enumerator.

### Mutation rows (this round, build-evidence/mutations/restart.toml)

Each row below was hand-applied to `/tmp/restart/src4`, rebuilt, run for
real, reverted, and `cmp`-verified byte-identical against the clean file
before moving to the next row -- this project's established convention,
not the source-anchor system `mutate.py --check-anchors` verifies.

| # | Mutation | File:line | Killed by |
|---|---|---|---|
| 1 | `is_exact_widening()` always returns `true` | `element_type.cpp:183` (body replaced with unconditional `return true`) | `TypedRestart.WideningRefusesANonExactConversion` (`restart_test.cpp`), `TypedRestartHdf5.HandBuiltX86LongDoubleExactAndWideningRefuseAllowNarrowingReportsDelta` (its `Widening`-refuses assertion wrongly succeeds, then throws `bad_variant_access` reading a `Failure` as a value), `ElementType.IsExactWideningMatchesEveryOwnerNamedCase` |
| 2 | `bfloat16::from_float()`'s round-up branch disabled (`if (false && ...)`), reducing it to plain truncation | `bfloat16.hpp` (round-to-nearest-even branch) | `BFloat16.RoundToNearestEvenAtExactTies`, `BFloat16.RoundingOverflowAtTheTopOfTheRangeBecomesInfinity` -- NOT caught by the exhaustive round-trip test, by construction: that test's inputs are bf16 bit patterns widened losslessly to `float` first, so every value it round-trips already has all-zero bits in the range truncation vs. rounding would differ on, which is exactly why the two dedicated tie/overflow tests exist separately |
| 3 | `narrow_float()`'s `result.absolute_change` forced to `0.0` regardless of the computed delta | `float_bits.cpp:301` | `FloatBits.OwnerScenario_X86ExtendedLongDoubleNarrowsToMacDoubleWithReportedDelta`, `TypedRestart.FloatNarrowingReportsAMeasuredChange` (`restart_test.cpp`), `TypedRestartHdf5.HandBuiltX86LongDoubleExactAndWideningRefuseAllowNarrowingReportsDelta` (reports `0` against an independently computed `1.0842021724855044e-19`) |

### Fortran build-configuration correction (carried forward from Round 2)

This round's own instructions repeated Round 1's original ("no Fortran
compiler in sandbox") premise and asked for `-DFUSE_ENABLE_FORTRAN=OFF`.
That premise is the SAME one Round 2 Item 7 already measured and corrected:
`gfortran` (GNU 16.2.0) is installed at `/opt/homebrew/bin/gfortran`, not
on `PATH` by default -- verified again this round (`gfortran --version`
reports `GNU Fortran (Homebrew GCC 16.2.0) 16.2.0`). Flagged to the owner
mid-round rather than silently deciding either way; the owner chose to
enable Fortran, matching Round 2's own established configuration
(`-DFUSE_ENABLE_FORTRAN=ON -DCMAKE_Fortran_COMPILER=/opt/homebrew/bin/
gfortran`). Built WITHOUT this flag first to confirm the two
`abi.conformance` Fortran-plugin tests (`LagOneGraphAllFourKindsMatch
Native`, `SubcycledMeanGraphAllFourKindsMatchNative`) fail identically
(`FortranPlugin: E-ABI-LOAD`) on the UNMODIFIED pristine tree under the
same flag -- confirming the failure is a build-configuration effect, not a
Slice 1b regression -- before reconfiguring with Fortran on and deleting
that temporary verification tree.

### Full tier-0 run, this round

With `-DFUSE_ENABLE_HDF5=ON -DFUSE_ENABLE_FORTRAN=ON -DCMAKE_Fortran_
COMPILER=/opt/homebrew/bin/gfortran -DFUSE_ENABLE_MPI=OFF` (otherwise
identical to Round 2's configuration, Debug, `FUSE_FP_REPRODUCIBLE` ON by
default): **100% tests passed, 0 tests failed out of 223** (up from Round
2's 219 -- this round added 4 new ctest entries: `fuse_model_bfloat16_test`,
`fuse_model_element_type_test`, `fuse_model_float_bits_test`, and
`fuse_runtime_typed_restart_hdf5_test`, the last gated by
`FUSE_ENABLE_HDF5`). A separate build with `-DFUSE_ENABLE_HDF5=OFF`
(otherwise identical) confirmed nothing else breaks without HDF5.

### Scope reductions declared this round (in addition to Round 2's, unresolved)

- **Integer narrowing under `AllowNarrowing` reports no measured change**
  (unlike float narrowing's `max_absolute_change`) -- an integer value
  that does not fit refuses outright (`E-RESTART-INTEGER-OVERFLOW`); one
  that DOES fit is exact by construction (narrowing a wider integer type
  to a narrower one that still holds the value loses nothing), so there is
  no partial-information case analogous to float rounding to report a
  magnitude for.
- **`int` -> `float` narrowing is not implemented/not authorized** --
  item 6's design restricts `int -> float` conversion to the `Widening`
  case where the float's significand width exactly covers the integer's
  range (`is_exact_widening()`'s int-to-float branch); a NARROWING int-to-
  float conversion (one that could lose integer precision) was never
  part of the owner's ten-item list and was not added speculatively.
- **`bool` -> `float` widening is not implemented** -- not named in any of
  the owner's ten items; `Bool` only interconverts with itself under every
  `ReadConversion` mode in this round's implementation.
- **No Python binding code was written** (item 10, read-only by
  instruction).
- **Form A's element-type tag is a documentation note only** (item 9), not
  an actual ABI/`entries.def` change -- no Form-A struct exists yet for it
  to attach to.
