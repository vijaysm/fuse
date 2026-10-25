# Restart-to-disk and field output

Sections 0-8 are the design; Section 9 describes what is implemented and its limitations. Every
claim is a citation to `file:line`, not an assumption about what the code "should" do.

Fixed design decisions: the restart format is a custom HDF5 schema; the mesh is written once and
state is written per restart point; field data is keyed by global id so a run can restart on a
different rank count; component state is carried as an opaque, versioned record; an XDMF (`.xmf`)
sidecar lets ParaView/VisIt read the restart file directly; a CF/UGRID NetCDF-4 writer is a
second, output-only deliverable.

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

Read from the Runner, the iteration machinery, the accelerators and the four worked examples,
not assumed. Each row: owner (class/member), distribution, whether a global-id
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

**This is the state inventory's central negative result: every accelerator buffer is indexed by position in the stacked interface vector
(`RegionBlock::offset`/`length`, `src/runtime/runner_impl.hpp`), which is an artifact of how
the current rank's own `RegionRuntime` concatenates its hosted blocks -- it has no relationship
to a mesh global id and is not even stable across a rank-count change for the SAME graph**,
because the set of blocks a rank hosts, and therefore the stacked vector's own layout, is a
function of `WindowMemberConfig::hosted`/`consumer_hosted` (`src/runtime/runner_impl.hpp:1098,
1127` -- explained at length in that file's own comment), which depends on the
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
read from a stale cache). This is the "which is recomputable, which is not" split: remap/transport state is recomputable; **the mesh and the field values are not**.

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
restart-relevant state beyond what Section 1.8 already covers; flagged as an open item if one exists.

### 1.10 Summary -- what has NO global-id index today

Explicitly: window waveform/history element position
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
| `fuse::linalg::DistVector` | **None.** `DistVector`'s own header states its layout is "owned block first" in "its own local order" (`include/fuse/linalg/dist_vector.hpp:7-8,24-26`); `GhostPattern::remote_local_index` is explicitly "never a global id" (`comm.hpp:79-83`) | N/A -- no id space exists | **Cannot**, today. A `DistVector` built directly (not through a `MeshView`-backed field) has no way for a reader at a different rank count to know which logical row its position 0 corresponds to. This is the concrete gap: it is not a hypothetical "a component only offering an opaque blob," it is `DistVector` itself when used without a mesh behind it (e.g. a hand-assembled linear system in a Newton region) |
| Window waveform samples (`WindowMemberState::history`/`last_window_waveform`) | None (Section 1.2) -- indexed by element position in the member's own interface vector | N/A | Not re-partitionable by global id today; restarts at the SAME rank count only, unless the member's element ordering is itself tied to a global id the component chooses to expose (an open question) |
| Brusselator example | Native mesh, `cell_gid = ix + iy*nx` (Section 2.2); `NativeGrid::make` recovers the local-index<->(i,j) correspondence by reading `MeshView::centroid` back (`examples/brusselator/brusselator_physics.hpp:23-25`) | Yes | Reader computes `(ix, iy)` from the stored global id, reads by hyperslab into whichever row band this rank's own partition owns |
| dn_heat example | **None.** `DirichletSlab`/`NeumannSlab` are placed one-component-per-rank (`examples/dn_heat/main.cpp:176,179`: `ranks = {0,1}` and `{1,1}` -- i.e. the whole component lives on exactly one rank each), and each component's own `u_in_`/`q_`/`u_`/`q_in_` (`examples/dn_heat/dn_component.hpp:380-382,519-521`) is a plain, fully-owned `std::vector<double>` -- not partitioned across ranks at all | N/A | Not applicable: there is no sub-component partitioning to re-partition. A restart that moves this component to a DIFFERENT rank still reads the whole array whole; "rank-count independence" for this example means only "the run as a whole may use a different number of ranks," never "this component's own array is split differently" |
| Gray-Scott example | Same pattern as dn_heat: `GrayScottSpeciesComponent::state_` (`examples/gray_scott/species_component.hpp:278`) is a single-rank-owned array, no `MeshView` in this example (confirmed: no `MeshView`/`NativeMeshView` reference anywhere under `examples/gray_scott/`) | N/A | Same as dn_heat |

**This table is the direct, measured answer to which components can restart on a different rank count.** Only the
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
  // or a component-chosen numbering for a non-mesh index space).
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

The recommended default, following the decision that field data is keyed
by global id generally: implement both interfaces, document Option A as the one a component
MUST implement to be restarted on a different rank count (checked, not trusted -- a component
that implements only Option B is tagged in the schema's component-record as "same-decomposition
only," Section 4, and the reader refuses it collectively if the restart's rank count differs
from the write's, exactly as a pure-Option-B component would be refused). This matches
Section 2.3's finding exactly: dn_heat/Gray-Scott/fsi_channel are honest Option-B candidates (no
partitioning to lose), Brusselator is the one example for which Option A is load-bearing, not
decorative.

**Resolution:** a component may declare "I offer Option B only" as a first-class, intentional
choice (for example a legacy or third-party physics code that cannot be modified). The cost is
that "is this run restartable at this rank count" becomes a per-component property rather than a
single yes/no for the graph; the graph refuses a rank-count change collectively when any
component offers only Option B (Section 9).

---

## 4. HDF5 schema

### 4.1 Group/dataset/attribute layout

```
/                                   (root)
  format_version        = <uint32>           (bumped per Section 4.4's rule)
  fuse_version          = <string>           (the library version string)
  descriptor_hash       = <uint64>           (fnv1a over serialize(graph), see below)
  written_by_rank_count = <uint32>           (MPI_Comm_size() at write time; Option-B refusal key, Section 3)
  mesh_checksum         = <uint64>
  (names as written by both writers and read by the reader; this table's draft used "@"-prefixed
   names and a format_name attribute that were never implemented)

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
little-endian like every other canonical encoding in `fuse/base/bytes.hpp` -- a known open limitation.
**Consequence: `@descriptor_hash` is currently only comparable between hosts of the same
endianness**, which this design does not fix but must document at the point the hash is used, so
a cross-architecture restart failure is correctly diagnosed as the descriptor's host-endianness,
not as a restart defect.

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
  is a tuning question deferred (robustness over efficiency: get a correct
  collective write first).
- **Who writes replicated data** (the mesh, Section 4.1's `/mesh` group; provenance attributes):
  rank 0 writes, every other rank's corresponding collective call participates with a zero-size
  selection -- the ordinary parallel-HDF5 pattern for "logically global, physically replicated"
  data, so the write is still one collective call every rank enters (never a rank-0-only
  `H5Dwrite` that leaves other ranks not calling the collective operation at all, which would be
  exactly the "collective path returns on a locally computed condition" mistake the project's
  collective-call rule forbids).

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
  files instead of one growing one). This decision is still open.
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
  geometry instead of an explicit connectivity list -- an optimization, not required for
  correctness, and deferred (robustness first).
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
  claiming UGRID conformance it cannot deliver. This split is an open question: whether volume output is wanted
  at all, given the honest answer is "CF, not UGRID, for a volume mesh".
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
**Precisely what can and cannot be bit-identical, stated precisely:**

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
  is identical," checked to the derived tolerance the project's convention requires (a
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
  derives from, per the project's tolerance convention.

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

## 8. Build options and constraints

- **CMake option**: `FUSE_ENABLE_HDF5` (default OFF, following the `FUSE_ENABLE_MOAB`/
  `FUSE_ENABLE_PETSC` naming convention, `cmake/FuseDependencies.cmake:119-122`), not folded into
  `FUSE_ENABLE_LOWFIVE`. LowFive's interception is in-memory only, "no file ever reaches disk"
  (`docs/user/lowfive.md:6-7`). Reusing LowFive's HDF5 *discovery* (the `find_package(HDF5 REQUIRED
  COMPONENTS C)` call, `cmake/FuseDependencies.cmake:1054`) is correct, but the restart writer must
  call `H5Fcreate`/`H5Dwrite` with the default (non-VOL) property list, never LowFive's
  `H5VOLProperty`, or the restart write would silently become in-memory-only too. Because the
  restart writer and LowFive would link the same HDF5 into one process for different purposes
  (disk I/O vs. VOL interception), the two-HDF5-image hazard check (`tools/probe-hdf5-images.sh`,
  `cmake/FuseDependencies.cmake:1107-1217`) that gates `FUSE_ENABLE_LOWFIVE` plus
  `FUSE_ENABLE_MOAB` applies to this combination as well.
- **Period boundaries only**: restart-to-disk cannot capture a period mid-suspension (an
  externally driven ABI participant's coroutine frame, `src/runtime/resume.hpp`). Restart points
  are period-boundary only, by construction. This is a design constraint (Section 1.1), not a gap
  to close later.
- **XDMF sidecar** acceptance is manual inspection (open ParaView/VisIt against the sidecar); it
  cannot be automated in a headless build.
- **NetCDF-UGRID output** is independent of restart (Section 6: output-only). It would use its own
  option following the same naming convention. Brusselator is the natural first example, being
  the only one of the four examples with a mesh to put a UGRID topology variable over
  (Section 2.3).

---

## Summary table of file:line citations used above

Every citation above is repeated here with no new claims.

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

---

## 9. Implementation

### 9.1 What is built

**Form A and Form B.** Form A is primary; Form B is same-rank-count only. A graph refuses a
rank-count change collectively if any component offers only Form B. The two forms are a
capability-bit pair, `kCapabilityRestartFormA`/`kCapabilityRestartFormB`
(`include/fuse/model/capabilities.hpp`, bits 6-7), checked by `fuse::check_restart_rank_count()`
(`src/runtime/restart.cpp`). `FormBOnlyComponentOnDifferentRankCountRefuses` in
`test/runtime/restart_test.cpp` proves the refusal fires and that equal rank counts never trigger
it. `DirichletSlab`/`NeumannSlab` (`examples/dn_heat/dn_component.hpp`) and
`UComponent`/`VComponent` (`examples/brusselator/species_component.hpp`) set
`kCapabilityRestartFormA` and implement `write_restart()`/`read_restart()`.

**One batched uniqueness and coverage check.** FUSE verifies uniqueness and coverage of the
declared global size for all arrays of a restart point in one collective, never one per array.
`RestartWriter::put_field()` takes a `declared_global_size`, and
`Hdf5GroupWriter::finish_point()` (`src/restart_io/hdf5_restart_io.cpp`) runs one combined sort
over every field's `(name, id)` pairs. At a single rank this re-confirms what `put_field()`'s
local check already found (`BadGlobalIdNumberingRefuses`, `DeclaredGlobalSizeMismatchRefuses`).
The cross-rank form is `fuse::restart_io::collective_duplicate_and_coverage_check()`
(`include/fuse/restart_io/collective_check_mpi.hpp`). It takes an `int` (`MPI_Comm_c2f` handle)
rather than an `MPI_Comm`, following `fuse/linalg/comm.hpp`, because `tools/check-layering.py`
forbids `mpi.h` in public headers. It serializes `(name, id)` and `(name, declared_size)` pairs as
length-prefixed buffers and makes exactly two `MPI_Allgatherv` calls in total, after which every
rank runs the identical local check and reaches the same verdict without a further reduction. It
is covered by `test/parallel/restart_collective_mpi_test.cpp` (ranks 2 and 3).

**File layouts.** `fuse::restart_io::FileLayout::{SeparateMeshAndPoints,GrowingFile}`
(`include/fuse/restart_io/hdf5_restart_io.hpp`) are selectable at run time. `GrowingFile` is
copy-append-rename, never an in-place append: `create_restart_point()` copies the existing file to
a temporary name before opening it read-write for the new point, and `finish()` is the one atomic
rename in either layout. The cost is one file copy per point.
`GrowingLayoutAppendsASecondPointWithoutDamagingTheFirst` checks the first point is still readable.

**Accelerator state.** `Accelerator::export_restart_state()`/`import_restart_state()`
(`include/fuse/num/accelerator.hpp`) are implemented for every `AccelKind`: `None` and `Constant`
trivially, `AitkenAccel`/`SteffensenAccel` in `src/num/accel_relaxation.cpp`, and
`QuasiNewtonAccel` (IQN-ILS/IQN-IMVJ) in `src/num/accel_quasi_newton.cpp`. The quasi-Newton export
covers `v_`/`w_` (one dataset per column, `<prefix>/iqn/v/col<j>` and `.../w/col<j>`),
`x_new_prev_`, `r_prev_`, `have_prev_`, and the IQN-IMVJ folded inverse-Jacobian chain
(`blocks_`, `chain_reprieves_`). History is keyed by (field, global id), so it moves across a
rank-count change.

- *No silent reset.* `Accelerator::restart_state_implemented()` (default `false`) is checked by
  `Runner::restore_restart_fields()` before `import_restart_state()` is called. `false` without
  opt-in refuses with `E-RESTART-ACCEL-NOT-IMPLEMENTED`; with `allow_accelerator_reset_on_mismatch`
  it calls `Accelerator::reset()` and sets `any_accelerator_reset`. The base
  `import_restart_state()` unconditionally returns `E-RESTART-NOT-IMPLEMENTED` and is never
  reached. The check is collective by construction: every rank's graph assigns the same
  `AccelKind` to the same region, so the flag evaluates identically everywhere. The refusal tests
  (`AccelNotImplementedRefusesWithoutOptIn`, `AccelNotImplementedWithAllowResetSucceedsAndReports`)
  use a test-only subclass, `NeverImplementsRestartAccel`, injected through
  `Runner::debug_replace_region_accelerator()`.
- *Exact keys, no hash.* `fuse::RestartRowBlock` (`include/fuse/model/restart.hpp`, `{field_name,
  ids}`) is passed per block, and `detail::put_field_per_block()`/`get_field_per_block()`
  (`accelerator.hpp`, defined in `src/num/accel_relaxation.cpp`) write one dataset per block under
  `name_prefix + "/" + block.field_name`. Two blocks with identical local ids (for example
  `dirichlet_neumann`'s two blocks, which both report ids `0..n-1`) therefore cannot collide.
  `PerBlockKeysNeverCollideOnIdenticalLocalIds` drives this directly. A component's Form-A field
  name must equal the graph's `FieldSpec::name` for any field an accelerated channel feeds,
  because the row resolution looks the owning field up by `RegionBlock::target_field`
  (`src/runtime/iteration.cpp`); `docs/user/restart.md` states the convention.
- *Window-scoped blocks.* `RestartRowBlock::window_capacity` (0 means not window-scoped) makes the
  per-block helpers treat a block's slice of the flat buffer as `ids.size() * window_capacity`
  rows, element-major and sample-minor (`index(e, s) = e*window_capacity + s`), demultiplexed into
  `window_capacity` datasets `<qualified>/sample<k>`. This is needed because a window-scoped
  channel's `RegionBlock::length` is the K-wide stacked window-transfer buffer, not the registered
  field's single-sample size. `AitkenAccel`, `SteffensenAccel` and `QuasiNewtonAccel` need no
  change; only the helpers know about window scoping.
  `Restart.AitkenAccelWindowCapacityPerSampleRestartRoundTripsBitForBit` covers it in isolation.
- *Block order.* `region.blocks` does not iterate in ascending `RegionBlock::offset` order for
  `brusselator_window`, so `write_restart_fields()` and `restore_restart_fields()` sort a local
  copy by offset before building the `RestartRowBlock` list; the helpers' running-offset
  accumulation depends on it.

**Window-member state.** `detail::WindowMemberState::clock`, `straddle_pending`,
`straddle_window_start`/`straddle_window_end` (scalars) and `history` (per element, keyed by the
global ids of the member's exported field, one `time_num`/`time_den`/`value` triple per sample
under `__window/<owner>/<field>/history/sample<k>/`) are written and restored by
`write_window_member_restart()`/`restore_window_member_restart()` (`src/runtime/runner.cpp`).
`WindowMemberConfig::exported_field` (`src/runtime/runner_impl.hpp`, set in `resolve_regions()`)
lets this code find a member's qualified field name without a second graph walk.

**The Runner's clock.** `restore_restart_fields()` does not touch `Impl::period_index`, which the
Runner uses to compute `ctx.t_begin`/`ctx.t_end`. `Runner::set_current_period_index()` is called
from `fuse::restore_restart_point()` (`src/runtime/restart.cpp`) right after the fields succeed.
Without it, restored fields are correct immediately after restore but wrong after the next
`run_period()`.

**Typed fields.** Field and scalar I/O is templated on element type.

- `fuse::ElementType` (`include/fuse/model/element_type.hpp`) is a closed enum: `Int8`..`Int64`,
  `UInt8`..`UInt64`, `Binary16`, `BFloat16`, `Binary32`, `Binary64`, `LongDouble`, `Bool`.
  `ElementTypeOf<T>` resolves a C++ type at compile time; any other `T` fails a `static_assert`.
- `fuse::bfloat16` (`include/fuse/model/bfloat16.hpp`) is a storage type with 1 sign, 8 exponent
  (bias 127) and 7 explicit significand bits. `from_float()` rounds to nearest, ties to even, and
  preserves quiet NaN. `BFloat16`'s exhaustive test round-trips all 65536 bit patterns.
- `Bool` is stored as one `UInt8` byte; `validate_bool_bytes()` refuses any byte other than 0 or 1
  (`E-RESTART-BAD-BOOL-BYTE`).
- `put_field<T>()`/`get_field<T>()` and `put_scalar<T>()`/`get_scalar<T>()`
  (`include/fuse/model/restart.hpp`) are non-virtual templates over one virtual byte-based pair
  each (`put_field_bytes`/`get_field_bytes`, `put_scalar_bytes`/`get_scalar_bytes`). All callers
  use the templated interface; there is no double-only legacy overload. `get_scalar<T>()` takes an
  out-parameter `T&`, because a value and a report cannot share one `Result<>` slot. The
  period index is stored as `std::int64_t`, never cast through `double`.
- Per-field metadata (`element_type`, `byte_size`, `significand_bits`, `exponent_bits`,
  `exponent_bias`, `byte_order`, `is_signed`) is written as HDF5 attributes beside each dataset by
  `write_type_metadata()`. `read_and_verify_type_metadata()` builds a fresh `ElementTypeInfo` from
  the file's own attributes and checks it only for internal consistency against the dataset's
  actual HDF5 datatype (`E-RESTART-TYPE-METADATA-MISMATCH`); it never compares against this
  build's own table, because that would make a genuinely foreign-platform file unreadable under
  `AllowNarrowing`. An unrecognised `element_type` name refuses with
  `E-RESTART-UNKNOWN-ELEMENT-TYPE`.
- `ReadConversion::Exact`/`Widening`/`AllowNarrowing` (`element_type.hpp`) are dispatched in
  `convert_one_element()` (`hdf5_restart_io.cpp`): an identical-type fast path; `Exact` refuses
  any mismatch (`E-RESTART-TYPE-MISMATCH`); `Widening` allows only conversions that
  `fuse::detail::is_exact_widening()` accepts; float narrowing goes through
  `fuse::detail::narrow_float()` (`src/model/float_bits.cpp`) and reports a measured
  `max_absolute_change`; integer narrowing (`encode_int_checked()`) refuses overflow even under
  `AllowNarrowing` (`E-RESTART-INTEGER-OVERFLOW`) and signed-to-unsigned unconditionally
  (`E-RESTART-SIGNED-TO-UNSIGNED`). All float conversion is done in FUSE code
  (`include/fuse/model/float_bits.hpp`: `decode_float_bits()`, `encode_float_bits()`,
  `round_significand()`, `narrow_float()`), never HDF5's soft conversion (see 9.2).
  `TypedRestartHdf5.HandBuiltX86LongDoubleExactAndWideningRefuseAllowNarrowingReportsDelta` builds
  an x86-64-style 80-bit `long double` dataset by hand (mantissa `0x8000000000000001`, biased
  exponent 16383, explicit leading bit) and checks that `Exact` and `Widening` refuse while
  `AllowNarrowing` succeeds with `max_absolute_change` equal to `std::ldexp(1.0, -63)`.
- Accelerator and waveform histories stay `Binary64`.

**ABI.** Provider-side Form B is `save_restart_blob`/`load_restart_blob` in
`include/fuse/abi/entries.def`, using `fuse_restart_blob_sink`/`fuse_restart_blob_source` in the
same query-then-fill idiom as `fuse_graph_sink`, with capability bit `FUSE_CAP_RESTART_FORM_B`
(value 64). `ProviderAdapter::save_restart_blob()`/`load_restart_blob()`
(`src/abi/provider_adapter.cpp`) are the caller side. The participant-side entry,
`request_restart_point`, is a flag rather than a trigger, because the ABI's fixed `(ctx, in, out)`
shape cannot carry a file path or an HDF5 layout choice:
`Runner::request_restart_point()`/`take_restart_point_request()` is consume-not-peek, idempotent,
and polled by the embedding host after each `run_period()`/`resume_period()`
(`AbiParticipant.RequestRestartPointSetsTheRunnersFlagIdempotently`, through the test-only seam
`fuse::abi_test::runner_of(ctx)`). All generated ABI files and `test/abi/struct_sizes.golden` are
regenerated with `tools/gen-abi.py --write` and verified with `--check`.
`ConformanceTest.ProviderSaveAndLoadRestartBlobRoundTrips` is a full round trip through a
`dlopen`'d plugin. Form A has no ABI surface (see 9.4).

**Collective write.** `fuse::restart_io::create_restart_point_collective()`
(`include/fuse/restart_io/hdf5_restart_io_mpi.hpp`, `src/restart_io/hdf5_restart_io_mpi.cpp`) uses
`H5Pset_fapl_mpio` for the file access property list, an `MPI_Exscan`-derived per-rank row offset
into one collective hyperslab write per field (`H5FD_MPIO_COLLECTIVE`), collective `H5Dcreate2` of
a `declared_global_size`-sized dataset, `finish_point()` wired to
`collective_duplicate_and_coverage_check()`, and a collective rename-on-success
(`MPI_Allreduce(MIN)` to agree every rank's `put_field` succeeded, rank-0-only `std::rename`, a
second `MPI_Allreduce`/`MPI_Barrier` so no rank proceeds past `finish()` before the rename is
visible). It is compiled only under `FUSE_ENABLE_MPI`. The serial `Hdf5GroupWriter` is
single-writer and refuses a second `put_field()` for the same qualified name
(`E-RESTART-DUPLICATE-CALL`).

**Rank-count-independent read.** `Hdf5GroupReader::get_field_bytes()` resolves every row through
an id-to-position map built from the file's own `ids` dataset, never from the caller's request
order, so every rank reads its own local ids from a file written at any rank count.
`Restart.GetFieldResolvesRowsByIdNotByFilePosition` reads three ids in reversed order and checks
the values come back reversed.

**Brusselator.** `UComponent` keys `u_out`/`u_in` and its private `u_state`/`u_q_prev` by the U
grid's ids (`NativeGrid::global_id()` in `examples/brusselator/brusselator_physics.hpp`, filled once
in `NativeGrid::make()`). `VComponent` keys `v_out`/`v_in` by U-grid ids (both members use
`WindowMember::length = N_u`) and its private `v_state` by the V grid's own ids: two id spaces in
one component, as Section 2.2 anticipated. The only scalars are `double`, stored with
`put_scalar<double>()`. `test/verification/brusselator_restart_test.cpp` runs windows 0-2,
restarts, and compares windows 3-5 bit for bit with an uninterrupted reference, for both Jacobi
and Gauss-Seidel and both file layouts, plus descriptor-hash mismatch detection and the
window-scoped no-silent-reset tests.

### 9.2 Measured behaviour

- **Bit identity does not depend on `FUSE_FP_REPRODUCIBLE`.** The ULP divergence that option
  guards against comes from comparing two different code shapes over the same data
  (`cmake/FuseFpPolicy.cmake:1-19`). `SameRankRoundTripSeparateLayout`/`SameRankRoundTripGrowingLayout`
  compare one binary's two executions (an uninterrupted 5-period run, and a 3-period run, restore,
  plus 2 more periods) through the same compiled `advance()` and accelerator code, so there is
  only one code shape and `EXPECT_EQ` (no tolerance) on every field element is valid with the
  option on or off. They also compare both components' private state and the last period's
  `IterationStats::iterations`, so Aitken's dynamic relaxation factor and residual history are
  covered.
- **A Runner-level IQN-ILS round trip cannot observe `v_`/`w_`/`cols_`/`have_prev_` restoring
  wrongly.** A restart point is only taken at a window boundary, and the next window's first
  `update()` (`iteration == 0`) calls `reset()` before anything else, which discards those members
  whenever `retain_ == false` (every kind except IQN-IMVJ).
  `Restart.QuasiNewtonAccelExportImportRestartStateRoundTripsBitForBit` therefore drives
  `fuse::Accelerator` directly with `AccelKind::IQN_IMVJ` across two windows, so `blocks_` is
  populated before the restart point is written.
- **HDF5's soft type conversion refuses `H5T_NORM_MSBSET`.** Converting an x86-extended-style
  `long double` type (explicit leading significand bit) to `double` in `H5Dread()` fails in
  `H5T__conv_f_f()` with "normalization method not implemented yet" (HDF5 1.14.1-2). FUSE cannot
  rely on HDF5 conversion for any explicit-leading-bit format, hence the FUSE-side conversion
  pipeline.
- **`H5Tset_fields()`/`H5Tset_precision()`/`H5Tset_size()` ordering is direction-dependent**
  (`make_derived_float_type()` in `hdf5_restart_io.cpp`). For a precision decrease (for example
  `Binary32` to `BFloat16`), `H5Tset_fields()` must run before the precision/size shrink, or HDF5
  reports "exponent bit field size/location is invalid". For a precision increase (for example
  `Binary64` to an 80-bit extended type), size/precision must grow first, then `H5Tset_fields()`,
  or HDF5 reports "adjust sign, mantissa, and exponent fields first".
- **Every float `ElementType` builds its HDF5 type from its own `ElementTypeInfo`.**
  `make_hdf5_type_for()` never uses a native-type shortcut such as `H5T_NATIVE_LDOUBLE`, because
  that type (53 bits on the measured platform) forces HDF5 to soft-convert against a foreign file's
  80-bit type. The stored precision is `stored_mantissa_bits + exponent_bits + 1`, not
  `significand_bits + exponent_bits + 1`, because `ElementTypeInfo::significand_bits` already
  includes an implicit leading bit.
- **Decomposition-order round-off.** One restart window of Schur-decomposed solves disagrees with
  an undecomposed reference by 1.155e-14 (2 partitions) and 4.396e-14 (3 partitions)
  (`BrusselatorSpeciesCoreDecomposition.OneWindowAtRestartCadenceMeasuresDecompositionOrderSensitivity`).
  A restored, not-yet-advanced state therefore compares bit-identically by global id across any
  rank-count change (a restore is a file read), but a state advanced one further window after a
  restore at a different rank count must be compared with a measured tolerance; 1e-9 leaves three
  orders of margin over the serial measurement.
- **A parallel HDF5 build needs MPI headers even with MPI disabled.** A parallel HDF5's public header
  includes `<mpi.h>` even with `FUSE_ENABLE_MPI=OFF`, and `find_package(MPI REQUIRED COMPONENTS C)`
  in the HDF5 block of `cmake/FuseDependencies.cmake` must resolve a usable `MPI_C_COMPILER`
  (put the MPI prefix on `CMAKE_PREFIX_PATH`); otherwise configure succeeds with
  `MPI_C_COMPILER-NOTFOUND` and the build fails at `#include <mpi.h>`.

### 9.3 Verification

Tier 0 (no MPI): `test/runtime/restart_test.cpp`, `test/runtime/typed_restart_hdf5_test.cpp`,
`test/model/bfloat16_test.cpp`, `test/model/element_type_test.cpp`,
`test/model/float_bits_test.cpp`, `test/verification/brusselator_restart_test.cpp`, and the ABI
tests named above. Mutation rows for restart live in `build-evidence/mutations/restart.toml`; they
cover the accelerator not-implemented gate, per-block key collapse, `is_exact_widening()` always
true, `bfloat16::from_float()` reduced to truncation, and `narrow_float()` reporting a zero change.
The `bfloat16` rounding mutation is not caught by the exhaustive round-trip test, by construction
(its inputs are bf16 patterns widened losslessly, so truncation and rounding agree), which is why
dedicated tie and overflow tests exist.

Tier 1 (needs a working MPI): `parallel.restart_collective_mpi` and
`parallel.brusselator_restart_mpi` (ranks 2 and 3, gated on `FUSE_LINALG_BACKEND_IS_EIGEN` and
`FUSE_ENABLE_HDF5`). The latter exercises the restart_io layer directly over a Brusselator-shaped
field, not a full rank-partitioned Runner round trip, in four scenarios: write at N ranks and
restore at N; write at one rank and restore at the job size; write on a 2-rank sub-communicator
and restore at the job size; and write at the job size and restore at a smaller size. The last
cannot happen within one `mpiexec` invocation, so it reads a path named by
`FUSE_BRUSSELATOR_RESTART_MPI_FIXED_PATH` set by a two-invocation host sequence and skips, saying
why, when the variable is absent.

### 9.4 Limitations

- **Accelerated window-scoped round trip.** A Runner-level round trip with Aitken, Steffensen or a
  quasi-Newton accelerator over a window-scoped region (`brusselator_window`) does not reproduce
  the reference run's iteration count after a restore. `cfg.window_import`'s consumer-side
  transfer buffer is a coupler-owned slot primed once in `resolve_regions()` and never refreshed
  from restored state, so the first post-restore iteration starts from a stale seed in `R.x_old`.
  Re-priming the export slot (`Impl::reprime_window_transfer_buffers()`) is correct and kept;
  re-running the import channel's delivery chain made the mismatch worse and was not adopted. A
  future fix should trace `window_import`'s write timing across a continuous run. `AccelKind::None`
  is unaffected (its `update()` never reads `x_old`), and field and history restart are
  unaffected. The accelerator-state restore itself is covered by the isolated per-sample test
  above, not by a Runner-level round trip.
- **Gauss-Seidel windows with an accelerator.** A real accelerator over a
  `WaveformIteration::GaussSeidel` window refuses with `E-RT-WINDOW-ACCEL`
  (`resolve_regions()` in `src/runtime/iteration.cpp`); `docs/dev/newton-krylov.md` documents the
  same gap from the window-Newton side. This is outside restart.
- **`last_window_waveform`.** Restart code exists for it, but the Runner's `advance_window()` never
  populates the member (only the standalone `fuse::WindowCoupler` does), so only the "nothing to
  write" branch is reachable through the Runner.
- **Collective writer scope.** Only `ElementType::Binary64`/`Int64` are supported (anything else
  refuses `E-RESTART-TYPE-MISMATCH`), only `FileLayout::SeparateMeshAndPoints`
  (`GrowingFile` refuses `E-RESTART-COLLECTIVE-LAYOUT-UNSUPPORTED`), and scalars, rationals and
  blobs refuse with `E-RESTART-COLLECTIVE-SCALAR-UNSUPPORTED`, because HDF5 metadata operations
  are collective even for a single-rank-hosted scalar.
- **Mesh and provenance replication.** The rank-0-writes, others-zero-size pattern of Section 4.3
  is not implemented; Brusselator restart uses the `mesh_checksum = 0` sentinel.
- **Multi-rank execution.** The collective writer and the parallel tests have been compiled
  against an MPI-enabled build but not run in the development sandbox, where MPI cannot start.
  Writing at one rank and restoring at several is covered by the serial writer and the id-keyed
  reader; writing at several ranks and restoring at a different count needs a host with a working
  MPI.
- **Form A ABI surface.** Only Form B (opaque, same-decomposition-only bytes) has C/Fortran ABI
  entries. `entries.def` has no precedent for a nested-table or opaque-handle object, and every
  entry shares one flat `(ctx, in, out)` shape. A future Form-A field-descriptor struct would add
  a `std::uint8_t` element-type tag after the existing `byte_size`/`components` pair, append-only.
  There is no fault-injection provider variant in which `save_restart_blob`/`load_restart_blob`
  returns an error.
- **Python.** No binding code exists for restart. `ForwardingComponent` infers
  `kCapabilityRollback` from `hasattr(obj_, "save_state")`/`"restore_state"`
  (`python/src/component_bindings.cpp`); restart would need the same hasattr-gated pattern, new
  pybind11 `RestartWriter`/`RestartReader` wrappers over NumPy arrays (ids as `int64`, values as
  `float64`), and registration in `python/src/module.cpp`. Accelerators are never subclassed in
  Python, so they need no binding work. `fuse::bfloat16` has no standard numpy dtype: a binding
  would need `ml_dtypes` or a `uint16` view with a documented reinterpretation contract.
  `LongDouble` maps to numpy's `longdouble` only when both platforms' extended formats agree.
- **Type conversion.** Integer narrowing under `AllowNarrowing` reports no measured change (a value
  either fits exactly or refuses). `int` to `float` is allowed only as an exact `Widening`; a
  narrowing int-to-float conversion does not exist. `Bool` interconverts only with itself.
- **Not yet built.** The XDMF sidecar (Section 5) and the NetCDF-UGRID writer (Section 6) are
  designed but not implemented.
