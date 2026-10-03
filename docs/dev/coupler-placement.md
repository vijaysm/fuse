# The coupler-placement track

Design C of `plans/2026-09-23-coupler-placement-and-migration.md` (plans/ is
gitignored; this page is the durable record). The user's own statement of the
requirement: *"The coupler by design is distributed and target PEs act as
coupler PEs as well. However, there are several workflows that may require
the coupler to be defined separately as a component, in which case we need
to make the connection from source to coupler, and then coupler to target
seamlessly, instead of just transferring data from source to target
directly."*

## The representation decision

The IR's placement space is the component index: `ValueType::rank_set`,
`ComponentId` everywhere in the Runner, `MeshBackend::placed_view(MeshId,
ComponentId)`. A coupler is therefore **another entry in the same array**,
not a new kind of id:

```cpp
enum class ComponentKind { Physics, Coupler };
// ComponentDecl gains: ComponentKind kind = ComponentKind::Physics;
CouplingGraph::add_coupler(CouplerDecl{name, ranks});  // kind = Coupler
```

This keeps every existing `component_count()` reader and every
`placed_view()` caller working unchanged — a coupler is just a component a
channel can point `RemapSpec::remap_on` at. The cost, which this decision
accepts deliberately: "component" now leaks into places that mean
"placement", and every one of those has to ask `kind` explicitly rather than
assume every array entry is a physics participant with state and a clock.

## The rule

One rule, not a second mode. A channel's Remap runs on some placement `C`
(`remap_on`, default: the target's own component). The value moves
source -> `C` (Transport + Epoch, or nothing when `C` is the distributed
remap operator's own move — fuse/model/remap_kind.hpp), is remapped on `C`'s
decomposition of both meshes, and moves `C` -> target (Transport + Epoch).
Any hop between identical placements (`RankSet::same_as`, not `overlaps`) is
elided, exactly like every other hop in the lowering. `C = target` (the
default) must reproduce, and does reproduce (golden IR unchanged;
`CouplerLowering.CouplerEqualToTargetReproducesTodaysIRExactly`), today's IR
byte for byte.

`src/ir/rules_channel.cpp`'s `coupler_rank_set` is `remap_on`'s value only
when `remap_on` **could ever matter** — `could_ever_emit_remap()`
(fuse/model/remap_kind.hpp): explicit remap flags, differing regions, or a
placement-moving method whenever the two ends differ. When none of those
hold, no choice of placement could produce a Remap at all, so `remap_on` is
ignored and the lowering is exactly the no-coupler shape; `validate()` warns
`W-CHANNEL-REMAP-ON-UNUSED` on the same predicate, so the warning and the
lowering can never disagree about which channels a coupler declaration
touches.

## The enumerated call sites

Every loop over `component_count()`/`ComponentId` in `src/schedule`,
`src/runtime`, `src/ir` and `src/model` that could mean "a physics component
with state and a clock" rather than "a placement", checked against
`ComponentKind` where it mattered:

| Site | What it does | Coupler-safe? |
|---|---|---|
| `include/fuse/schedule/clock.hpp` `ClockSet` | one `Clock` per component, dense array | **Fixed** (Conformance Unit 2): a coupler gets a `Clock` object (kept dense so `ComponentId` indexing stays valid) and nothing ever calls `advance()` on it, true -- but the original text stopped there, and missed that `common_period()`/`finest_interval()`/`checked_common_period()`/`is_lockstep()`/`earliest_next()`/`due_at()` all folded EVERY clock's own `interval()` into their answer regardless of whether `advance()` is ever called on it. `ComponentDecl::interval` defaults to `Rational{1}` (a coupler declares no interval of its own), so adding a coupler to a window-scoped graph whose real period was shorter than 1 silently pulled `lcm(period, 1) = 1` into the compiled schedule (measured: the Brusselator's own 1/50 window re-ran 50 times before one period elapsed). Every one of those six methods now skips a `ComponentKind::Coupler` entry explicitly (`coupler_[i]`, parallel to `clocks_`) |
| `src/runtime/runner.cpp` `build_ticks()` | matches every graph component to a supplied `Component`, generates its `Tick`s | **Fixed**: `ComponentKind::Coupler` entries are skipped in both directions — no `Component` implementation is required, and no `Tick` is generated. Pinned by `Runner.ACouplerNeedsNoComponentAndDeliversTheIdenticalValue`, `Runner.ACouplerContributesNoTicks` |
| `src/runtime/runner.cpp` `attach_comm_graphs()` | one `MPI_Comm_split` per component, colored by its `RankSet` | Already coupler-safe: loops over every `component_count()` entry unconditionally, so a coupler's ranks get a communicator exactly like any physics component's — this is what lets a channel's Transport/Epoch reach them at all |
| `src/runtime/runner.cpp` `resolve_remap()` (the identity check, `moves_placement`, `RemapPlan::placement`) | decides where the operator runs and whether it moves data itself | Already coupler-safe: reads the **arc's own** `type.rank_set` (set by lowering to `C`), never re-derives the placement from the channel's declared source/target |
| `src/runtime/runner.cpp` `resolve_remap()`'s MLS-serial and nodal-parallel-refusal checks (`src_component`/`dst_component`, ~line 594) | decides whether a method needs the distributed path, sized by rank COUNT | **Open question, not fixed** — see below |
| `src/runtime/runner.cpp` `resolve_epoch_participants()` (~line 2260) | the ranks that must reach one rank-agreement point | Already coupler-safe: reads `ins.attrs.src_rank_set`/`dst_rank_set` directly, so a coupler-routed Epoch's participants resolve correctly with no change |
| `src/schedule/scheduler.cpp` `rank_participates()`/`epoch_sequence()` | per-rank Epoch sequence, and the Transport/Epoch rank-set-match check | Already coupler-safe: purely a function of `attrs.src_rank_set`/`dst_rank_set`, which name whatever placement lowering put there |
| `src/runtime/iteration.cpp` `resolve_regions()`'s `R.participants` | the ranks a region's residual reduction runs over | **Fixed**: was built only from `derive_checkpoint_set()`'s `components` (a coupler is never one — it owns no state, so no `Source` node ever names it), which misses a coupler's ranks even though its Transport/Epoch instructions sit inside the region body. Now also unions in every in-region Transport/Epoch's rank sets. Needs `RunnerConfig::base_communicator`, i.e. real MPI, to exercise — **not assessed**, no host run available this track |
| `src/ir/checkpoint.cpp` `derive_checkpoint_set()` | which components need `save_state()`/`restore_state()` | Correct by construction: keyed off `OpKind::Source` provenance, and a coupler has no fields and therefore no `Source` node, so it can never be named |
| `src/runtime/runner.cpp` `spacetime_window()`/the slab decomposition (~lines 1145, 1232) | a channel's temporal interval, from its OWN source/target component's clock | Correctly NOT coupler-aware: a coupler has no clock, and space-time timing is about the channel's declared endpoints, never about `remap_on` |
| `src/model/yaml_io.cpp` | text (de)serialization | **New code**: couplers get their own `couplers:` list (no `fields`/`interval` to nest); `remap_on:` resolves through the SAME name table `source`/`target` use |
| `include/fuse/model/descriptor.hpp` | binary (de)serialization | **New code**: `kGraphVersion` 5 -> 6; a component record gains a `kind` byte, a channel record gains a `remap_on` word |
| `include/fuse/model/validate.hpp` | authoring-time checks | **New code**: `E-COUPLER-FIELDS`, `E-CHANNEL-REMAP-ON`, `W-CHANNEL-REMAP-ON-UNUSED`, and the `E-PLACEMENT-EMPTY` extension to a used `remap_on` |

## The open question: resolve_remap's parallel-safety refusals

`resolve_remap()`'s MLS-serial (`E-RT-REMAP-MLS-PARALLEL`) and nodal-parallel
(`E-RT-REMAP-NODAL-PARALLEL`) refusals size their "is this genuinely
partitioned" check from `src_component = graph.owner_of(channel.source)` and
`dst_component = graph.owner_of(channel.target)` — the channel's DECLARED
endpoints, not the operator's actual runtime placement (`in_rank_set`/
`placement`, computed a few lines earlier in the same function from the
arc's own `type.rank_set`). For a coupler-routed channel these can differ:
when the coupler is a genuinely separate, multi-rank placement, the
refusal's rank COUNT should come from the coupler, not from the target.

This was deliberately **not changed**. The existing derivation was tuned
through several rounds of real host MPI debugging (the function's own
comments name "Round 4", "Round 5" — each fixing a case only a genuine
multi-rank run exposed), and this sandbox cannot start MPI at all, not even
`-np 1`. A change here that cannot be verified against a real multi-rank run
risks reintroducing exactly the class of bug those rounds fixed. Flagged for
whoever has host access to test it: build a coupler-routed nodal
`"consistent"`/`"consistent-transpose"` channel where the coupler spans more
than one rank and the channel's declared source/target are each on a single
rank, and check whether the refusal (or its absence) is still correct.
