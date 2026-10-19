#include "fuse/backend/remap_halo_instructions.hpp"

#include <cstdint>
#include <string>

#include "fuse/backend/native_backend.hpp"
#include "fuse/backend/native_halo.hpp"
#include "fuse/backend/remap_plan.hpp"
#include "fuse/base/config.hpp"  // FUSE_HAVE_MOAB
#include "fuse/model/component.hpp"  // fuse::RankSet
#if FUSE_HAVE_MOAB
#include "fuse/backend/moab_backend.hpp"  // MoabMeshBackend, for the dispatch below
#endif

namespace fuse {
namespace {

/// `plan.src`'s (or `plan.tgt`'s) region, resolved to this backend's own
/// `MeshId` and dimension -- the same small resolution `intx_weights.cpp`'s
/// file-local `supermesh_side()` performs for `fuse::build_weights()`
/// (not callable from here: file-local to that translation unit), and the
/// same one `Runner::Impl::resolve_remap_halo()` (runner.cpp) already
/// duplicates for its own verification. Kept here, not shared, for the
/// same reason that one stayed independent: this is a handful of lines
/// reading the graph and backend, not logic worth a new coupling between
/// two tracks' own files.
[[nodiscard]] Result<std::pair<MeshId, int>> resolve_mesh_and_dim(
    const CouplingGraph& graph, const MeshBackend& backend, RegionId region_id,
    const char* which) {
  const Region& r = graph.region(region_id);
  if (!r.mesh.valid() || r.mesh.value() >= graph.mesh_count()) {
    return Failure{"E-IR-REMAP-HALO",
                 std::string(which) + " region '" + r.name + "' names no valid mesh"};
  }
  const Result<MeshId> mid = backend.mesh_id(graph.mesh(r.mesh).name);
  if (!mid.ok()) return mid.error();
  int dim = r.dimension;
  if (dim < 0) {
    const Result<const MeshView*> view = backend.view(mid.value());
    if (!view.ok()) return view.error();
    dim = view.value()->dimension();
  }
  return std::make_pair(mid.value(), dim);
}

}  // namespace

Result<std::size_t> execute_remap_halo_instructions(const CouplingGraph& graph,
                                                     const OperatorDAG& dag,
                                                     MeshBackend& backend) {
  std::size_t total = 0;
  for (std::uint32_t i = 0; i < dag.node_count(); ++i) {
    const NodeId nid{i};
    const Node&  n = dag.node(nid);
    if (n.kind != OpKind::BuildRemapHalo) continue;

    // Defensive: `emit_remap_halo_instructions()` (src/ir/lowering.cpp)
    // always stamps a valid channel origin on this kind; a DAG built some
    // other way is the only way this could fire.
    if (!n.origin.valid() || n.origin.value() >= graph.channel_count()) {
      return Failure{"E-IR-REMAP-HALO",
                   "execute_remap_halo_instructions: BuildRemapHalo node " +
                       std::to_string(i) +
                       " carries no valid channel origin"};
    }

    const Channel& ch = graph.channel_at(n.origin);

    RemapPlan plan;
    plan.src = graph.field(ch.source).region;
    plan.tgt = graph.field(ch.target).region;
    plan.flags = ch.remap.flags;
    plan.method = ch.remap.method_override;
    // ROUND 2 (lead's design decision 2026-10-18): read the placement BACK from
    // the instruction's own `attrs.dst_rank_set`/`attrs.src_rank_set` --
    // the components `emit_remap_halo_instructions()` (src/ir/rules_
    // channel.cpp) already resolved when it decided to emit this node --
    // rather than re-deriving them a second time here. What this
    // instruction acts on is what lowering froze into it.
    if (n.attrs.dst_rank_set < 0 || n.attrs.src_rank_set < 0) {
      return Failure{"E-IR-REMAP-HALO",
                   "execute_remap_halo_instructions: BuildRemapHalo node " +
                       std::to_string(i) +
                       " carries no placement in attrs.src_rank_set/dst_rank_set"};
    }
    plan.placement = ComponentId{static_cast<std::uint32_t>(n.attrs.dst_rank_set)};
    const ComponentId source_placement =
        ComponentId{static_cast<std::uint32_t>(n.attrs.src_rank_set)};

    // ROUND 3 (KNOWN_ISSUES D5 HALO BRIDGING, 2026-10-18): the single-
    // placement `fuse::build_remap_halo(plan, graph, backend)` resolves
    // BOTH source and target through `plan.placement` -- wrong for the
    // source whenever `source_placement != plan.placement`, which is
    // every qualifying channel this instruction was emitted for (measured
    // on the owner's host: the halo "succeeded" having adopted nothing,
    // KNOWN_ISSUES.md D73's own account). Both backends now have a
    // two-placement form; dispatch to each directly rather than through
    // the generic single-placement entry point.
    Result<std::size_t> r = Failure{"E-REMAP-BACKEND", "unreachable"};
    if (auto* native_backend = dynamic_cast<fuse::native::NativeMeshBackend*>(&backend)) {
      const Result<std::pair<MeshId, int>> src_md =
          resolve_mesh_and_dim(graph, backend, plan.src, "source");
      if (!src_md.ok()) return src_md.error();
      const Result<std::pair<MeshId, int>> tgt_md =
          resolve_mesh_and_dim(graph, backend, plan.tgt, "target");
      if (!tgt_md.ok()) return tgt_md.error();
      if (src_md.value().second != tgt_md.value().second) {
        return Failure{"E-REMAP-DIM",
                     "execute_remap_halo_instructions: channel '" + ch.name +
                         "': source dimension " + std::to_string(src_md.value().second) +
                         " disagrees with target dimension " +
                         std::to_string(tgt_md.value().second)};
      }
      r = fuse::native::build_native_remap_halo(*native_backend, src_md.value().first,
                                                tgt_md.value().first, src_md.value().second,
                                                source_placement, plan.placement);
    } else {
#if FUSE_HAVE_MOAB
      // The MOAB cross-placement overload (remap_plan.hpp) for a MOAB
      // backend. Any other backend (the serial backend, a test double) has
      // no partitioned read and therefore nothing to adopt: a successful
      // no-op, the same answer the no-MOAB build below gives, so the two
      // configurations agree. Before 2026-10-19 this branch handed every
      // non-native backend to the MOAB overload, which refuses
      // E-REMAP-BACKEND for a serial backend; the refusal was invisible in a
      // no-MOAB tier-0 build and turned 30 serial rows red on a MOAB-enabled
      // host (KNOWN_ISSUES.md D73 round 8's host run).
      if (dynamic_cast<MoabMeshBackend*>(&backend) != nullptr) {
        r = build_remap_halo(plan, graph, backend, source_placement);
      } else {
        r = std::size_t{0};
      }
#else
      // The 4-argument overload's own DEFINITION lives only in
      // intx_weights.cpp's FUSE_HAVE_MOAB half (remap_plan.hpp declares
      // it unconditionally, which is a link error waiting to happen for
      // any tier-0 caller that references it) -- a build with no MOAB at
      // all can never reach a MoabMeshBackend instance here regardless
      // (the `native_backend` branch above already covers the only other
      // backend a no-MOAB build can construct), so this falls back to the
      // single-placement entry point's own tier-0 definition, which
      // answers the identical question (no backend here can need a
      // second placement without MOAB) without naming the undefined
      // symbol.
      r = build_remap_halo(plan, graph, backend);
#endif
    }
    if (!r.ok()) {
      return Failure{r.error().code,
                   "execute_remap_halo_instructions: channel '" + ch.name +
                       "': " + r.error().message};
    }
    total += r.value();
  }
  return total;
}

}  // namespace fuse
