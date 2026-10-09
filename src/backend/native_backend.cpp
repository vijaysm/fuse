#include "fuse/backend/native_backend.hpp"
#include "fuse/backend/native_mesh_io.hpp"

#include <map>
#include <string>
#include <utility>

#include "fuse/base/error.hpp"

namespace fuse::native {

struct NativeMeshBackend::Impl {
  std::vector<std::unique_ptr<NativeMeshView>> meshes;
  std::vector<std::string> names;
  // Keyed by (mesh index, ComponentId value); MeshBackend's own contract
  // reserves an invalid ComponentId to mean "the default decomposition",
  // so that key never appears here.
  std::map<std::pair<std::uint32_t, std::uint32_t>, std::unique_ptr<NativeMeshView>> placed;
};

NativeMeshBackend::NativeMeshBackend(int rank, int num_ranks)
    : impl_(std::make_unique<Impl>()), rank_(rank), num_ranks_(num_ranks) {}
NativeMeshBackend::~NativeMeshBackend() = default;
NativeMeshBackend::NativeMeshBackend(NativeMeshBackend&&) noexcept = default;
NativeMeshBackend& NativeMeshBackend::operator=(NativeMeshBackend&&) noexcept = default;

Result<void> NativeMeshBackend::load(const MeshDecl& decl) {
  constexpr std::string_view kPrefix = "generated:";
  std::unique_ptr<NativeMeshView> view;

  if (decl.source.rfind(kPrefix, 0) == 0) {
    const std::string spec = decl.source.substr(kPrefix.size());
    auto arrays_r = generate_native_mesh(spec, rank_, num_ranks_);
    if (!arrays_r.ok()) return arrays_r.error();
    auto view_r = NativeMeshView::create_from_host(arrays_r.value(), rank_);
    if (!view_r.ok()) return view_r.error();
    view = std::move(view_r.value());
  } else {
    auto arrays_r = read_native_mesh_file(decl.source);
    if (!arrays_r.ok()) return arrays_r.error();
    auto view_r = NativeMeshView::create_from_host(arrays_r.value(), rank_);
    if (!view_r.ok()) return view_r.error();
    view = std::move(view_r.value());
  }

  impl_->meshes.push_back(std::move(view));
  impl_->names.push_back(decl.name);
  return Result<void>::success();
}

Result<void> NativeMeshBackend::register_default_placements(const CouplingGraph& graph,
                                                             MeshId id) {
  const MeshDecl& decl = graph.mesh(id);
  constexpr std::string_view kPrefix = "generated:";
  if (decl.source.rfind(kPrefix, 0) != 0) {
    // A file mesh cannot be regenerated on an arbitrary RankSet -- see
    // this method's own header comment.
    return Result<void>::success();
  }
  const std::string spec = decl.source.substr(kPrefix.size());

  // EVERY component in the graph, unconditionally -- not just the ones
  // that look like they own a field on this mesh or are named by some
  // channel's remap_on. Tried the narrower, field-ownership-plus-
  // coupler-routing version first and it was WRONG: a plain (no
  // remap_on) cross-mesh Remap resolves its weights on `dag.arc(ins.
  // out).type.rank_set` (runner.cpp's own resolve_remap()), which is a
  // DEDUPLICATED RankSet index, not necessarily the owning component's
  // own ComponentId -- two components sharing an IDENTICAL RankSet
  // (the common tier-0 case: every component on rank 0) collapse to
  // the SAME index, which can then numerically coincide with a
  // DIFFERENT component's own id and ask for that component's
  // placement on a mesh it owns no field on at all (confirmed: AbiNegotiation's
  // two-component lag1 fixture, both components on RankSet{0,1}, hit
  // exactly this). Reverse-engineering every path that can derive a
  // placement this way is not worth it when every placement this
  // backend can ever generate is the SAME cheap, deterministic
  // (rank_, num_ranks_) view regardless of which component asks for it
  // (tier-0: one process IS every placement) -- so hand out all of
  // them.
  for (std::uint32_t c = 0; c < graph.component_count(); ++c) {
    // EACH PLACEMENT OVER ITS OWN RANKS. The paragraph above is right that
    // the view is cheap and deterministic, and was wrong about what it
    // should be a function of: `(rank_, num_ranks_)` is this process's
    // place in the BASE communicator, so every component was handed this
    // rank's slice of a decomposition over every rank in the run --
    // "one process IS every placement", which holds on one process and
    // fails the moment a component's rank set is a strict subset of it.
    //
    // Measured on the host 2026-10-09, the first run with each component
    // in its own executable: a 176-cell mesh declared on U (one rank) and
    // on a coupler (one other rank) gave U 96 cells and the coupler 80 --
    // one decomposition cut across two placements instead of each holding
    // its own whole mesh. The exchange then had no source for the ids the
    // other side wanted, which is what the destination saw
    // (`CommGraph`'s NoMatch; KNOWN_ISSUES D24).
    //
    // A component's decomposition is over ITS ranks, indexed by this
    // process's position WITHIN that rank set -- so a one-rank component
    // gets the whole mesh wherever its rank happens to sit in the base
    // communicator. On one process this is the same answer as before
    // (every rank set is {0}, position 0 of 1), which is why no tier-0
    // test moved.
    const RankSet& rs = graph.component(ComponentId{c}).ranks;
    int placement_rank = rank_;
    int placement_size = num_ranks_;
    bool hosted_here    = true;
    if (!rs.empty()) {
      hosted_here = rs.contains(rank_);
      if (hosted_here) {
        const auto& r  = rs.ranks();
        placement_rank = static_cast<int>(
            std::lower_bound(r.begin(), r.end(), rank_) - r.begin());
        placement_size = static_cast<int>(rs.size());
      }
    }
    std::unique_ptr<NativeMeshView> view;
    if (hosted_here) {
      auto arrays = generate_native_mesh(spec, placement_rank, placement_size);
      if (!arrays.ok()) return arrays.error();
      auto view_r = NativeMeshView::create_from_host(arrays.value(), rank_);
      if (!view_r.ok()) return view_r.error();
      view = std::move(view_r.value());
    } else {
      // Outside the processor set: an EMPTY view of the right dimension,
      // which is what MoabMeshBackend::placed_view already answers there
      // (mesh_view.hpp's contract). Not a failure -- every rank asks for
      // every placement, because the exchange plan is collective and both
      // sides must be resolvable everywhere; only the ranks that host a
      // side read anything from it.
      MeshArrays<Kokkos::HostSpace> empty;
      const auto                    dflt = this->view(id);
      empty.dim = dflt.ok() ? dflt.value()->dimension() : 0;
      auto view_r = NativeMeshView::create_from_host(empty, rank_);
      if (!view_r.ok()) return view_r.error();
      view = std::move(view_r.value());
    }
    const Result<void> set = set_placed_view(id, ComponentId{c}, std::move(view));
    if (!set.ok()) return set;
  }
  return Result<void>::success();
}

Result<void> NativeMeshBackend::load(const MeshDecl& decl, const CouplingGraph& graph) {
  const Result<void> loaded = load(decl);
  if (!loaded.ok()) return loaded;
  const MeshId id{static_cast<std::uint32_t>(impl_->meshes.size() - 1)};
  return register_default_placements(graph, id);
}

Result<MeshId> NativeMeshBackend::mesh_id(std::string_view name) const {
  if (name.empty()) {
    return Failure{"E-MESH-NAME", "the empty mesh name matches nothing"};
  }
  std::size_t found = 0, first = 0;
  for (std::size_t i = 0; i < impl_->names.size(); ++i) {
    if (impl_->names[i] == name) {
      if (found == 0) first = i;
      ++found;
    }
  }
  if (found == 0) {
    return Failure{"E-MESH-NAME",
                   fmt::format("no loaded mesh is named '{}' ({} loaded)", name,
                               impl_->names.size())};
  }
  if (found > 1) {
    return Failure{"E-MESH-NAME-AMBIGUOUS",
                   fmt::format("{} loaded meshes are named '{}'", found, name)};
  }
  return MeshId{static_cast<std::uint32_t>(first)};
}

Result<const MeshView*> NativeMeshBackend::view(MeshId id) const {
  if (!id.valid() || id.value() >= impl_->meshes.size()) {
    return Failure{"E-MESH-ID",
                   fmt::format("no mesh with id {} ({} loaded)", id.value(),
                               impl_->meshes.size())};
  }
  return static_cast<const MeshView*>(impl_->meshes[id.value()].get());
}

Result<std::size_t> NativeMeshBackend::region_size(const Region& r) const {
  if (!r.mesh.valid() || r.mesh.value() >= impl_->meshes.size()) {
    return Failure{"E-MESH-ID",
                   fmt::format("region '{}' names mesh id {}, of which {} are loaded", r.name,
                               r.mesh.value(), impl_->meshes.size())};
  }
  const NativeMeshView& mesh = *impl_->meshes[r.mesh.value()];
  if (r.dimension < -1) {
    return Failure{"E-MESH-DIM",
                   fmt::format("region '{}' has dimension {}", r.name, r.dimension)};
  }
  if (!r.entity_set.empty() && r.entity_set != "all") {
    return Failure{"E-MESH-SET",
                   fmt::format("the native backend defines no named entity set '{}' -- only "
                              "the whole mesh (\"\" or \"all\")", r.entity_set)};
  }
  const int dim = r.dimension >= 0 ? r.dimension : mesh.dimension();
  if (mesh.entity_count(dim) == 0 && dim != 0 && dim != mesh.dimension()) {
    return Failure{"E-MESH-DIM",
                   fmt::format("region '{}' asks for dimension {}, which this mesh does not "
                              "represent (mesh dimension {})", r.name, dim, mesh.dimension())};
  }
  return mesh.entity_count(dim);
}

Result<void> NativeMeshBackend::set_placed_view(MeshId id, ComponentId placement,
                                                std::unique_ptr<NativeMeshView> view) {
  if (!id.valid() || id.value() >= impl_->meshes.size()) {
    return Failure{"E-MESH-ID", fmt::format("no mesh with id {}", id.value())};
  }
  if (!placement.valid()) {
    return Failure{"E-MESH-PLACEMENT",
                   "set_placed_view needs a valid ComponentId; an invalid one always means "
                   "\"the default decomposition\" and cannot be overridden"};
  }
  impl_->placed[{id.value(), placement.value()}] = std::move(view);
  return Result<void>::success();
}

Result<const MeshView*> NativeMeshBackend::placed_view(MeshId id, ComponentId placement) const {
  if (!placement.valid()) return view(id);
  if (!id.valid() || id.value() >= impl_->meshes.size()) {
    return Failure{"E-MESH-ID", fmt::format("no mesh with id {}", id.value())};
  }
  const auto it = impl_->placed.find({id.value(), placement.value()});
  if (it == impl_->placed.end()) {
    return Failure{"E-MESH-PLACEMENT",
                   fmt::format("mesh {} has no decomposition registered for placement {}; "
                              "call set_placed_view first, or use the default view()",
                              id.value(), placement.value())};
  }
  return static_cast<const MeshView*>(it->second.get());
}

Result<std::size_t> NativeMeshBackend::placed_region_size(const Region& r,
                                                          ComponentId placement) const {
  if (!placement.valid()) return region_size(r);
  const auto view_r = placed_view(r.mesh, placement);
  if (!view_r.ok()) return view_r.error();
  const int dim = r.dimension >= 0 ? r.dimension : view_r.value()->dimension();
  if (!r.entity_set.empty() && r.entity_set != "all") {
    return Failure{"E-MESH-SET",
                   fmt::format("the native backend defines no named entity set '{}'",
                              r.entity_set)};
  }
  return view_r.value()->entity_count(dim);
}

}  // namespace fuse::native
