#include "fuse/base/config.hpp"
#include "fuse/runtime/epoch_schedule.hpp"
#include "fuse/transport/tags.hpp"
#include "fuse/runtime/runner.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "runner_impl.hpp"
#include "fuse/num/kernels.hpp"
#include "fuse/schedule/slab.hpp"

#if FUSE_HAVE_MOAB
// Names no MOAB type -- that is the whole point of the header -- but the class
// only exists in a build that has the backend.
#include "fuse/backend/moab_backend.hpp"
#endif

#if FUSE_HAVE_MPI
#include <mpi.h>

#include "fuse/transport/comm_graph.hpp"
#endif

namespace fuse {

using detail::ConstrainInfo;
using detail::Resolved;
using detail::Slot;
using detail::SpacetimeInfo;
using detail::Tick;

namespace {

/// Does this registered view address `count * components` doubles as one
/// contiguous, component-fastest run?
///
/// The interpreter walks every buffer as a flat `double[n]` (that is what
/// `num::copy` and friends mean), so any other stride would be read as if it
/// were contiguous: no crash, just a silently permuted field. `FieldRef`
/// deliberately supports strided and blocked layouts, so this has to be
/// checked rather than assumed.
[[nodiscard]] bool is_flat_contiguous(const Layout& l) noexcept {
  if (l.components < 1) return false;
  if (l.count == 0) return true;
  const auto nc = static_cast<std::size_t>(l.components);
  if (nc == 1) return l.stride == 1;
  return l.stride == nc && l.effective_component_stride() == 1;
}

// ---------------------------------------------------------------------------
// Spec §5.4 (Task 5.5): when may a `Remap` be an element-wise copy?
// ---------------------------------------------------------------------------

/// The outcome of asking whether two regions are, cell for cell and index for
/// index, the same thing. `why` is filled in only when the answer is no, and
/// is written into the diagnostic so the user learns which of the several
/// possible reasons applied.
struct Coincidence {
  bool        same = false;
  std::string why;
};

/// Relative slack on a cell measure. Two regions that really are the same
/// entities give bit-identical measures through the same backend; the slack
/// exists for the case of one geometry described by two files, where the
/// coordinates may differ in the last ulp.
constexpr double kMeasureRelTol = 1e-12;

/// Slack on a centroid, RELATIVE TO THE CELL'S OWN SIZE (`measure^(1/dim)`).
/// The quantity being separated is "the same cell" (distance 0) from "a
/// different cell of the same mesh" (distance of order one cell width), so
/// nine decades of margin is not a tuned number -- anything from 1e-3 to
/// 1e-14 gives the same verdict on every mesh in this repository.
constexpr double kCentroidRelTol = 1e-9;

/// IS AN IDENTITY COPY THE CORRECT CONSERVATIVE MAP FROM `a` TO `b`?
///
/// This function exists because the check it replaces was `elements_a ==
/// elements_b`, and equal counts prove nothing at all: two meshes of the same
/// domain with the cells enumerated in a different order have equal counts,
/// equal measures, equal everything a count-based guard looks at, and an
/// identity copy between them delivers every cell its neighbour's value with
/// no diagnostic whatsoever. That is the exact failure this guards.
///
/// WHAT IS PROVED. Same topological dimension; same local entity count; and,
/// for EVERY local index `i`, the two regions' cell `i` has the same measure
/// and the same centroid. Since a local index is the only thing a buffer
/// offset means (`MeshView`'s indexing contract), that is precisely the
/// statement "buffer slot `i` on the source names the same piece of the domain
/// as buffer slot `i` on the target", which is what makes `dst[i] = src[i]`
/// the conservative map.
///
/// WHAT IS NOT PROVED, stated because the difference matters. Equal measure
/// and equal centroid do not imply congruent cells: a square and the same
/// square rotated about its centre satisfy both. What rules that out here is
/// that the comparison holds for every cell of two meshes that each tile the
/// domain, not that any single cell was matched. This is a proof of
/// CORRESPONDENCE strong enough to justify the copy, not a proof of geometric
/// identity, and a caller who wants the latter should let the supermesh build
/// the weights -- which is what happens whenever this returns false.
///
/// A region is trivially coincident with itself, entity set and all. Two
/// DIFFERENT regions that name an entity set are never accepted: `MeshView`
/// cannot enumerate a set's members, so there is no way to compare cell `i` of
/// one against cell `i` of the other, and `MeshBackend` does not even say
/// whether a buffer for such a region is indexed by set position or by
/// mesh-local index.
[[nodiscard]] Coincidence regions_coincide(const CouplingGraph& graph,
                                           const MeshBackend& backend,
                                           RegionId a, RegionId b) {
  Coincidence out;
  if (!a.valid() || a.value() >= graph.region_count() || !b.valid() ||
      b.value() >= graph.region_count()) {
    out.why = "one of the two regions is not declared by the graph";
    return out;
  }
  if (a == b) {
    out.same = true;
    return out;
  }
  const Region& ra = graph.region(a);
  const Region& rb = graph.region(b);

  const bool set_a = !(ra.entity_set.empty() || ra.entity_set == "all");
  const bool set_b = !(rb.entity_set.empty() || rb.entity_set == "all");
  if (set_a || set_b) {
    // Same mesh, same dimension, same set string is the one case that can be
    // settled without enumerating anything: the two regions designate the
    // identical set of entities by construction.
    if (ra.mesh == rb.mesh && ra.dimension == rb.dimension &&
        ra.entity_set == rb.entity_set) {
      out.same = true;
      return out;
    }
    out.why = "region '" + ra.name + "' or '" + rb.name +
              "' names an entity set, whose members MeshView cannot "
              "enumerate, so their local indices cannot be compared";
    return out;
  }

  auto view_of = [&](const Region& r,
                     const MeshView** into) -> std::string {
    if (!r.mesh.valid() || r.mesh.value() >= graph.mesh_count()) {
      return "region '" + r.name + "' names a mesh the graph does not declare";
    }
    // By NAME: the graph's MeshIds and the backend's are independent
    // positional id spaces (MeshBackend::mesh_id).
    const Result<MeshId> id = backend.mesh_id(graph.mesh(r.mesh).name);
    if (!id.ok()) return id.error().message;
    const Result<const MeshView*> v = backend.view(id.value());
    if (!v.ok()) return v.error().message;
    if (v.value() == nullptr) {
      return "the backend returned no view for region '" + r.name + "'";
    }
    *into = v.value();
    return {};
  };

  const MeshView* va = nullptr;
  const MeshView* vb = nullptr;
  out.why = view_of(ra, &va);
  if (!out.why.empty()) return out;
  out.why = view_of(rb, &vb);
  if (!out.why.empty()) return out;

  const int dim_a = ra.dimension >= 0 ? ra.dimension : va->dimension();
  const int dim_b = rb.dimension >= 0 ? rb.dimension : vb->dimension();
  if (dim_a != dim_b) {
    out.why = "they hold entities of different dimension (" +
              std::to_string(dim_a) + " and " + std::to_string(dim_b) + ")";
    return out;
  }
  const std::size_t n = va->entity_count(dim_a);
  if (n != vb->entity_count(dim_b)) {
    out.why = "they hold " + std::to_string(n) + " and " +
              std::to_string(vb->entity_count(dim_b)) + " entities";
    return out;
  }
  if (n == 0) {
    out.why = "both regions are empty, so there is nothing to establish a "
              "correspondence between";
    return out;
  }

  // MeshView::measure and ::centroid throw by contract; setup returns a
  // Result and must not, so a backend that has no centroid for this dimension
  // is "not provable", not a crash.
  try {
    for (std::size_t i = 0; i < n; ++i) {
      const double ma = va->measure(dim_a, i);
      const double mb = vb->measure(dim_b, i);
      const double scale = std::max(std::abs(ma), std::abs(mb));
      if (std::abs(ma - mb) > kMeasureRelTol * scale) {
        out.why = "entity " + std::to_string(i) + " measures " +
                  std::to_string(ma) + " on one side and " +
                  std::to_string(mb) + " on the other";
        return out;
      }
      const Point pa = va->centroid(dim_a, i);
      const Point pb = vb->centroid(dim_b, i);
      const double dx = pa.x - pb.x;
      const double dy = pa.y - pb.y;
      const double dz = pa.z - pb.z;
      const double d = std::sqrt(dx * dx + dy * dy + dz * dz);
      const double h = std::pow(scale, 1.0 / static_cast<double>(dim_a));
      if (d > kCentroidRelTol * h) {
        out.why = "entity " + std::to_string(i) +
                  " sits at a different place on the two sides (centroids " +
                  std::to_string(d) + " apart, cell size " +
                  std::to_string(h) +
                  "), so local index " + std::to_string(i) +
                  " does not name the same cell";
        return out;
      }
    }
  } catch (const std::exception& e) {
    out.why = std::string("the backend cannot report a measure or a centroid "
                          "for these entities: ") +
              e.what();
    return out;
  }

  out.same = true;
  return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

Result<void> Runner::Impl::resolve_buffers(FieldRegistry& registry) {
  const std::vector<BufferDecl>& buffers = plan.buffers().buffers;

  storage.assign(buffers.size(), {});
  slots.assign(buffers.size(), Slot{});
  previous.assign(buffers.size(), {});
  sample_count.assign(buffers.size(), 0);
  slot_source_owner.assign(buffers.size(), -1);

  // Map each component-owned arc back to the field it aliases. `plan_buffers`
  // marks exactly the Source and Sink outputs component-owned, and lowering
  // labels those nodes "component.field", so the two agree by construction --
  // but a mismatch is a silent wrong-pointer bug, so it is an error, not a
  // fallback to allocating a shadow buffer.
  std::unordered_map<std::uint32_t, FieldId> arc_field;
  for (std::uint32_t i = 0; i < dag.node_count(); ++i) {
    const Node& n = dag.node(NodeId{i});
    if (n.kind != OpKind::Source && n.kind != OpKind::Sink) continue;
    if (!n.output.valid()) continue;
    const auto dot = n.label.find('.');
    if (dot == std::string::npos) continue;
    const auto cid = graph.find_component(n.label.substr(0, dot));
    if (!cid.has_value()) continue;
    const auto fid = graph.find_field(*cid, n.label.substr(dot + 1));
    if (!fid.has_value()) continue;
    arc_field.emplace(n.output.value(), *fid);
    // Only a Source output is state the component wrote in advance(); a Sink
    // output is the coupler's delivery into the component's Requires array,
    // which the component is not expected to checkpoint. Spec §5.3's runtime
    // verification (iteration.cpp) compares the former and never the latter.
    if (n.kind == OpKind::Source &&
        n.output.value() < slot_source_owner.size()) {
      slot_source_owner[n.output.value()] =
          static_cast<std::int32_t>(cid->value());
    }
  }

  for (std::size_t i = 0; i < buffers.size(); ++i) {
    const BufferDecl& b = buffers[i];
    Slot& s = slots[i];
    s.components = b.components;
    s.elements = b.elements;
    s.persistent = b.is_persistent;

    if (!b.is_component_owned) {
      // Spec §4.2: a coupler intermediate. The coupler allocates it, once.
      storage[i].assign(s.total(), 0.0);
      s.data = storage[i].data();
      s.owned_by_component = false;
      continue;
    }

    // Spec §4.2: the coupler NEVER owns component data. This arc aliases the
    // array the component registered; allocating a shadow buffer here would
    // mean the component never sees the coupled values.
    const auto it = arc_field.find(b.arc.value());
    if (it == arc_field.end()) {
      return Failure{"E-RT-UNBOUND",
                   "component-owned arc %" + std::to_string(b.arc.value()) +
                       " (" + b.label + ") does not resolve to a field"};
    }
    const Result<FieldRef> ref = registry.get(it->second);
    if (!ref.ok()) {
      return Failure{"E-RT-UNBOUND",
                   "field '" + b.label +
                       "' is not bound; did the component call "
                       "register_fields?"};
    }
    FieldRef view = ref.value();  // by value: FieldRegistry hands back a copy
    if (view.space() != MemorySpace::Host) {
      return Failure{"E-RT-SPACE",
                   "field '" + b.label + "' is registered in " +
                       to_cstring(view.space()) +
                       " memory; the Phase 4 interpreter dereferences host "
                       "pointers directly"};
    }
    if (!is_flat_contiguous(view.layout())) {
      return Failure{"E-RT-LAYOUT",
                   "field '" + b.label +
                       "' is registered with a non-contiguous layout (stride=" +
                       std::to_string(view.layout().stride) +
                       ", component_stride=" +
                       std::to_string(view.layout().component_stride) +
                       ", components=" +
                       std::to_string(view.layout().components) +
                       "); the interpreter walks every buffer as a flat, "
                       "component-fastest array and would silently permute it"};
    }
    if (view.layout().count != b.elements ||
        view.layout().components != b.components) {
      return Failure{"E-RT-SIZE",
                   "field '" + b.label + "' is registered with " +
                       std::to_string(view.layout().count) + "x" +
                       std::to_string(view.layout().components) +
                       " values but the plan sizes its region at " +
                       std::to_string(b.elements) + "x" +
                       std::to_string(b.components) +
                       "; a short registration is a heap overflow in the hot "
                       "loop, where there is no bounds check left"};
    }
    s.data = view.data();
    s.owned_by_component = true;
  }
  return Result<void>::success();
}

Result<std::uint32_t> Runner::Impl::weights_for(RegionId region,
                                                std::size_t elements,
                                                int components) {
  const auto key = (static_cast<std::uint64_t>(region.value()) << 32) |
                   static_cast<std::uint32_t>(components);
  const auto cached = weights_by_region.find(key);
  if (cached != weights_by_region.end()) return cached->second;

  if (!region.valid() || region.value() >= graph.region_count()) {
    return Failure{"E-RT-MEASURE", "arc names region " +
                                     std::to_string(region.value()) +
                                     ", which the graph does not declare"};
  }
  const Region& reg = graph.region(region);

  // A named entity set is a SUBSET of the mesh's entities at that dimension,
  // and `MeshView` offers no way to enumerate which -- so buffer index i is
  // not mesh entity index i, and any weight taken from `measure(dim, i)` would
  // be the wrong element's. Refused rather than silently mis-weighted.
  if (!reg.entity_set.empty() && reg.entity_set != "all") {
    return Failure{"E-RT-MEASURE",
                 "region '" + reg.name + "' is the entity set '" +
                     reg.entity_set +
                     "', whose members MeshView cannot enumerate, so the "
                     "measures a conservation check needs cannot be taken; "
                     "drop the Conservation flag or wait for the Phase 5 "
                     "backend"};
  }

  const Result<const MeshView*> view = backend.view(reg.mesh);
  if (!view.ok()) return view.error();
  const MeshView* mv = view.value();
  if (mv == nullptr) {
    return Failure{"E-RT-MEASURE",
                 "mesh backend returned no view for region '" + reg.name + "'"};
  }
  const int dim = reg.dimension >= 0 ? reg.dimension : mv->dimension();
  if (mv->entity_count(dim) != elements) {
    return Failure{"E-RT-MEASURE",
                 "region '" + reg.name + "' holds " +
                     std::to_string(mv->entity_count(dim)) +
                     " entities of dimension " + std::to_string(dim) +
                     " but the plan sized its buffers at " +
                     std::to_string(elements)};
  }

  const auto nc = static_cast<std::size_t>(components);
  std::vector<double> w;
  w.reserve(elements * nc);
  try {
    for (std::size_t i = 0; i < elements; ++i) {
      // MeshView contract point 2: owned entities are not a prefix, so a
      // ghost is zeroed individually rather than by truncating the range.
      // Including one would double count against another rank's copy.
      const double m = mv->is_owned(dim, i) ? mv->measure(dim, i) : 0.0;
      for (std::size_t c = 0; c < nc; ++c) w.push_back(m);
    }
  } catch (const std::exception& e) {
    return Failure{"E-RT-MEASURE", "region '" + reg.name +
                                     "' has no usable measure at dimension " +
                                     std::to_string(dim) + ": " + e.what()};
  }

  const auto idx = static_cast<std::uint32_t>(weight_sets.size());
  weight_sets.push_back(std::move(w));
  weights_by_region.emplace(key, idx);
  return idx;
}

Result<void> Runner::Impl::resolve_constrain(const Instruction& ins,
                                             Resolved& r) {
  // Walk back up the SSA arcs to the Remap this Constrain verifies. Lowering
  // only emits Constrain when the channel had a Remap (src/ir/rules_derived.cpp),
  // so failing to find one means the DAG is not the one this plan describes.
  ArcId cur = ins.in[0];
  ArcId before;
  for (std::size_t guard = 0; guard <= dag.node_count(); ++guard) {
    if (!cur.valid() || cur.value() >= dag.arc_count()) break;
    const NodeId producer = dag.arc(cur).producer;
    if (!producer.valid()) break;
    const Node& n = dag.node(producer);
    if (n.kind == OpKind::Remap) {
      if (n.inputs.empty()) break;
      before = n.inputs.front();
      break;
    }
    if (n.inputs.empty()) break;
    cur = n.inputs.front();
  }
  if (!before.valid() || before.value() >= slots.size()) {
    return Failure{"E-RT-CONSTRAIN",
                 "no Remap upstream of the Constrain writing arc %" +
                     std::to_string(ins.out.value()) +
                     ", so there is no pre-remap integral to compare against"};
  }

  const Slot& in_slot = slots[before.value()];
  const Slot& out_slot = slots[ins.out.value()];
  const Result<std::uint32_t> wb = weights_for(
      dag.arc(before).type.region, in_slot.elements, in_slot.components);
  if (!wb.ok()) return wb.error();
  const Result<std::uint32_t> wa = weights_for(
      dag.arc(ins.out).type.region, out_slot.elements, out_slot.components);
  if (!wa.ok()) return wa.error();

  r.constrain = static_cast<std::int32_t>(constrains.size());
  constrains.push_back(
      ConstrainInfo{before.value(), wb.value(), wa.value()});
  return Result<void>::success();
}

// ---------------------------------------------------------------------------
// Spec §5.4: the spatial remap (Task 5.5)
// ---------------------------------------------------------------------------

Result<void> Runner::Impl::resolve_remap(const Instruction& ins, Resolved& r) {
  if (ins.n_in == 0 || !ins.out.valid()) return Result<void>::success();

  const Slot& in = slots[ins.in[0].value()];
  const Slot& out = slots[ins.out.value()];
  const RegionId src = dag.arc(ins.in[0]).type.region;
  const RegionId tgt = dag.arc(ins.out).type.region;

  std::string label = "a Remap writing arc %" + std::to_string(ins.out.value());
  if (ins.origin_node.valid() && ins.origin_node.value() < dag.node_count() &&
      !dag.node(ins.origin_node).label.empty()) {
    label = "'" + dag.node(ins.origin_node).label + "'";
  }

  // The weights act per entity and are replayed for each component, so the two
  // ends must agree on how many components an entity carries even when they
  // disagree on how many entities there are.
  if (in.components != out.components) {
    return Failure{"E-RT-REMAP-SIZE",
                 label + " reads " + std::to_string(in.components) +
                     " components per entity and writes " +
                     std::to_string(out.components) +
                     "; a spatial remap moves values between meshes, it does "
                     "not change a field's rank"};
  }

  // ---- 1. The identity, but only when it is provably the right answer -----
  const Coincidence c = regions_coincide(graph, backend, src, tgt);
  if (c.same) {
    if (in.total() != out.total()) {
      return Failure{"E-RT-REMAP-SIZE",
                   label + " connects two regions that hold the same entities "
                   "in the same order, yet the plan sized their buffers at " +
                       std::to_string(in.total()) + " and " +
                       std::to_string(out.total()) +
                       " values; the buffer plan and the mesh backend "
                       "disagree about the same region"};
    }
    r.remap = -1;
    return Result<void>::success();
  }

  // ---- 2. Otherwise, real conservative weights -----------------------------
  RemapPlan p;
  p.src = src;
  p.tgt = tgt;
  p.flags = ins.attrs.remap_flags;
  // THE METHOD STRING IS A BUILDER SELECTOR, AND IT IS THE USER'S. It is
  // `Channel::remap.method_override`, carried verbatim through lowering, and
  // it is forwarded here unaltered rather than being defaulted, normalised or
  // upgraded. Empty means "whatever `build_weights` documents as its default";
  // anything else is an explicit request for a named builder, and two builders
  // asked for the same order do not produce the same matrix. The runtime is
  // not the place to choose between them: it has no information the user does
  // not, and silently substituting one would make a channel's numerics depend
  // on which library version it ran against.
  p.method = ins.attrs.method;
  // Discretization and order live on the FIELDS, not on the operator, so they
  // are read from the channel this node was lowered from. A pair the builder
  // cannot handle (spectral element, point cloud, high order) is refused there
  // with E-REMAP-UNSUPPORTED rather than silently downgraded here.
  const NodeId origin = ins.origin_node;
  if (origin.valid() && origin.value() < dag.node_count()) {
    const ChannelId ch = dag.node(origin).origin;
    if (ch.valid() && ch.value() < graph.channel_count()) {
      const Channel& channel = graph.channel_at(ch);
      const FieldSpec& sf = graph.field(channel.source);
      const FieldSpec& tf = graph.field(channel.target);
      p.src_disc = sf.disc;
      p.tgt_disc = tf.disc;
      p.src_order = sf.order;
      p.tgt_order = tf.order;
    }
  }

#if FUSE_HAVE_MOAB
  // ==========================================================================
  // A REMAP HALO IS NOT WIRED THROUGH THE RUNTIME YET, AND THAT IS A REFUSAL
  // RATHER THAN A DETAIL.
  //
  // `build_remap_halo` makes `build_weights` correct on a partitioned read by
  // adopting the off-rank source cells as non-owned local entities. The matrix
  // that comes back therefore has columns for them -- and those columns
  // multiply whatever is in the corresponding slots of the input buffer.
  // Filling them is `MoabMeshBackend::exchange_remap_halo`, which has to run
  // between the operator's input being ready and the matvec.
  //
  // The runtime does not call it. Two things have to be settled first, and
  // neither is a line of code:
  //
  //   * the exchange is COLLECTIVE, so it has to sit somewhere every rank
  //     reaches in the same order -- which the instruction stream does give,
  //     but nothing in the runtime currently asserts;
  //   * a Remap whose input arc is COMPONENT OWNED would have FUSE writing
  //     into a component's array, and whether the halo slots of a registered
  //     field are the coupler's to fill is a contract question, not an
  //     implementation one.
  //
  // Proceeding without the exchange would produce a plausible matrix applied
  // to uninitialised halo values: silently wrong answers, which is the exact
  // failure this whole area exists to prevent. So the runtime refuses, and
  // says which call is missing. The library-level path
  // (build_remap_halo -> build_weights -> exchange_remap_halo ->
  // apply_weights) is complete and is conservation-tested in parallel by
  // test/unit/remap_halo_parallel_test.cpp.
  // ==========================================================================
  if (const auto* mb = dynamic_cast<const MoabMeshBackend*>(&backend)) {
    const Region& sr = graph.region(src);
    if (sr.mesh.valid() && sr.mesh.value() < graph.mesh_count()) {
      const Result<MeshId> smid = mb->mesh_id(graph.mesh(sr.mesh).name);
      if (smid.ok()) {
        const Result<const MeshView*> sview = mb->view(smid.value());
        const int sdim = sr.dimension >= 0
                             ? sr.dimension
                             : (sview.ok() ? sview.value()->dimension() : -1);
        if (sdim >= 0 && mb->remap_halo_size(smid.value(), sdim) > 0) {
          return Failure{
              "E-RT-REMAP-HALO",
              label + " reads region '" + graph.region(src).name +
                  "', whose mesh carries a remap halo of " +
                  std::to_string(mb->remap_halo_size(smid.value(), sdim)) +
                  " cells. The weights would have columns for those cells, "
                  "and the runtime does not yet run "
                  "MoabMeshBackend::exchange_remap_halo() to fill them, so "
                  "the matvec would multiply uninitialised values and return "
                  "a plausible wrong answer. Drive the remap directly "
                  "(build_remap_halo, build_weights, exchange_remap_halo, "
                  "apply_weights), or run the coupled case without a "
                  "communicator on the mesh backend."};
        }
      }
    }
  }
#endif

  const std::size_t before_builds = remap_cache.misses();
  const Result<const RemapWeights*> w =
      remap_cache.get_or_build(p, graph, backend);
  if (!w.ok()) {
    // The builder's own code is preserved: E-REMAP-COVERAGE, E-REMAP-GEOMETRY
    // and friends each tell the caller something different about what to do
    // next, and folding them all into one runtime code would throw that away.
    // What is added is the fact that the cheap alternative was tried first and
    // why it was rejected -- without that the message reads as though a remap
    // were optional here, and it is not.
    return Failure{w.error().code,
                 label + " must transfer between region '" +
                     graph.region(src).name + "' and region '" +
                     graph.region(tgt).name +
                     "', which are not provably the same cells in the same "
                     "order (" +
                     c.why +
                     "), so an element-wise copy would deliver wrong values "
                     "with no diagnostic. Building conservative weights "
                     "instead failed: " +
                     w.error().message};
  }
  const RemapWeights* weights = w.value();
  if (remap_cache.misses() != before_builds) {
    ++stats.remap_weight_builds;
    stats.remap_weight_nonzeros += static_cast<std::int64_t>(weights->nnz());
  }

  // The matrix is indexed by MeshView local entity indices; the buffers are
  // indexed the same way, but nothing in either type says so, and a mismatch
  // is an out-of-bounds write that apply_weights would only catch per triplet.
  if (weights->n_cols != in.elements || weights->n_rows != out.elements) {
    return Failure{"E-RT-REMAP-SIZE",
                 label + " was given a " + std::to_string(weights->n_rows) +
                     "x" + std::to_string(weights->n_cols) +
                     " matrix, but the plan sizes its input at " +
                     std::to_string(in.elements) +
                     " entities and its output at " +
                     std::to_string(out.elements) +
                     "; the buffer plan and the remap weights disagree about "
                     "the two regions"};
  }

  r.remap = static_cast<std::int32_t>(remap_weights.size());
  remap_weights.push_back(weights);
  return Result<void>::success();
}

// ---------------------------------------------------------------------------
// Spec §7.3: the space-time remap (Task 6.3)
// ---------------------------------------------------------------------------

namespace {

/// The length of a channel's space-time window, or the reason it has none.
///
/// Only a window equal to the plan's common period is supported, and the other
/// two spellings are accepted only when they *are* that period. A shorter
/// window is a genuinely different operator -- `T` would have to be rebuilt per
/// sub-window and the source history re-based -- and quietly widening it to the
/// period would give the user a projection over data they did not ask to
/// include.
[[nodiscard]] Result<Rational> spacetime_window(const CouplingGraph& graph,
                                                const Window& w,
                                                const std::string& channel) {
  switch (w.kind) {
    case Window::Kind::CouplingPeriod:
      return Rational{0};  // "the period", resolved by the caller
    case Window::Kind::Explicit:
      return w.length;
    case Window::Kind::FromComponent:
      if (!w.component.valid() ||
          w.component.value() >= graph.component_count()) {
        return Failure{"E-RT-SPACETIME",
                     "channel '" + channel +
                         "' takes its space-time window from a component the "
                         "graph does not declare"};
      }
      return graph.component(w.component).interval;
  }
  return Failure{"E-RT-SPACETIME",
               "channel '" + channel + "' has an unknown window kind"};
}

}  // namespace

Result<void> Runner::Impl::resolve_spacetime() {
  spacetimes.clear();
  spacetime_by_node.clear();

  // Count first, then reserve: `Resolved::spacetime` is an index into this
  // vector and the sweep loop holds no pointers into it, but growing it while
  // building would still reallocate every TemporalProjection it already holds.
  std::size_t n_st = 0;
  for (std::uint32_t i = 0; i < dag.node_count(); ++i) {
    const Node& n = dag.node(NodeId{i});
    if (n.kind == OpKind::Interpolate &&
        n.attrs.basis != TemporalBasis::None) {
      ++n_st;
    }
  }
  if (n_st == 0) return Result<void>::success();
  spacetimes.reserve(n_st);

  const Rational period = plan.period();

  for (std::uint32_t i = 0; i < dag.node_count(); ++i) {
    const NodeId nid{i};
    const Node& n = dag.node(nid);
    if (n.kind != OpKind::Interpolate) continue;
    if (n.attrs.basis == TemporalBasis::None) continue;

    if (!n.origin.valid() || n.origin.value() >= graph.channel_count()) {
      return Failure{"E-RT-SPACETIME",
                   "node \"" + n.label +
                       "\" carries a temporal basis but belongs to no channel, "
                       "so the two clocks its operator is built from are "
                       "unknown"};
    }
    const Channel& ch = graph.channel_at(ChannelId{n.origin.value()});

    // Accumulation and a space-time projection are two competing temporal
    // treatments of the same channel: `Mean` collapses the source's time
    // dimension to one value, and projecting one value onto the target's time
    // partition is not the operator either spelling asks for. Refused rather
    // than silently letting one win -- both were written down, so the
    // ambiguity is the user's to resolve.
    //
    // `expand()` already refuses this with `E-SCHED-SPACETIME` and a fuller
    // message, so a plan from `schedule()` never reaches here. This is the
    // second line for a `Runner` handed a plan built some other way.
    if (ch.accumulate.op != AccumOp::Instant) {
      return Failure{"E-RT-SPACETIME",
                   "channel '" + ch.name + "' asks for both accumulate(" +
                       to_cstring(ch.accumulate.op) +
                       ") and a space-time basis. Those are two different "
                       "temporal operators on the same data; drop one. A "
                       "degree-0 space-time projection onto a single target "
                       "step already IS the window mean"};
    }

    const Result<Rational> win =
        spacetime_window(graph, ch.spacetime.window, ch.name);
    if (!win.ok()) return win.error();
    const Rational window =
        ch.spacetime.window.kind == Window::Kind::CouplingPeriod ? period
                                                                 : win.value();
    if (window != period) {
      return Failure{"E-RT-SPACETIME",
                   "channel '" + ch.name + "' asks for a space-time window of " +
                       window.to_string() +
                       ", but only a window equal to the plan's common period " +
                       period.to_string() +
                       " is implemented; a shorter window is a different "
                       "operator, not a truncation of this one"};
    }

    const ComponentId src_c = graph.owner_of(ch.source);
    const ComponentId dst_c = graph.owner_of(ch.target);
    if (!src_c.valid() || src_c.value() >= graph.component_count() ||
        !dst_c.valid() || dst_c.value() >= graph.component_count()) {
      return Failure{"E-RT-SPACETIME",
                   "channel '" + ch.name + "' has an endpoint no component "
                   "owns, so its source and target intervals are unknown"};
    }
    const Rational src_dt = graph.component(src_c).interval;
    const Rational dst_dt = graph.component(dst_c).interval;

    Result<SlabDecomposition> slabs = decompose(src_dt, dst_dt, period);
    if (!slabs.ok()) return slabs.error();
    // Every rejection below (E-ST-BASIS for None/Spline, E-ST-DEGREE for a
    // degree the basis does not admit or one above kMaxTemporalDegree,
    // E-ST-INTERVAL, E-ST-SLABS) is forwarded verbatim. There is deliberately
    // no fallback to a lower degree: a user who asked for DG(2) and silently
    // got piecewise-constant gets plausible numbers and a wrong answer.
    Result<TemporalProjection> proj = build_temporal_projection(
        slabs.value(), n.attrs.basis, n.attrs.degree, src_dt, dst_dt);
    if (!proj.ok()) return proj.error();

    if (!n.output.valid() || n.output.value() >= slots.size()) {
      return Failure{"E-RT-SPACETIME",
                   "the space-time node of channel '" + ch.name +
                       "' has no output buffer"};
    }
    const Slot& out = slots[n.output.value()];

    SpacetimeInfo info;
    info.temporal = std::move(proj.value());
    info.elements = out.elements;
    info.components = out.components;
    info.width = out.total();

    const std::size_t n_tgt = info.temporal.n_tgt_steps;
    const std::size_t n_src = info.temporal.n_src_steps;
    info.row_begin.assign(n_tgt + 1, info.temporal.nnz());
    {
      // `temporal.row` is ascending, so one pass fills the offsets. Rows with
      // no entry keep the offset of the next populated row, which makes the
      // half-open range empty rather than wrong.
      std::size_t k = 0;
      for (std::size_t t = 0; t <= n_tgt; ++t) {
        while (k < info.temporal.nnz() &&
               static_cast<std::size_t>(info.temporal.row[k]) < t) {
          ++k;
        }
        info.row_begin[t] = k;
      }
      info.row_begin[n_tgt] = info.temporal.nnz();
    }

    constexpr std::size_t kMax = static_cast<std::size_t>(-1);
    if (info.width != 0 && n_src > kMax / info.width) {
      return Failure{"E-RT-SPACETIME",
                   "channel '" + ch.name + "' would need a history of " +
                       std::to_string(n_src) + " source steps x " +
                       std::to_string(info.width) +
                       " values, which overflows std::size_t"};
    }
    info.history.assign(n_src * info.width, 0.0);

    info.identity_row.resize(info.elements);
    info.identity_col.resize(info.elements);
    info.identity_val.assign(info.elements, 1.0);
    for (std::size_t e = 0; e < info.elements; ++e) {
      info.identity_row[e] = static_cast<std::int32_t>(e);
      info.identity_col[e] = static_cast<std::int32_t>(e);
    }
    if (info.elements >
        static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
      return Failure{"E-RT-SPACETIME",
                   "channel '" + ch.name + "' spans " +
                       std::to_string(info.elements) +
                       " entities, more than a COO index can hold"};
    }

    spacetime_by_node.emplace(i, static_cast<std::uint32_t>(spacetimes.size()));
    spacetimes.push_back(std::move(info));
  }
  return Result<void>::success();
}

Result<void> Runner::Impl::check_spacetime_schedule() {
  if (spacetimes.empty()) return Result<void>::success();

  const std::vector<Instruction>& stream = plan.instructions();

  // Per space-time channel: how many times each source step is sampled and
  // each target step projected. A step covered zero times is a value the
  // operator reads but nothing writes; twice is a double count.
  std::vector<std::vector<std::int32_t>> sampled(spacetimes.size());
  std::vector<std::vector<std::int32_t>> projected(spacetimes.size());
  for (std::size_t s = 0; s < spacetimes.size(); ++s) {
    sampled[s].assign(spacetimes[s].temporal.n_src_steps, 0);
    projected[s].assign(spacetimes[s].temporal.n_tgt_steps, 0);
  }

  for (std::size_t i = 0; i < stream.size(); ++i) {
    const Instruction& ins = stream[i];
    const std::int32_t idx = resolved[i].spacetime;
    if (idx < 0) continue;
    const auto s = static_cast<std::size_t>(idx);
    const SpacetimeInfo& st = spacetimes[s];

    if (ins.kind == OpKind::Sample) {
      if (ins.src_step < 0 ||
          static_cast<std::size_t>(ins.src_step) >= st.temporal.n_src_steps) {
        return Failure{"E-RT-SPACETIME",
                     "instruction " + std::to_string(i) +
                         " samples source step " +
                         std::to_string(ins.src_step) + ", outside [0, " +
                         std::to_string(st.temporal.n_src_steps) + ")"};
      }
      ++sampled[s][static_cast<std::size_t>(ins.src_step)];
      continue;
    }
    if (ins.kind != OpKind::Interpolate) continue;

    if (ins.dst_step < 0 ||
        static_cast<std::size_t>(ins.dst_step) >= st.temporal.n_tgt_steps) {
      return Failure{"E-RT-SPACETIME",
                   "instruction " + std::to_string(i) +
                       " projects onto target step " +
                       std::to_string(ins.dst_step) + ", outside [0, " +
                       std::to_string(st.temporal.n_tgt_steps) + ")"};
    }
    ++projected[s][static_cast<std::size_t>(ins.dst_step)];

    // CAUSALITY. `expand()` places this instance at the end of target step
    // `dst_step`, by which time source steps 0..`src_step` have been sampled
    // and no more. Row `dst_step` of `T` may name a later column: the
    // reconstruction stencil of degree p leans forward by up to p/2 steps (see
    // `stencil_start` in src/remap/spacetime.cpp), so DG(2) and above, and any
    // positive degree in the refining direction, need source data the source
    // component has not produced yet.
    //
    // That is refused, not served from the previous period's history. Serving
    // it would be a one-window-lagged operator wearing this one's name -- the
    // numbers would look plausible and the order would be wrong.
    const std::size_t lo = st.row_begin[static_cast<std::size_t>(ins.dst_step)];
    const std::size_t hi =
        st.row_begin[static_cast<std::size_t>(ins.dst_step) + 1];
    for (std::size_t k = lo; k < hi; ++k) {
      if (static_cast<std::int64_t>(st.temporal.col[k]) <= ins.src_step) {
        continue;
      }
      return Failure{
          "E-RT-SPACETIME",
          "the " + std::string{to_cstring(st.temporal.basis)} + "(" +
              std::to_string(st.temporal.effective_degree) +
              ") temporal projection needs source step " +
              std::to_string(st.temporal.col[k]) +
              " to write target step " + std::to_string(ins.dst_step) +
              ", but only source steps up to " + std::to_string(ins.src_step) +
              " have been produced when that target step ends. The "
              "reconstruction stencil reaches forward in time, so this "
              "combination of basis, degree and interval ratio is not causal "
              "within one coupling period. Use a lower degree, or couple in "
              "the coarsening direction (source interval shorter than target)"};
    }
  }

  for (std::size_t s = 0; s < spacetimes.size(); ++s) {
    for (std::size_t j = 0; j < sampled[s].size(); ++j) {
      if (sampled[s][j] == 1) continue;
      return Failure{"E-RT-SPACETIME",
                   "source step " + std::to_string(j) +
                       " of a space-time channel is sampled " +
                       std::to_string(sampled[s][j]) +
                       " times in a period; the temporal operator needs "
                       "exactly one value per source step"};
    }
    for (std::size_t t = 0; t < projected[s].size(); ++t) {
      if (projected[s][t] == 1) continue;
      return Failure{"E-RT-SPACETIME",
                   "target step " + std::to_string(t) +
                       " of a space-time channel is projected " +
                       std::to_string(projected[s][t]) +
                       " times in a period; expected exactly once"};
    }
  }
  return Result<void>::success();
}

Result<void> Runner::Impl::resolve_instructions() {
  const std::vector<Instruction>& stream = plan.instructions();
  resolved.assign(stream.size(), Resolved{});
  constrains.clear();
  weight_sets.clear();
  weights_by_region.clear();
  remap_weights.clear();
  // Both are APPENDED to by the OpKind::Transport case below, so they must be
  // reset here like every other setup-owned container. Runner::initialize() is
  // re-callable (it clears `initialized` at entry), and without these a second
  // initialize() would carry the first call's entries forward and trip
  // attach_comm_graphs()'s "one CommGraph-backed Transport per run" guard
  // (E-RT-TRANSPORT) on a plan that has exactly one.
  transport_plans.clear();
  transport_instruction.clear();
  stats.remap_weight_builds = 0;
  stats.remap_weight_nonzeros = 0;

  // Before the loop: the loop below tags each half of a space-time channel
  // with the index of the operator it drives, so the operator has to exist
  // first.
  const Result<void> st = resolve_spacetime();
  if (!st.ok()) return st;

  // Spec §5.3, Phase 7 Task 7.5. The loop itself lives in
  // src/runtime/iteration.cpp; resolve_regions() builds it. What is still
  // refused here is a marker with no region behind it: the interpreter would
  // walk straight over an enter/exit pair and run the body exactly once, which
  // turns a `max_iterations = N` implicit scheme into a single explicit sweep
  // with no diagnostic.
  if (plan.region_count() == 0) {
    for (std::size_t i = 0; i < stream.size(); ++i) {
      if (stream[i].region_enter >= 0 || stream[i].region_exit >= 0) {
        return Failure{"E-RT-REGION",
                     "instruction " + std::to_string(i) +
                         " carries an iteration-region boundary, but the plan "
                         "declares no iteration region, so nothing would "
                         "execute the loop it delimits"};
      }
    }
  }

  const std::size_t n_slots = slots.size();
  for (std::size_t i = 0; i < stream.size(); ++i) {
    const Instruction& ins = stream[i];
    Resolved& r = resolved[i];

    // Bounds, once. `schedule()` already checks these against the buffer
    // table, but the hot loop indexes `slots` unchecked, so a plan that did
    // not come from `schedule()` must not reach it.
    if (ins.out.valid() && ins.out.value() >= n_slots) {
      return Failure{"E-RT-ARC", "instruction " + std::to_string(i) + " (" +
                                   to_cstring(ins.kind) + ") writes arc %" +
                                   std::to_string(ins.out.value()) +
                                   ", which has no buffer"};
    }
    for (std::uint8_t k = 0; k < ins.n_in; ++k) {
      if (!ins.in[k].valid() || ins.in[k].value() >= n_slots) {
        return Failure{"E-RT-ARC", "instruction " + std::to_string(i) + " (" +
                                     to_cstring(ins.kind) + ") reads arc %" +
                                     std::to_string(ins.in[k].value()) +
                                     ", which has no buffer"};
      }
      if (slots[ins.in[k].value()].data == nullptr &&
          slots[ins.in[k].value()].total() != 0) {
        return Failure{"E-RT-UNBOUND",
                     "instruction " + std::to_string(i) + " (" +
                         to_cstring(ins.kind) + ") reads arc %" +
                         std::to_string(ins.in[k].value()) +
                         ", whose buffer was never resolved"};
      }
    }
    if (!ins.out.valid()) continue;

    const Slot& out = slots[ins.out.value()];
    r.out_total = out.total();

    // Every kernel below reads `out_total` doubles from each input. An input
    // shorter than that is a heap overread with no bounds check left, so the
    // sizes are reconciled here rather than clamped there.
    //
    // `Remap` is exempt, and is the ONLY operator that is: transferring
    // between two differently-sized meshes is exactly what it is for. Its
    // sizes are reconciled against the matrix it is actually given, in
    // resolve_remap() below, which is a stronger check than this one -- equal
    // lengths never implied the two ends corresponded.
    if (ins.kind != OpKind::Remap) {
      for (std::uint8_t k = 0; k < ins.n_in; ++k) {
        const std::size_t in_total = slots[ins.in[k].value()].total();
        if (in_total == r.out_total) continue;
        return Failure{"E-RT-SIZE",
                     std::string{to_cstring(ins.kind)} + " instruction " +
                         std::to_string(i) + " reads " +
                         std::to_string(in_total) + " values from arc %" +
                         std::to_string(ins.in[k].value()) + " into " +
                         std::to_string(r.out_total) + " at arc %" +
                         std::to_string(ins.out.value())};
      }
    }

    r.accum = ins.attrs.accum;

    switch (ins.kind) {
      case OpKind::Interpolate: {
        const auto it = spacetime_by_node.find(ins.origin_node.value());
        if (it == spacetime_by_node.end()) break;  // the lag no-op, not §7.3
        r.spacetime = static_cast<std::int32_t>(it->second);
        const SpacetimeInfo& info = spacetimes[it->second];
        if (info.width != r.out_total) {
          return Failure{"E-RT-SPACETIME",
                       "the space-time operator was sized for " +
                           std::to_string(info.width) +
                           " values but instruction " + std::to_string(i) +
                           " writes " + std::to_string(r.out_total)};
        }
        break;
      }
      case OpKind::Sample: {
        // A space-time channel's Sample performs no reduction: it files the
        // source step's value in the temporal history the Interpolate below
        // projects. Checked before the Accumulate search, because such a
        // Sample deliberately has no Accumulate to find.
        {
          bool spacetime_sink = false;
          for (const NodeId c : dag.consumers_of(ins.out)) {
            const Node& cn = dag.node(c);
            if (cn.kind != OpKind::Interpolate) continue;
            const auto it = spacetime_by_node.find(c.value());
            if (it == spacetime_by_node.end()) continue;
            if (spacetime_sink && r.spacetime !=
                                      static_cast<std::int32_t>(it->second)) {
              return Failure{"E-RT-SPACETIME",
                           "arc %" + std::to_string(ins.out.value()) +
                               " feeds two different space-time projections; "
                               "one history cannot serve both"};
            }
            r.spacetime = static_cast<std::int32_t>(it->second);
            spacetime_sink = true;
          }
          if (spacetime_sink) {
            const SpacetimeInfo& info =
                spacetimes[static_cast<std::size_t>(r.spacetime)];
            if (info.width != r.out_total) {
              return Failure{"E-RT-SPACETIME",
                           "the space-time history was sized for " +
                               std::to_string(info.width) +
                               " values but instruction " + std::to_string(i) +
                               " samples " + std::to_string(r.out_total)};
            }
            break;
          }
        }
        // Lowering emits `Sample` with default OpAttrs -- only the downstream
        // `Accumulate` carries the channel's AccumOp (see
        // src/ir/rules_channel.cpp). Reading `ins.attrs.accum` here therefore
        // yields AccumOp::Instant, which makes the running reduction overwrite
        // instead of accumulate: a Mean over four samples then comes out as
        // last_sample / 4. The op is taken from the consuming Accumulate
        // instead, resolved once here so the loop still does no lookup.
        bool found = false;
        for (const NodeId c : dag.consumers_of(ins.out)) {
          const Node& cn = dag.node(c);
          if (cn.kind != OpKind::Accumulate) continue;
          if (found && cn.attrs.accum != r.accum) {
            return Failure{"E-RT-SAMPLE",
                         "arc %" + std::to_string(ins.out.value()) +
                             " feeds two Accumulate operators with different "
                             "reductions; the Sample cannot serve both"};
          }
          r.accum = cn.attrs.accum;
          found = true;
        }
        if (!found) {
          return Failure{"E-RT-SAMPLE",
                       "Sample writes arc %" + std::to_string(ins.out.value()) +
                           " but no Accumulate consumes it, so the reduction "
                           "it should perform is unknown"};
        }
        break;
      }
      case OpKind::Transport: {
        detail::TransportPlan p;
        p.spec.src_rank_set = ins.attrs.src_rank_set;
        p.spec.dst_rank_set = ins.attrs.dst_rank_set;
        p.spec.elements     = out.elements;
        p.spec.components   = out.components;
        // Transport preserves region; the input and output arcs agree. Task 3
        // checks that rather than trusting it.
        p.region = dag.arc(ins.out).type.region;
        r.transport = static_cast<std::int32_t>(transport_plans.size());
        transport_plans.push_back(std::move(p));
        transport_instruction.push_back(i);
        break;
      }
      case OpKind::Relax:
      case OpKind::Residual: {
        // The previous-iterate buffer is sized here, never in the loop.
        std::vector<double>& prev = previous[ins.out.value()];
        if (prev.size() != r.out_total) prev.assign(r.out_total, 0.0);
        break;
      }
      case OpKind::Remap: {
        const Result<void> m = resolve_remap(ins, r);
        if (!m.ok()) return m;
        break;
      }
      case OpKind::Constrain: {
        if (!config.check_conservation || ins.n_in == 0) break;
        const Result<void> c = resolve_constrain(ins, r);
        if (!c.ok()) return c;
        break;
      }
      default:
        break;
    }
  }
  return check_spacetime_schedule();
}

Result<void> Runner::Impl::build_ticks(ComponentSet& components) {
  ticks.clear();

  const Rational period = plan.period();
  if (!(period > Rational{0})) {
    return Failure{"E-RT-INTERVAL",
                 "the plan's common period is " + period.to_string() +
                     ", which is not positive"};
  }

  // Every graph component needs an implementation and vice versa. A component
  // present in only one of the two is never advanced and never exchanged,
  // which is a wrong-physics failure with no symptom.
  std::vector<bool> matched(components.size(), false);
  for (std::uint32_t i = 0; i < graph.component_count(); ++i) {
    const ComponentDecl& decl = graph.component(ComponentId{i});
    std::size_t slot = components.size();
    for (std::size_t j = 0; j < components.size(); ++j) {
      if (components.at(j).name() == decl.name) {
        slot = j;
        break;
      }
    }
    if (slot == components.size()) {
      return Failure{"E-RT-COMPONENT",
                   "the graph declares component '" + decl.name +
                       "' but no Component with that name was supplied, so it "
                       "would never be advanced"};
    }
    matched[slot] = true;

    const Rational dt = decl.interval;
    if (!(dt > Rational{0})) {
      return Failure{"E-RT-INTERVAL", "component '" + decl.name +
                                        "' has a non-positive interval " +
                                        dt.to_string()};
    }
    const Rational n_exact = period / dt;
    if (!n_exact.is_integer() || n_exact.num() <= 0) {
      return Failure{"E-RT-INTERVAL",
                   "component '" + decl.name + "' has interval " +
                       dt.to_string() +
                       ", which does not divide the plan's common period " +
                       period.to_string() + " an integer number of times"};
    }
    for (std::int64_t k = 0; k < n_exact.num(); ++k) {
      ticks.push_back(Tick{Rational{k} * dt, dt, slot, i, k, n_exact.num()});
    }
  }
  for (std::size_t j = 0; j < components.size(); ++j) {
    if (!matched[j]) {
      return Failure{"E-RT-COMPONENT",
                   "component '" + components.at(j).name() +
                       "' is not declared in the coupling graph, so nothing "
                       "schedules it"};
    }
  }

  std::stable_sort(ticks.begin(), ticks.end(),
                   [](const Tick& a, const Tick& b) {
                     if (a.time != b.time) return a.time < b.time;
                     return a.order < b.order;
                   });
  return Result<void>::success();
}

Result<void> Runner::Impl::attach_comm_graphs() {
#if !FUSE_HAVE_MPI
  if (config.base_communicator.has_value()) {
    return Failure{"E-RT-TRANSPORT-NOMPI",
                 "RunnerConfig::base_communicator was set, but this build of "
                 "FUSE was built without MPI; configure with -DFUSE_ENABLE_MPI=ON"};
  }
  return Result<void>::success();
#else
  if (!config.base_communicator.has_value()) return Result<void>::success();

  const MPI_Comm base =
      MPI_Comm_f2c(static_cast<MPI_Fint>(*config.base_communicator));
  if (base == MPI_COMM_NULL) {
    return Failure{"E-RT-TRANSPORT-COMM",
                 "RunnerConfig::base_communicator is MPI_COMM_NULL"};
  }
  int my_rank = 0;
  if (MPI_Comm_rank(base, &my_rank) != MPI_SUCCESS) {
    return Failure{"E-RT-TRANSPORT-COMM",
                 "MPI_Comm_rank failed on the base communicator"};
  }

  // One communicator per component, coloured by the graph's own RankSet.
  //
  // EVERY RANK CALLS EVERY SPLIT, in component-id order: MPI_Comm_split is
  // collective over `base`, and a rank in no component still takes part with
  // MPI_UNDEFINED and gets MPI_COMM_NULL back. A failure therefore must NOT
  // return from inside this loop -- the other ranks are already inside the
  // next split. Record it and finish the loop.
  const std::uint32_t   n_comp = graph.component_count();
  std::vector<MPI_Comm> comms(n_comp, MPI_COMM_NULL);
  Result<void>          status = Result<void>::success();
  for (std::uint32_t i = 0; i < n_comp; ++i) {
    const ComponentId cid{i};
    const RankSet&    rs = graph.component(cid).ranks;
    const int colour = rs.contains(my_rank) ? 0 : MPI_UNDEFINED;
    if (MPI_Comm_split(base, colour, my_rank, &comms[i]) != MPI_SUCCESS &&
        status.ok()) {
      status = Failure{"E-RT-TRANSPORT-COMM",
                     "MPI_Comm_split failed for component '" +
                         graph.component(cid).name + "'"};
    }
  }

  if (status.ok() && transport_plans.size() > 1) {
    status = Failure{"E-RT-TRANSPORT",
                   "this build wires one CommGraph-backed Transport per run, "
                   "but the plan has " +
                       std::to_string(transport_plans.size()) +
                       ". One Transport instance per instruction is the next "
                       "step."};
  }

  for (std::size_t pi = 0; pi < transport_plans.size() && status.ok(); ++pi) {
    detail::TransportPlan& p   = transport_plans[pi];
    const Instruction&     ins = plan.instructions()[transport_instruction[pi]];

    // (1) Transport preserves region -- rules_channel.cpp changes only
    // `rank_set` -- and the matching below depends on it: both sides are the
    // same mesh, so global ids agree by construction. If the lowering ever
    // moves data between regions, matching them needs a NearestNeighbor
    // tolerance the graph cannot express. Refuse loudly rather than match the
    // wrong entities silently.
    if (!(dag.arc(ins.in[0]).type.region == dag.arc(ins.out).type.region)) {
      status = Failure{"E-RT-TRANSPORT-XREGION",
                     "Transport crosses regions, which the runtime cannot "
                     "match: both ends must be the same region on the same "
                     "mesh, and global-id matching is the only mode available"};
      break;
    }

    // (2) Sub-set regions, refused exactly as remap refuses them.
    const Region& reg = graph.region(p.region);
    if (!(reg.entity_set.empty() || reg.entity_set == "all")) {
      status = Failure{"E-RT-TRANSPORT-REGION",
                     "region '" + reg.name + "' names entity set '" +
                         reg.entity_set +
                         "', and MeshBackend offers no way to say whether a "
                         "buffer for it is indexed by set position or by "
                         "mesh-local index. Same restriction, and the same "
                         "reason, as E-REMAP-REGION."};
      break;
    }

    // From here down the loop is COLLECTIVE over `base`. `break` above this
    // point is safe because those checks are local and identical on every
    // rank; a `break` below it would leave peers blocked inside
    // CommGraph::build. CommGraph agrees every one of its own errors with an
    // MPI_Allreduce before returning (comm_graph.hpp:196-198), so a failure
    // here is already the same failure everywhere -- which is what makes the
    // `status.ok()` loop condition safe to break on afterwards.
    const Region& reg_side = graph.region(p.region);
    const auto    mid      = backend.mesh_id(graph.mesh(reg_side.mesh).name);
    if (!mid.ok()) { status = mid.error(); break; }
    const auto view = backend.view(mid.value());
    if (!view.ok()) { status = view.error(); break; }
    const int dim =
        reg_side.dimension >= 0 ? reg_side.dimension : view.value()->dimension();

    const auto src_cid = static_cast<std::uint32_t>(p.spec.src_rank_set);
    const auto dst_cid = static_cast<std::uint32_t>(p.spec.dst_rank_set);
    const bool in_src =
        graph.component(ComponentId{src_cid}).ranks.contains(my_rank);
    const bool in_dst =
        graph.component(ComponentId{dst_cid}).ranks.contains(my_rank);

    // A side's mesh must be non-null exactly when its comm_handle names a real
    // communicator (comm_graph.hpp:46-47).
    CommGraphSide src_side;
    src_side.dim         = dim;
    src_side.comm_handle = MPI_Comm_c2f(comms[src_cid]);
    src_side.mesh        = in_src ? view.value() : nullptr;

    CommGraphSide dst_side;
    dst_side.dim         = dim;
    dst_side.comm_handle = MPI_Comm_c2f(comms[dst_cid]);
    dst_side.mesh        = in_dst ? view.value() : nullptr;

    CommGraphOptions opts;
    // NOT a default: the only reachable mode. Guard (1) above proved Transport
    // preserves region, so both sides are the same mesh and ids match by
    // construction. NearestNeighbor becomes reachable only when Transport can
    // cross regions, which needs a per-channel tolerance the graph cannot yet
    // express -- see docs/plans/2026-09-11-comm-graph-runner-wiring.md.
    opts.mode         = MatchMode::GlobalId;
    opts.nn_tolerance = 0.0;

    auto built = CommGraph::build(MPI_Comm_c2f(base), src_side, dst_side, opts);
    if (!built.ok()) { status = built.error(); break; }
    const CommGraph& g = built.value();

    if (!g.specs_are_disjoint()) {
      status = Failure{"E-RT-TRANSPORT-OVERLAP",
                     "the two components' rank sets overlap, which one "
                     "global_ids list per rank cannot describe. The lowering "
                     "does not emit Transport in that case today; closing it "
                     "needs send_plan()/recv_plan() and a backend that "
                     "consumes them (R31)."};
      break;
    }

    // CommGraph knows nothing of components-per-entity; carry the arc's count
    // across the assignment.
    const int nc_planned = p.spec.components;
    p.spec            = in_src ? g.source_spec() : g.dest_spec();
    p.spec.components = nc_planned;

    p.gather  = in_src ? g.source_indices() : std::vector<std::size_t>{};
    p.scatter = in_dst ? g.dest_indices() : std::vector<std::size_t>{};

    p.local_src.clear();
    p.local_dst.clear();
    p.local_src.reserve(g.local_pairs().size());
    p.local_dst.reserve(g.local_pairs().size());
    for (const LocalCopyPair& lp : g.local_pairs()) {
      p.local_src.push_back(lp.source_local_index);
      p.local_dst.push_back(lp.dest_local_index);
    }

    // Both buffers, both sized total(), on every rank -- see TransportPlan.
    p.send.assign(p.total(), 0.0);
    p.recv.assign(p.total(), 0.0);
    p.has_graph = true;
  }

  if (status.ok() && transport_plans.size() == 1 &&
      transport_plans[0].has_graph) {
    if (auto* mpi = dynamic_cast<MpiTransport*>(transport.get())) {
      // COLLECTIVE, once, and every rank reaches it: the condition is derived
      // from the plan, which is identical everywhere.
      status = mpi->build_plan(transport_plans[0].spec);
    } else {
      // A CommGraph-backed plan carries subset gather/scatter indices that
      // only MpiTransport::build_plan() knows how to turn into a real
      // cross-rank exchange -- a "copy" or other backend given this spec
      // would silently move the wrong data rather than fail. `has_graph`'s
      // own doc comment (runner_impl.hpp) already excludes this case; this
      // is that exclusion made a refusal instead of an assumption. Safe
      // below the collective marker because the condition (`dynamic_cast`
      // on `config.transport`'s concrete type) depends only on
      // `RunnerConfig`, which is identical on every rank.
      status = Failure{"E-RT-TRANSPORT",
                     "a CommGraph-backed Transport plan requires the \"mpi\" "
                     "transport backend, but the configured transport is '" +
                         transport->name() +
                         "'; set RunnerConfig::transport = \"mpi\" alongside "
                         "base_communicator"};
    }
  }

  // Nothing outlives this function. `CommGraph` is a pure value
  // (comm_graph.hpp:151-159) and holds no MPI resource, so the Runner never
  // acquires a collective destructor and the R24 asymmetric-destruction hang
  // stays out.
  for (MPI_Comm& c : comms) {
    if (c != MPI_COMM_NULL) MPI_Comm_free(&c);
  }
  return status;
#endif
}

Result<void> Runner::Impl::setup(ComponentSet& components,
                                 FieldRegistry& registry) {
  // NO EARLY RETURN BETWEEN HERE AND THE STATUS-AGREEMENT CHECKPOINT BELOW.
  // attach_comm_graphs() is collective over `config.base_communicator`, but
  // these two resolutions are not: they read THIS rank's own FieldRegistry and
  // buffer sizes, and fail rank-divergently on an ordinary misconfiguration
  // (E-RT-UNBOUND, E-RT-SPACE, E-RT-LAYOUT, E-RT-SIZE above; E-RT-UNBOUND in
  // resolve_instructions()). A rank returning here while its peers walked on
  // would leave them blocked in MPI_Comm_split forever.
  //
  // resolve_instructions() is still skipped when resolve_buffers() failed --
  // it indexes `slots`, which resolve_buffers() only finishes filling on
  // success -- but the rank reaches the checkpoint either way, which is what
  // the collective needs.
  Result<void> r = resolve_buffers(registry);
  if (r.ok()) r = resolve_instructions();

  // THE STATUS-AGREEMENT CHECKPOINT. One MPI_Allreduce turns the rank-local
  // verdict above into a job-wide one, so either every rank enters
  // attach_comm_graphs() or none does. Gated on exactly the condition
  // attach_comm_graphs() itself uses to decide whether it is a real collective
  // (see the top of its body), so a caller that never sets base_communicator
  // -- the vast majority -- executes not one extra instruction.
#if FUSE_HAVE_MPI
  if (config.base_communicator.has_value()) {
    const MPI_Comm base =
        MPI_Comm_f2c(static_cast<MPI_Fint>(*config.base_communicator));
    if (base == MPI_COMM_NULL) {
      // Same verdict attach_comm_graphs() would reach, and reached from the
      // same input on every rank, so returning here is not divergent.
      return Failure{"E-RT-TRANSPORT-COMM",
                   "RunnerConfig::base_communicator is MPI_COMM_NULL"};
    }
    const int local_ok = r.ok() ? 1 : 0;
    int       all_ok   = 0;
    if (MPI_Allreduce(&local_ok, &all_ok, 1, MPI_INT, MPI_MIN, base) !=
        MPI_SUCCESS) {
      return Failure{"E-RT-TRANSPORT-COMM",
                   "MPI_Allreduce failed while agreeing setup status across "
                   "the base communicator"};
    }
    if (all_ok == 0 && r.ok()) {
      // This rank's own setup is fine; a peer's is not. Refuse anyway -- the
      // job cannot proceed, and proceeding alone means hanging alone. The rank
      // that actually failed returns its own diagnostic below instead, which is
      // the one worth reading.
      return Failure{"E-RT-SETUP-PEER",
                   "another rank of the base communicator failed to resolve "
                   "its buffers or instructions, so this rank must not enter "
                   "the collective CommGraph construction; see that rank's own "
                   "error for the cause"};
    }
  }
#endif
  if (!r.ok()) return r;
  {
    const Result<void> ec = resolve_epoch_participants();
    if (!ec.ok()) return ec;
    const Result<void> cg = attach_comm_graphs();
    if (!cg.ok()) return cg;
  }
  r = build_ticks(components);
  if (!r.ok()) return r;
  // Last: resolve_regions() needs the ticks and the resolved buffers, and it
  // is the single place spec §5.3's preconditions are checked
  // (E-ITER-NOCONVERGE, E-ITER-MAXITER, E-ITER-NOROLLBACK).
  r = resolve_regions(components);
  if (!r.ok()) return r;
  initialized = true;
  return Result<void>::success();
}

// ---------------------------------------------------------------------------
// Interpretation
// ---------------------------------------------------------------------------

namespace {

/// Packs the two component ids an `Epoch` names into one key.
[[nodiscard]] std::uint64_t epoch_key(int src, int dst) {
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(src)) << 32) |
         static_cast<std::uint32_t>(dst);
}

}  // namespace

/// The tag the agreement reduction uses on the base communicator.
/// Defined with every other tag in fuse/transport/tags.hpp, which explains why
/// they have to be chosen together.
constexpr int kEpochTag = tags::kEpoch;

Result<void> Runner::Impl::resolve_epoch_participants() {
#if FUSE_HAVE_MPI
  if (!config.base_communicator.has_value()) return Result<void>::success();

  // Checked once, here, rather than assumed: the guaranteed floor is 32767 and
  // `kEpochTag` is below it, so this cannot fire on a conforming MPI -- but a
  // refusal at setup is a better failure than "Invalid tag" from inside a
  // reduction on one rank while the others wait.
  {
    void* ub = nullptr;
    int   found = 0;
    if (MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &ub, &found) == MPI_SUCCESS &&
        found != 0 && ub != nullptr && *static_cast<int*>(ub) < tags::kMaxUsed) {
      return Failure{"E-RT-EPOCH-TAG",
                     "this MPI allows tags only up to " +
                         std::to_string(*static_cast<int*>(ub)) +
                         ", below the highest tag this library uses (" +
                         std::to_string(tags::kMaxUsed) + ")"};
    }
  }

  epoch_participants.clear();
  for (const Instruction& ins : plan.instructions()) {
    if (ins.kind != OpKind::Epoch) continue;
    const std::uint64_t key =
        epoch_key(ins.attrs.src_rank_set, ins.attrs.dst_rank_set);
    if (epoch_participants.count(key) != 0) continue;

    // The union of the two components' rank sets, sorted and de-duplicated.
    // Derived from the graph, which is identical on every rank, so every
    // participant builds the same list -- that is what makes the schedule in
    // reconcile_epoch() agree without anyone communicating to establish it.
    std::vector<int> ranks;
    for (const int component : {ins.attrs.src_rank_set, ins.attrs.dst_rank_set}) {
      if (component < 0) continue;
      const auto id = static_cast<std::uint32_t>(component);
      if (id >= graph.component_count()) continue;
      const RankSet& rs = graph.component(ComponentId{id}).ranks;
      for (int r = rs.first; r < rs.first + rs.count; ++r) ranks.push_back(r);
    }
    std::sort(ranks.begin(), ranks.end());
    ranks.erase(std::unique(ranks.begin(), ranks.end()), ranks.end());
    if (ranks.empty()) {
      return Failure{"E-RT-EPOCH-EMPTY",
                     "a rank-agreement point names no participating ranks; "
                     "the plan should not have compiled (the scheduler checks "
                     "that its participants match its data move's)"};
    }
    epoch_participants.emplace(key, std::move(ranks));
  }
#endif
  return Result<void>::success();
}

Result<void> Runner::Impl::reconcile_epoch(const Instruction& ins) {
#if FUSE_HAVE_MPI
  if (!config.base_communicator.has_value()) return Result<void>::success();
  const auto it =
      epoch_participants.find(epoch_key(ins.attrs.src_rank_set, ins.attrs.dst_rank_set));
  if (it == epoch_participants.end()) return Result<void>::success();

  const MPI_Comm base =
      MPI_Comm_f2c(static_cast<MPI_Fint>(*config.base_communicator));
  if (base == MPI_COMM_NULL) {
    return Failure{"E-RT-EPOCH-COMM",
                   "RunnerConfig::base_communicator is MPI_COMM_NULL at a "
                   "rank-agreement point"};
  }
  int me = 0;
  if (MPI_Comm_rank(base, &me) != MPI_SUCCESS) {
    return Failure{"E-RT-EPOCH-COMM", "MPI_Comm_rank failed at a rank-agreement point"};
  }

  const std::vector<int>& part = it->second;
  const auto self = std::find(part.begin(), part.end(), me);
  if (self == part.end()) {
    // This rank does not take part in the move, so it must not take part in
    // the agreement either: joining would make the others wait for a rank
    // with nothing to report.
    return Result<void>::success();
  }
  const auto n = static_cast<int>(part.size());
  const auto my_index = static_cast<int>(std::distance(part.begin(), self));

  // The schedule is arithmetic over (participants, index) and lives in
  // fuse/runtime/epoch_schedule.hpp, where a Tier 0 test checks that every
  // send has exactly one matching receive at 2..16 participants. A mismatch
  // here is a hang, which is only observable in a real multi-rank run, so it
  // is worth having a form that a unit test can falsify.
  //
  // WHY NOT A SUB-COMMUNICATOR. Splitting one per pair would mean holding it
  // for the Runner's lifetime and freeing it in a destructor, and
  // `MPI_Comm_free` is collective -- so destruction order would become a
  // correctness question. The transport layer avoids that deliberately
  // ("nothing outlives this function"); this follows the same rule.
  int status = step_failure.has_value() ? 0 : 1;
  const EpochSchedule sched = epoch_schedule(n, my_index);

  for (int ri = 0; ri < sched.reduce_recv_count; ++ri) {
    const int from = sched.reduce_recv_from[static_cast<std::size_t>(ri)];
    int peer_status = 1;
    if (MPI_Recv(&peer_status, 1, MPI_INT, part[static_cast<std::size_t>(from)],
                 kEpochTag, base, MPI_STATUS_IGNORE) != MPI_SUCCESS) {
      return Failure{"E-RT-EPOCH-COMM",
                     "MPI_Recv failed while agreeing status at a "
                     "rank-agreement point"};
    }
    status = status < peer_status ? status : peer_status;
  }
  if (sched.reduce_send_to >= 0) {
    if (MPI_Send(&status, 1, MPI_INT,
                 part[static_cast<std::size_t>(sched.reduce_send_to)], kEpochTag,
                 base) != MPI_SUCCESS) {
      return Failure{"E-RT-EPOCH-COMM",
                     "MPI_Send failed while agreeing status at a "
                     "rank-agreement point"};
    }
  }
  if (sched.result_recv_from >= 0) {
    if (MPI_Recv(&status, 1, MPI_INT,
                 part[static_cast<std::size_t>(sched.result_recv_from)],
                 kEpochTag, base, MPI_STATUS_IGNORE) != MPI_SUCCESS) {
      return Failure{"E-RT-EPOCH-COMM",
                     "MPI_Recv failed while receiving the agreed status"};
    }
  }
  for (int to = 1; to <= sched.result_send_count; ++to) {
    if (MPI_Send(&status, 1, MPI_INT, part[static_cast<std::size_t>(to)],
                 kEpochTag, base) != MPI_SUCCESS) {
      return Failure{"E-RT-EPOCH-COMM",
                     "MPI_Send failed while publishing the agreed status"};
    }
  }

  ++stats.epochs_reconciled;
  if (status == 0 && !step_failure.has_value()) {
    // This rank is fine and a partner is not. Stopping here is the point of
    // the whole mechanism: proceeding alone means blocking alone at the next
    // exchange. The rank that actually failed reports its own diagnostic,
    // which is the one worth reading.
    return Failure{"E-RT-EPOCH-PEER",
                   "a rank taking part in this exchange failed; stopping here "
                   "rather than waiting for it at the next one. The failing "
                   "rank reports the cause."};
  }
#else
  (void)ins;
#endif
  return Result<void>::success();
}

Result<void> Runner::Impl::execute(const Instruction& ins, const Resolved& r) {
  const auto in_ptr = [&](std::uint8_t k) -> const double* {
    return slots[ins.in[k].value()].data;
  };
  const auto out_ptr = [&]() -> double* { return slots[ins.out.value()].data; };
  const std::size_t n = r.out_total;

  switch (ins.kind) {
    case OpKind::Source:
      // The arc already aliases component storage; nothing to do.
      break;

    case OpKind::Sink:
    case OpKind::Accelerate: {  // Phase 7 replaces this with a real accelerator.
      if (ins.n_in == 0 || !ins.out.valid()) break;
      num::copy(in_ptr(0), out_ptr(), n);
      break;
    }

    case OpKind::Interpolate: {
      if (ins.n_in == 0 || !ins.out.valid()) break;
      if (r.spacetime < 0) {
        // The `lag(k)` no-op of src/ir/rules_channel.cpp: it exists so the
        // TimeLevel shift has an arc to live on and moves no data.
        num::copy(in_ptr(0), out_ptr(), n);
        break;
      }

      // Spec §7.3, tier 1: `dst = sum_j T(dst_step, j) * (W x_j)`, with `W`
      // the identity here because the channel's spatial half is the separate
      // `Remap` node and `T (x) W == (T (x) I)(I (x) W)`.
      //
      // The loop is exactly the one `apply_spacetime` documents: zero once,
      // then accumulate over the source steps that feed this target step.
      // Which source steps those are was resolved at setup (`row_begin`), and
      // `check_spacetime_schedule()` has already proved every one of them was
      // written this period.
      SpacetimeInfo& st = spacetimes[static_cast<std::size_t>(r.spacetime)];
      double* out = out_ptr();
      for (std::size_t k = 0; k < n; ++k) out[k] = 0.0;

      const auto tgt = static_cast<std::size_t>(ins.dst_step);
      const SpatialWeightsView spatial = st.identity();
      for (std::size_t k = st.row_begin[tgt]; k < st.row_begin[tgt + 1]; ++k) {
        const auto src_step = static_cast<std::size_t>(st.temporal.col[k]);
        const Result<void> a = apply_spacetime(
            spatial, st.temporal, st.history.data() + src_step * st.width, out,
            st.elements, st.components, src_step, tgt);
        if (!a.ok()) return a;
      }
      break;
    }

    case OpKind::Remap: {
      if (ins.n_in == 0 || !ins.out.valid()) break;
      if (r.remap < 0) {
        // The identity -- and it IS the conservative map here, not a stand-in
        // for one. resolve_remap() reached this branch only after checking
        // every cell of both regions for the same measure and the same
        // centroid at the same local index, so `dst[i] = src[i]` moves each
        // value to the cell it already belonged to. Any pair that failed that
        // check either carries real weights below or never got past setup.
        num::copy(in_ptr(0), out_ptr(), n);
        break;
      }
      // Spec §5.4: `g = W f`, with `W` the conservative FV<->FV matrix built
      // from the supermesh at SETUP (src/remap/intx_weights.cpp). Nothing is
      // allocated, looked up or rebuilt here -- `remap_weights` was filled in
      // resolve_remap() and the pointers are owned by `remap_cache`.
      const RemapWeights& w =
          *remap_weights[static_cast<std::size_t>(r.remap)];
      return apply_weights(w, in_ptr(0), out_ptr(),
                           slots[ins.out.value()].components);
    }

    case OpKind::Sample: {
      if (ins.n_in == 0 || !ins.out.valid()) break;
      if (r.spacetime >= 0) {
        // Spec §7.3: file this source step's value in the temporal history.
        // The arc still carries the latest sample so that the SSA value is
        // defined and inspectable; the projection reads `history`.
        SpacetimeInfo& st = spacetimes[static_cast<std::size_t>(r.spacetime)];
        const auto j = static_cast<std::size_t>(ins.src_step);
        num::copy(in_ptr(0), st.history.data() + j * st.width, st.width);
        num::copy(in_ptr(0), out_ptr(), n);
        break;
      }
      const std::size_t idx = sample_count[ins.out.value()];
      // r.accum, not ins.attrs.accum: see resolve_instructions().
      num::accumulate_into(r.accum, in_ptr(0), out_ptr(), n, idx);
      sample_count[ins.out.value()] = idx + 1;
      break;
    }

    case OpKind::Accumulate: {
      if (ins.n_in == 0 || !ins.out.valid()) break;
      num::copy(in_ptr(0), out_ptr(), n);
      // The sample counter belongs to the accumulator arc this reduction
      // reads, not to the arc it writes.
      const std::size_t count = sample_count[ins.in[0].value()];
      num::accumulate_finalize(ins.attrs.accum, out_ptr(), n, count);
      sample_count[ins.in[0].value()] = 0;  // reset for the next window
      break;
    }

    case OpKind::Extrapolate: {
      if (ins.n_in == 0 || !ins.out.valid()) break;
      // Phase 4 keeps no history, so order 0 is the only honest behaviour.
      num::extrapolate(0, in_ptr(0), n, out_ptr(), n);
      break;
    }

    case OpKind::Merge: {
      if (!ins.out.valid() || ins.n_in == 0) break;
      double* out = out_ptr();
      switch (ins.attrs.merge) {
        case MergeOp::First:
          // The first contributing channel in declaration order wins; the
          // others are not read at all. An override, not a combination.
          num::copy(in_ptr(0), out, n);
          break;
        case MergeOp::Sum:
        case MergeOp::Mean: {
          for (std::size_t i = 0; i < n; ++i) out[i] = 0.0;
          for (std::uint8_t k = 0; k < ins.n_in; ++k) {
            const double* src = in_ptr(k);
            for (std::size_t i = 0; i < n; ++i) out[i] += src[i];
          }
          if (ins.attrs.merge == MergeOp::Mean) {
            // Unweighted, as declared: the model carries no coverage
            // information that could weight the contributions, and inventing
            // weights here would be inventing physics.
            const double inv = 1.0 / static_cast<double>(ins.n_in);
            for (std::size_t i = 0; i < n; ++i) out[i] *= inv;
          }
          break;
        }
      }
      break;
    }

    case OpKind::Scale: {
      if (ins.n_in == 0 || !ins.out.valid()) break;
      // The factor is exact in the plan and becomes a double exactly once,
      // here, at the one point where it has to. Converting per element, or
      // carrying a double through the plan, would round it repeatedly.
      const double f = ins.attrs.scale.to_double();
      const double* src = in_ptr(0);
      double* out = out_ptr();
      for (std::size_t i = 0; i < n; ++i) out[i] = f * src[i];
      break;
    }

    case OpKind::Epoch: {
      // Two jobs, in this order: agree, then pass the value on.
      //
      // The agreement is `reconcile_epoch`, which reduces this rank's status
      // over the ranks that took part in the move this point guards. It runs
      // BEFORE the copy so that a rank whose partners have failed does not
      // publish a value derived from a transfer that did not complete.
      const Result<void> agreed = reconcile_epoch(ins);
      if (!agreed.ok()) return agreed;
      if (ins.n_in == 0 || !ins.out.valid()) break;
      num::copy(in_ptr(0), out_ptr(), n);
      break;
    }

    case OpKind::Map:
    case OpKind::Reduce:
    case OpKind::Closure:
      // Refused by the scheduler (E-SCHED-MULTISCALE), so a plan containing
      // one never reaches the interpreter. Refusing again rather than falling
      // through to a copy: a reserved operator that behaved as a pass-through
      // would run a micro-macro coupling as though the closure had been
      // applied.
      return Failure{"E-RT-MULTISCALE",
                     std::string("operator '") + to_cstring(ins.kind) +
                         "' is reserved for multiscale coupling and has no "
                         "implementation; the plan should not have compiled"};

    case OpKind::Relax: {
      if (ins.n_in == 0 || !ins.out.valid()) break;
      double* out = out_ptr();
      std::vector<double>& prev = previous[ins.out.value()];  // pre-sized
      num::relax(ins.attrs.omega, prev.data(), in_ptr(0), out, n);
      num::copy(out, prev.data(), n);
      break;
    }

    case OpKind::Transport: {
      if (ins.n_in == 0 || !ins.out.valid()) break;
      if (r.transport < 0) {
        return Failure{"E-RT-TRANSPORT",
                     "Transport instruction resolved to no plan"};
      }
      detail::TransportPlan& p =
          transport_plans[static_cast<std::size_t>(r.transport)];

      if (!p.has_graph) {
        const Result<void> prep = transport->prepare(p.spec);
        if (!prep.ok()) return prep;
        return transport->move(in_ptr(0), out_ptr(), n);
      }

      const double*     in  = in_ptr(0);
      double*           out = out_ptr();
      const std::size_t nc =
          static_cast<std::size_t>(p.spec.components < 1 ? 1 : p.spec.components);

      // Matches whose two ends are on this rank never reach the Transport at
      // all -- comm_graph.hpp:75-86. One global_ids list cannot describe a
      // rank that is both source and destination, so splitting these out is
      // what makes the exchange expressible, not an optimization.
      for (std::size_t k = 0; k < p.local_src.size(); ++k) {
        const std::size_t s = p.local_src[k] * nc;
        const std::size_t d = p.local_dst[k] * nc;
        for (std::size_t c = 0; c < nc; ++c) out[d + c] = in[s + c];
      }

      // Gather. `global_ids` is a SUBSET of this rank's entities, so it does
      // not index the field buffer; source_indices() is the map that does.
      for (std::size_t k = 0; k < p.gather.size(); ++k) {
        const std::size_t s = p.gather[k] * nc;
        for (std::size_t c = 0; c < nc; ++c) p.send[k * nc + c] = in[s + c];
      }

      const Result<void> prep = transport->prepare(p.spec);
      if (!prep.ok()) return prep;
      // p.total(), NOT the arc's n: move() requires exactly elements *
      // components OF THE PLANNED SPEC, and the planned spec is the subset
      // this rank actually exchanges.
      const Result<void> mv =
          transport->move(p.send.data(), p.recv.data(), p.total());
      if (!mv.ok()) return mv;

      for (std::size_t k = 0; k < p.scatter.size(); ++k) {
        const std::size_t d = p.scatter[k] * nc;
        for (std::size_t c = 0; c < nc; ++c) out[d + c] = p.recv[k * nc + c];
      }
      return Result<void>::success();
    }

    case OpKind::Residual: {
      if (ins.n_in < 1 || !ins.out.valid()) break;
      double* out = out_ptr();
      const double* a = in_ptr(0);
      if (ins.n_in > 1) {
        const double* b = in_ptr(1);
        for (std::size_t i = 0; i < n; ++i) out[i] = a[i] - b[i];
      } else {
        std::vector<double>& prev = previous[ins.out.value()];  // pre-sized
        for (std::size_t i = 0; i < n; ++i) out[i] = a[i] - prev[i];
        num::copy(a, prev.data(), n);
      }
      break;
    }

    case OpKind::Constrain: {
      if (ins.n_in == 0 || !ins.out.valid()) break;
      double* out = out_ptr();
      // Phase 4 reports; it does not correct.
      num::copy(in_ptr(0), out, n);

      if (r.constrain < 0) break;
      const ConstrainInfo& ci = constrains[static_cast<std::size_t>(r.constrain)];
      const Slot& src = slots[ci.before_arc];
      const std::vector<double>& wb = weight_sets[ci.weights_before];
      const std::vector<double>& wa = weight_sets[ci.weights_after];

      ++stats.conservation_checks;
      // A true integral over each region: the field value per entity against
      // that entity's physical measure, ghosts already zero-weighted. This is
      // the quantity a conservative remap must preserve; comparing the
      // operator's own input with its own copy of it never can.
      const double before = num::weighted_sum(src.data, wb.data(), wb.size());
      const double after = num::weighted_sum(out, wa.data(), wa.size());
      const double scale = std::abs(before) > 0.0 ? std::abs(before) : 1.0;
      const double rel = std::abs(after - before) / scale;
      if (rel > stats.worst_conservation_error) {
        stats.worst_conservation_error = rel;
      }
      if (rel > config.conservation_tolerance) {
        ++stats.conservation_violations;
        if (config.abort_on_violation) {
          return Failure{"E-RT-CONSERVATION",
                       "relative conservation error " + std::to_string(rel) +
                           " (integral " + std::to_string(before) + " -> " +
                           std::to_string(after) + ") exceeds tolerance " +
                           std::to_string(config.conservation_tolerance)};
        }
      }
      break;
    }

    case OpKind::Diagnose: {
      if (ins.n_in == 0 || !ins.out.valid()) break;
      double* out = out_ptr();
      num::copy(in_ptr(0), out, n);
      if (num::has_nonfinite(out, n)) {
        return Failure{"E-RT-NONFINITE",
                     "non-finite value detected in arc %" +
                         std::to_string(ins.out.value()) + " after " +
                         std::to_string(stats.periods) + " complete period(s)"};
      }
      break;
    }
  }
  return Result<void>::success();
}

Result<void> Runner::Impl::advance(ComponentSet& components, const Tick& tick,
                                   int iteration) {
  StepContext ctx;
  ctx.t_begin = Rational{period_index} * plan.period() + tick.time;
  ctx.t_end = ctx.t_begin + tick.dt;
  ctx.step = period_index * tick.per_period + tick.local_step;
  // Spec §5.3: 0 outside a region, and the 0-based sweep index inside one. A
  // component re-advanced over the same interval with iteration > 0 has had
  // restore_state() called first (see iteration.cpp).
  ctx.iteration = iteration;
  // component.hpp puts this check on the driver: Component cannot enforce it,
  // and a negative dt handed to an integrator is a plausible wrong answer,
  // not a crash.
  if (!ctx.valid()) {
    return Failure{"E-RT-STEP",
                 "step context [" + ctx.t_begin.to_string() + ", " +
                     ctx.t_end.to_string() + ") for component '" +
                     components.at(tick.slot).name() + "' is not valid"};
  }
  return components.at(tick.slot).advance(ctx);
}

Result<void> Runner::Impl::step_period(ComponentSet& components) {
  const std::vector<Instruction>& stream = plan.instructions();

  // The period is one merged walk over two pre-built, pre-sorted sequences:
  // the component ticks and the instruction stream.
  //
  // TICK CONVENTION. Each instant t is three phases, in this order:
  //
  //   1. every instruction stamped t with `TickPhase::BeforeTicks` -- every
  //      chain instance, i.e. every delivery. It reads values valid AT t and
  //      writes a value the ticks below must see;
  //   2. every component tick that BEGINS at t, advancing over [t, t + dt);
  //   3. every instruction stamped t with `TickPhase::AfterTicks` -- `Sample`,
  //      and only `Sample`, which by construction wants the value the tick in
  //      phase 2 just produced. Over four substeps of a producer writing
  //      2, 4, 6, 8, that is what makes the window mean 5 and not the 8 that
  //      advancing the whole period up front would give.
  //
  // `schedule()` has already sorted the stream by (time, phase, ...), so this
  // walk only has to decide, per instruction, whether the ticks at exactly its
  // own time have run yet: `BeforeTicks` says no, `AfterTicks` says yes. The
  // ticks are consumed by a single monotone cursor, which is sound precisely
  // because the phase is part of the sort key -- within one instant no
  // `AfterTicks` instruction can precede a `BeforeTicks` one and demand the
  // ticks early.
  //
  // Phase 1 is why a target that takes several steps per coupling period sees
  // a different value on each of them (spec §7.3, and the plain multirate case
  // too). The chain that closes target step k is stamped exactly where step
  // k+1 begins; running it in phase 1 hands step k+1 the value step k earned,
  // which is the shortest lag causality allows -- the projection for step k is
  // not complete before its own end. Ticks that begin at t are still advanced
  // lazily, at the first instruction that cannot precede them, so a component
  // with no instruction after its tick is advanced by the drain loop below.
  //
  // Across the period boundary the exchange remains explicitly lagged: the
  // last target step's value is written at t = period, after every tick of the
  // period, and is consumed by the target's first advance of the next one.
  // Spec §5.3: the checkpoint is taken HERE, before the first advance of the
  // period, and not at the region's enter instruction. By the time control
  // reaches the region the participants have already advanced once -- the
  // ticks below run before the instructions that read their output -- so a
  // save_state() at region entry would capture post-advance state, and every
  // restore would leave the component one advance ahead of where the sweep
  // thinks it is. That is a wrong answer with no symptom: the residual history
  // still decreases, just towards the fixed point of a different operator.
  if (!regions.empty()) {
    const Result<void> c = checkpoint_regions(components);
    if (!c.ok()) return c;
  }

  std::size_t ti = 0;
  std::size_t i = 0;
  while (i < stream.size()) {
    const Instruction& ins = stream[i];
    const bool ticks_here = ins.phase == TickPhase::AfterTicks;
    while (ti < ticks.size() &&
           (ticks[ti].time < ins.time ||
            (ticks_here && !(ins.time < ticks[ti].time)))) {
      // A component refusing its step is the canonical reason a rank fails
      // mid-period, so it carries forward exactly like an instruction failure
      // does: record it, stop advancing (the remaining components would be
      // stepping on values this step never produced), and keep walking so the
      // agreement points are still reached. Returning here is what leaves
      // every partner blocked in the next exchange.
      if (!step_failure.has_value()) {
        const Result<void> a = advance(components, ticks[ti], 0);
        if (!a.ok()) step_failure = a.error();
      }
      ++ti;
    }
    if (ins.region_enter >= 0) {
      detail::RegionRuntime& r =
          regions.at(static_cast<std::size_t>(ins.region_enter));
      const Result<void> rr = run_region(components, r);
      if (!rr.ok()) return rr;
      i = r.exit + 1;
      continue;
    }
    // After a failure this rank keeps executing the COLLECTIVE instructions
    // and skips the local ones (`is_collective`, fuse/ir/op.hpp).
    //
    // Skipping the local arithmetic costs nothing: the period is being
    // abandoned. Skipping a collective would be fatal -- the partners are
    // already inside it, waiting for this rank, which is the same hang as
    // returning at the point of failure. So the data move still happens; the
    // values it moves are whatever the buffers hold, and they are never used,
    // because the agreement point immediately after it ends the period on
    // every participant.
    if (step_failure.has_value() && !is_collective(ins.kind)) { ++i; continue; }

    const Result<void> r = execute(ins, resolved[i]);
    if (!r.ok()) {
      if (!step_failure.has_value()) step_failure = r.error();
      // An Epoch that fails has already reconciled, so there is nothing left
      // to tell the partners and no reason to keep walking.
      if (ins.kind == OpKind::Epoch) return *step_failure;
      ++i;
      continue;
    }
    ++stats.instructions_executed;
    ++i;
  }
  // Every tick the walk above left behind. With `BeforeTicks` the common
  // phase, a tick is advanced only by the first instruction stamped strictly
  // later, so the tail of a period routinely ends up here -- this is the
  // normal path, not just the "plan with no instruction after the last tick"
  // corner it was written for.
  for (; ti < ticks.size(); ++ti) {
    if (step_failure.has_value()) break;
    const Result<void> a = advance(components, ticks[ti], 0);
    if (!a.ok()) {
      step_failure = a.error();
      break;
    }
  }

  if (step_failure.has_value()) {
    // Reported at the END of the period, not at the point of failure: by here
    // every agreement point after the failure has been reached, so the
    // partners know too. `periods` is deliberately not incremented -- a period
    // that failed did not happen.
    const Failure f = *step_failure;
    step_failure.reset();
    return f;
  }

  ++stats.periods;
  ++period_index;
  return Result<void>::success();
}

// ---------------------------------------------------------------------------
// Public surface
// ---------------------------------------------------------------------------

Runner::Runner(const CouplingGraph& graph, const OperatorDAG& dag,
               ExecutionPlan plan, const MeshBackend& backend)
    : impl_(std::make_unique<Impl>(graph, dag, std::move(plan), backend)) {}
Runner::~Runner() = default;
Runner::Runner(Runner&&) noexcept = default;
Runner& Runner::operator=(Runner&&) noexcept = default;

Result<void> Runner::initialize(ComponentSet& components,
                                FieldRegistry& registry,
                                const RunnerConfig& config) {
  impl_->initialized = false;
  impl_->config = config;

  // A function returning Result must never throw. Every allocation here, the
  // backend's measures, and Rational's overflow guards can; they become setup
  // diagnostics instead.
  try {
    Result<std::unique_ptr<Transport>> transport =
        TransportRegistry::instance().create(config.transport);
    if (!transport.ok()) return transport.error();
    impl_->transport = std::move(transport.value());
    // The one hook by which a caller reaches a backend's one-time,
    // concrete-typed setup (MpiTransport::set_communicator+build_plan,
    // LowFiveTransport::set_location+build_plan) before any Transport
    // instruction is resolved. Absent for CopyTransport and any config that
    // does not set it -- see RunnerConfig::transport_setup.
    if (config.transport_setup) {
      const Result<void> setup_r = config.transport_setup(*impl_->transport);
      if (!setup_r.ok()) return setup_r;
    }
    return impl_->setup(components, registry);
  } catch (const std::exception& e) {
    return Failure{"E-RT-INIT",
                 std::string{"runner setup failed: "} + e.what()};
  } catch (...) {
    return Failure{"E-RT-INIT",
                 "runner setup threw an exception that does not derive from "
                 "std::exception"};
  }
}

Result<void> Runner::run_period(ComponentSet& components) {
  if (!impl_->initialized) {
    return Failure{"E-RT-INIT",
                 "run_period() called before a successful initialize()"};
  }
  try {
    return impl_->step_period(components);
  } catch (const std::exception& e) {
    return Failure{"E-RT-RUN", std::string{"period failed: "} + e.what()};
  } catch (...) {
    return Failure{"E-RT-RUN",
                 "period threw an exception that does not derive from "
                 "std::exception"};
  }
}

Result<void> Runner::run(ComponentSet& components, std::int64_t periods) {
  if (periods < 0) {
    return Failure{"E-RT-PERIODS", "cannot run " + std::to_string(periods) +
                                     " periods; the count must not be "
                                     "negative"};
  }
  for (std::int64_t i = 0; i < periods; ++i) {
    const Result<void> r = run_period(components);
    if (!r.ok()) return r;
  }
  return Result<void>::success();
}

const RunStats& Runner::stats() const noexcept { return impl_->stats; }

const std::vector<IterationStats>& Runner::iteration_stats() const noexcept {
  return impl_->region_stats;
}

const std::vector<std::string>& Runner::checkpoint_explanations()
    const noexcept {
  return impl_->region_explain;
}

Result<const double*> Runner::buffer(ArcId arc) const {
  if (!arc.valid() || arc.value() >= impl_->slots.size()) {
    return Failure{"E-RT-ARC", "no buffer for arc " + std::to_string(arc.value())};
  }
  return static_cast<const double*>(impl_->slots[arc.value()].data);
}

}  // namespace fuse
