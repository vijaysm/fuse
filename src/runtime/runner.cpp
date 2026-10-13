#include "fuse/base/config.hpp"
#include "fuse/runtime/epoch_schedule.hpp"
#include "fuse/transport/tags.hpp"
#include "fuse/runtime/runner.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "runner_impl.hpp"
#include <fmt/format.h>
#include "fuse/num/kernels.hpp"
#include "fuse/num/residual_norm.hpp"
#include "fuse/schedule/slab.hpp"
#include "fuse/schedule/window.hpp"

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
                                           RegionId a, RegionId b,
                                           ComponentId placement) {
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
    // The placement's own decomposition: two regions coincide only as seen
    // by the SAME component (a Remap's two ends always share one -- the
    // Transport that changes placement is emitted before it).
    const Result<const MeshView*> v = backend.placed_view(id.value(), placement);
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
  device_storage.assign(buffers.size(), {});
  device_mirror.assign(buffers.size(), {});
  device_mirror_epoch.assign(buffers.size(), 0);
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
    if (view.space() != MemorySpace::Host &&
        config.execution_space != RunnerConfig::RunnerExecutionSpace::Device) {
      // docs/dev/device-resident-arcs.md: a device-registered field is only
      // reachable by the Space-generic kernels and the space-aware boundary
      // copies, both of which exist in Device mode alone; the host
      // interpreter dereferences slot pointers directly.
      return Failure{"E-RT-SPACE",
                   "field '" + b.label + "' is registered in " +
                       to_cstring(view.space()) +
                       " memory, but RunnerConfig::execution_space is Host; "
                       "select Device, or register the field in host memory"};
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
    s.device = view.space() == MemorySpace::Device;
  }
  return Result<void>::success();
}

void Runner::Impl::place_arcs_on_device() {
  const std::size_t n_arcs = slots.size();
  // 1 = some host-side reader exists; the arc stays in `storage`.
  std::vector<unsigned char> pinned(n_arcs, 0);
  // 1 = a Space-generic instruction reads it: a component-owned host field
  // with this set gets a device mirror (see Impl::device_mirror).
  std::vector<unsigned char> read_by_kernel(n_arcs, 0);
  const auto pin = [&](std::uint32_t a) {
    if (a < n_arcs) pinned[a] = 1;
  };
  const auto pin_arc = [&](ArcId a) {
    if (a.valid()) pin(a.value());
  };

  // Instructions: only execute_on()'s Space-generic cases may see a device
  // pointer, and only through span_c()/span_m(). Remap's distributed-nodal
  // operator and the spacetime projections run host code on their arcs.
  // Source/ExternalDelivery (alias or mirror), Sink/Accelerate/Epoch (a
  // space-aware copy) and Transport (docs/dev/device-resident-arcs.md
  // Phase 3: a device-resident gather/scatter when the backend is
  // GPU-aware, a host-staged pack/unpack otherwise -- either way its own
  // code decides the space, not this function) handle either space and pin
  // nothing.
  // FUSE_DEVICE_DEBUG: one line per instruction (its class) and per arc (where
  // it ended up), the same convention as FUSE_NODAL_DEBUG above.
  const bool trace = std::getenv("FUSE_DEVICE_DEBUG") != nullptr;
  const auto& instructions = plan.instructions();
  for (std::size_t i = 0; i < instructions.size(); ++i) {
    const Instruction& ins = instructions[i];
    const Resolved&    r   = resolved[i];
    bool generic = false, space_aware = false;
    switch (ins.kind) {
      case OpKind::Accumulate: case OpKind::Extrapolate: case OpKind::Merge:
      case OpKind::Scale: case OpKind::Relax: case OpKind::Residual:
        generic = true;
        break;
      case OpKind::Remap:       generic = r.nodal_dist < 0; break;
      case OpKind::Sample:      generic = r.spacetime < 0; break;   // history branch is host
      case OpKind::Interpolate: generic = r.spacetime < 0; break;   // apply_spacetime is host
      case OpKind::Source: case OpKind::ExternalDelivery: case OpKind::Sink:
      case OpKind::Accelerate: case OpKind::Epoch: case OpKind::Transport:
        space_aware = true;
        break;
      default: break;
    }
    if (trace) {
      std::string ins_arcs;
      for (std::uint8_t k = 0; k < ins.n_in; ++k) {
        ins_arcs += fmt::format(" %{}", ins.in[k].valid() ? static_cast<int>(ins.in[k].value()) : -1);
      }
      const char* cls = generic ? "generic" : space_aware ? "space-aware" : "PINS";
      const int out_arc = ins.out.valid() ? static_cast<int>(ins.out.value()) : -1;
      fmt::print(stderr, "[device-place] ins {:<3} {:<16} in:{} out:%{} {}\n", i,
                 std::string_view{to_cstring(ins.kind)}, ins_arcs, out_arc, cls);
    }
    if (generic) {
      for (std::uint8_t k = 0; k < ins.n_in; ++k) {
        if (ins.in[k].valid()) read_by_kernel[ins.in[k].value()] = 1;
      }
      continue;
    }
    if (space_aware) continue;
    for (std::uint8_t k = 0; k < ins.n_in; ++k) pin_arc(ins.in[k]);
    pin_arc(ins.out);
  }
  // An ordinary region's block, carried and verify arcs are reached from
  // iteration.cpp / fixed_point_residual.cpp only through the space-aware
  // slot_to_host()/host_to_slot()/slot_host_read() helpers (step 3), so
  // they may live on the device. A WINDOW-scoped region's bookkeeping
  // (advance_window(), priming, lagged imports, the t0 correction) still
  // dereferences its arcs on the host, so every arc it names stays there;
  // constrain reads its `before` arc on the host.
  for (const detail::RegionRuntime& R : regions) {
    if (!R.window_scoped) continue;
    for (const detail::RegionBlock& b : R.blocks) {
      pin(b.value_arc); pin(b.residual_arc); pin(b.feed_arc);
    }
    for (const std::uint32_t a : R.carried_arcs) pin(a);
    for (const std::uint32_t a : R.verify_arcs) pin(a);
    for (const ArcId a : R.window_lagged_import_arcs) pin_arc(a);
  }
  for (const auto& [node, cfg] : window_member_config) {
    pin_arc(cfg.window_import);
    pin_arc(cfg.window_import_delivery_arc);
  }
  for (const auto& [field, arc] : window_export_field_arc) pin_arc(arc);
  for (const auto& [field, arc] : window_import_field_arc) pin_arc(arc);
  for (const detail::ConstrainInfo& ci : constrains) pin(ci.before_arc);

  for (std::size_t i = 0; i < n_arcs; ++i) {
    Slot& s = slots[i];
    if (s.total() == 0) continue;
    if (s.owned_by_component) {
      // A host-registered field a kernel reads: a persistent mirror,
      // refreshed when a kernel reads it after component code ran, instead
      // of a staging lane at every consumer. A device-registered field is
      // already where the kernels run.
      if (!s.device && read_by_kernel[i] && device_mirror[i].extent(0) < s.total()) {
        device_mirror[i] = Kokkos::View<double*, DeviceMemSpace>(
            Kokkos::view_alloc("fuse::dev_mirror", Kokkos::WithoutInitializing),
            s.total());
      }
      continue;
    }
    if (pinned[i]) continue;
    if (device_storage[i].extent(0) < s.total()) {
      device_storage[i] = Kokkos::View<double*, DeviceMemSpace>(
          Kokkos::view_alloc("fuse::dev_arc"), s.total());  // zero-filled, like storage
    }
    std::vector<double>().swap(storage[i]);  // the host copy is gone for good
    s.data = device_storage[i].data();
    s.device = true;
    ++stats.device_resident_arcs;
  }
  if (trace) {
    const std::vector<BufferDecl>& buffers = plan.buffers().buffers;
    for (std::size_t i = 0; i < n_arcs; ++i) {
      const Slot& s = slots[i];
      const std::string label = i < buffers.size() ? buffers[i].label : std::string{};
      const char* where = s.owned_by_component ? "component" : pinned[i] ? "pinned" : "coupler";
      fmt::print(stderr, "[device-place] arc %{:<3} n={:<6} {:<10} {}{} {}\n", i, s.total(),
                 where, s.device ? "device" : "host",
                 device_mirror[i].extent(0) > 0 ? " mirrored" : "", label);
    }
  }
}

void Runner::Impl::build_transport_device_buffers() {
  const bool trace = std::getenv("FUSE_DEVICE_DEBUG") != nullptr;
  transport_device.clear();
  transport_device.resize(transport_plans.size());
  transport_in_scratch.clear();
  transport_in_scratch.resize(transport_plans.size());
  transport_out_scratch.clear();
  transport_out_scratch.resize(transport_plans.size());
  for (std::size_t pi = 0; pi < transport_plans.size(); ++pi) {
    const Instruction& ins = plan.instructions()[transport_instruction[pi]];
    const Slot& in_slot  = slots[ins.in[0].value()];
    const Slot& out_slot = slots[ins.out.value()];
    // A mixed pair (one arc device, one host) gets no mirror: the
    // OpKind::Transport case's own CommGraph-backed branch in execute_on()
    // cannot run a Kokkos gather/scatter kernel against one host and one
    // device pointer, so it stages whichever side is device through the
    // scratch buffer sized here, runs the ordinary scalar host loop, and
    // (for a device `out`) stages the result back. An all-host pair (the
    // common case, and every plan under Host) needs no scratch at all --
    // both vectors stay default-constructed (empty), which `.data()` on an
    // empty vector would make a null pointer the hot loop never reaches
    // because `in_slot.device`/`out_slot.device` are false.
    if (in_slot.device && out_slot.device) {
      transport_device[pi] =
          detail::mirror_transport_plan<DeviceMemSpace>(transport_plans[pi]);
      if (trace) {
        fmt::print(stderr,
                   "[device-place] transport plan {:<3} mirrored to device "
                   "(gpu_aware={})\n",
                   pi,
                   (!transport_plans[pi].has_graph ||
                    graph_transports[pi] == nullptr)
                       ? false
                       : graph_transports[pi]->is_gpu_aware());
      }
      continue;
    }
    if (in_slot.device) transport_in_scratch[pi].assign(in_slot.total(), 0.0);
    if (out_slot.device) transport_out_scratch[pi].assign(out_slot.total(), 0.0);
    if (trace && (in_slot.device || out_slot.device)) {
      fmt::print(stderr,
                 "[device-place] transport plan {:<3} MIXED arcs (in {} out "
                 "{}), host-staged per side\n",
                 pi, in_slot.device ? "device" : "host",
                 out_slot.device ? "device" : "host");
    }
  }
}

Result<std::uint32_t> Runner::Impl::weights_for(RegionId region,
                                                int rank_set,
                                                std::size_t elements,
                                                int components) {
  // Keyed by placement too: on a rank in two placements one region has two
  // decompositions, so two different measure vectors. `components` is small
  // (checked >= 1 by plan_buffers), the placement a component index.
  const auto key = (static_cast<std::uint64_t>(region.value()) << 32) |
                   (static_cast<std::uint64_t>(
                        static_cast<std::uint16_t>(rank_set + 1)) << 16) |
                   static_cast<std::uint16_t>(components);
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

  // Resolved by NAME: `reg.mesh` is the GRAPH's MeshId, a numbering
  // independent of the backend's (MeshBackend::mesh_id). Passing it straight
  // through -- as this did -- sizes whichever backend mesh happens to share
  // the position: the "wrong-but-plausible count with no diagnostic" that
  // make_region_sizes is written to avoid.
  if (!reg.mesh.valid() || reg.mesh.value() >= graph.mesh_count()) {
    return Failure{"E-RT-MEASURE", "region '" + reg.name +
                                       "' names a mesh the graph does not declare"};
  }
  const auto bid = backend.mesh_id(graph.mesh(reg.mesh).name);
  if (!bid.ok()) return bid.error();
  const ComponentId placement =
      rank_set >= 0 ? ComponentId{static_cast<std::uint32_t>(rank_set)}
                    : ComponentId{};
  const Result<const MeshView*> view = backend.placed_view(bid.value(), placement);
  if (!view.ok()) return view.error();
  const MeshView* mv = view.value();
  if (mv == nullptr) {
    return Failure{"E-RT-MEASURE",
                 "mesh backend returned no view for region '" + reg.name + "'"};
  }

  // Which of field_extent()'s two shapes the field on this region implies
  // (mesh_view.hpp). A cell field's conserved amount is its measure-weighted
  // sum, unchanged from before; a NODAL Extensive field carries no cell
  // measure to weight by at all (MeshView::measure(0, i) throws by
  // contract -- a vertex has none), and its conserved amount is documented
  // (consistent_interp.hpp's "DENSITY VS LOAD") to be the amount ITSELF: the
  // H^T D builder already converts a density source to a load before this
  // ever sees it, so the target-side integral this Constrain checks is the
  // plain sum of owned nodal loads -- weight 1 per owned vertex, 0 per ghost,
  // same "zero a ghost rather than truncate the range" rule the cell path
  // uses to avoid double-counting another rank's copy.
  const auto kind = region_extent_kind(graph, region);
  if (!kind.ok()) return kind.error();
  const bool nodal = kind.value() == RegionExtentKind::Vertex;
  const int dim = nodal ? 0 : (reg.dimension >= 0 ? reg.dimension : mv->dimension());
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
      const double m =
          !mv->is_owned(dim, i) ? 0.0 : (nodal ? 1.0 : mv->measure(dim, i));
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

  // N-transport-runtime-1 (REVIEW.md pass2), closed 2026-09-25 (KNOWN_ISSUES.md
  // D10). weights_for() builds this rank's OWNED share of each integral --
  // every ghost zeroed -- so once either region spans more than one rank the
  // two shares cannot be compared in isolation: mass that left this rank is
  // accounted for on the receiving one. The check used to REFUSE a multi-rank
  // target (E-RT-CONSERVATION-MPI) because reducing inside a non-collective
  // operator would hang: a rank that had already failed skips local
  // instructions while its partners wait in the reduction. Constrain is now
  // collective (fuse::is_collective), so every rank of the base communicator
  // reaches the reduction, failed or not, and the shares are summed there.
  // A failed rank does not poison the sum: it contributes a flag instead of
  // a value, and the verdict is skipped when any rank raised it, since the
  // agreement point that follows ends the period with the real error.
  //
  // Reduced whenever a communicator is configured, not only for multi-rank
  // targets: a one-rank source feeding a one-rank target on a DIFFERENT rank
  // has both shares on different ranks too, and the reduction is one
  // Allreduce of three doubles per conservative channel per period.
  const bool reduce = config.base_communicator.has_value();

  // weights_for() integrates a NODAL field as the plain sum of its owned
  // values (weight 1 per vertex): correct for an Extensive nodal load, which
  // is already an amount, and meaningless for an Intensive nodal field, whose
  // integral needs a vertex measure this library does not define. Refuse
  // rather than compare a vertex sum with an integral.
  for (const ArcId a : {before, ins.out}) {
    const ValueType& t = dag.arc(a).type;
    if (t.conservation != Conservation::Intensive || !t.region.valid()) continue;
    const auto kind = region_extent_kind(graph, t.region);
    if (kind.ok() && kind.value() == RegionExtentKind::Vertex) {
      return Failure{"E-RT-CONSTRAIN",
                     "the conservation check on arc %" + std::to_string(ins.out.value()) +
                         " would integrate an Intensive field on a vertex region (arc %" +
                         std::to_string(a.value()) +
                         "), and a vertex has no measure; a nodal field's conserved "
                         "amount is defined only for an Extensive load"};
    }
  }

  const Slot& in_slot = slots[before.value()];
  const Slot& out_slot = slots[ins.out.value()];
  const Result<std::uint32_t> wb = weights_for(
      dag.arc(before).type.region, dag.arc(before).type.rank_set,
      in_slot.elements, in_slot.components);
  if (!wb.ok()) return wb.error();
  const Result<std::uint32_t> wa = weights_for(
      dag.arc(ins.out).type.region, dag.arc(ins.out).type.rank_set,
      out_slot.elements, out_slot.components);
  if (!wa.ok()) return wa.error();

  r.constrain = static_cast<std::int32_t>(constrains.size());
  constrains.push_back(
      ConstrainInfo{before.value(), wb.value(), wa.value(), reduce});
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
  // A Remap's two ends share one placement: rules_channel.cpp emits the
  // Transport that changes placement BEFORE the Remap.
  const int remap_rank_set = dag.arc(ins.out).type.rank_set;
  const ComponentId placement =
      remap_rank_set >= 0
          ? ComponentId{static_cast<std::uint32_t>(remap_rank_set)}
          : ComponentId{};
  // The input's placement. Equal to the output's for every Remap except the
  // one that moves data between placements itself (fuse/model/remap_kind.hpp,
  // rules_channel.cpp): there the input lives on the SOURCE component's
  // ranks and no Transport preceded it. Such a Remap takes the distributed
  // path below and is exempt from the per-placement weights refusal, which
  // is about the rank-local weights this Remap does not build.
  const int in_rank_set = dag.arc(ins.in[0]).type.rank_set;
  const bool moves_placement = in_rank_set >= 0 && remap_rank_set >= 0 &&
                               in_rank_set != remap_rank_set;

  // Weights are built on THIS placement's decomposition: RemapPlan::placement
  // is set below and every builder resolves its sides through
  // MeshBackend::placed_view (KNOWN_ISSUES D5, closed 2026-09-25). Until then
  // a Remap on a non-default decomposition was refused (E-RT-REMAP-PLACEMENT)
  // because weights built on the default view would have indexed the wrong
  // entities with no diagnostic. A placement the backend was never told about
  // is still refused, by placed_view itself (E-MESH-PLACEMENT).

  // Discretization, order and conservation live on the FIELDS, not on the
  // operator, so they are read once here from the channel this node was
  // lowered from -- reused below both to build `RemapPlan p` (a pair the
  // builder cannot handle is refused there with E-REMAP-UNSUPPORTED rather
  // than silently downgraded here) and to decide, BEFORE either return
  // point, whether this Remap's target needs the cross-rank reconciliation
  // docs/dev/nodal-fields.md's "H^T on a distributed interface" specifies
  // (lead's ruling 2026-09-24): the decision depends only on the graph --
  // the target's discretization, its conservation, and the destination
  // component's declared rank-set SIZE, none of it rank-specific -- so it is
  // identical on every rank and safe to compute before either the coincident
  // or the real-weights path returns.
  const NodeId origin = ins.origin_node;
  const FieldSpec* sf_ptr = nullptr;
  const FieldSpec* tf_ptr = nullptr;
  ComponentId      src_component;
  ComponentId      dst_component;
  if (origin.valid() && origin.value() < dag.node_count()) {
    const ChannelId ch = dag.node(origin).origin;
    if (ch.valid() && ch.value() < graph.channel_count()) {
      const Channel& channel = graph.channel_at(ch);
      sf_ptr = &graph.field(channel.source);
      tf_ptr = &graph.field(channel.target);
      src_component = graph.owner_of(channel.source);
      dst_component = graph.owner_of(channel.target);
    }
  }

  // MLS weights are serial: they are built from each rank's own view of the
  // source cloud, and a partitioned cloud would give each rank a different
  // neighbourhood, and so a different, wrong answer, with no diagnostic.
  // mls_weights.cpp can only detect a partitioned NativeMeshBackend; the
  // declared placement can always be read here, so the refusal lives here.
  if (ins.attrs.method == "mls" || ins.attrs.method.rfind("mls:", 0) == 0) {
    for (const ComponentId owner : {src_component, dst_component}) {
      if (owner.valid() && owner.value() < graph.component_count() &&
          graph.component(owner).ranks.size() > 1U) {
        return Failure{
            "E-RT-REMAP-MLS-PARALLEL",
            label + " uses method '" + ins.attrs.method + "', and component '" +
                graph.component(owner).name + "' is placed on " +
                std::to_string(graph.component(owner).ranks.size()) +
                " ranks. Moving-least-squares weights are serial only: a "
                "partitioned point cloud gives each rank different neighbours "
                "(docs/user/point-cloud-remap.md). Place both components on one "
                "rank, or use method 'consistent'"};
      }
    }
  }

  std::int32_t nodal_dist_idx = -1;
  if (sf_ptr != nullptr && tf_ptr != nullptr) {
    // Round 4 (KNOWN_ISSUES.md D8, defect 2, found by a real host MPI run):
    // the ORIGINAL guard here read only `dst_ranks.size() > 1` and
    // `region_extent_kind(tgt) == Vertex` -- exactly right for
    // "consistent-transpose" (H^T D delivers a load TO a nodal target), but
    // it silently missed "consistent" (H interpolates FROM a nodal
    // SOURCE): when the target was a cell region, `kind(tgt)` was `Cell`,
    // the inner check never ran, and control fell through to the OLD
    // rank-local path below -- which assumes its local `MeshView` already
    // holds the region's WHOLE mesh, wrong (or E-INTERP-NOGEOMETRY on a
    // rank owning zero local entities of one side) whenever either
    // component is genuinely partitioned across ranks. Both `src_ranks`
    // and `dst_ranks` are checked now, and either region's kind, so the
    // decision still depends only on the declared graph -- unchanged from
    // the original guard's own invariant -- and is identical on every
    // rank.
    const auto src_kind = region_extent_kind(graph, src);
    if (!src_kind.ok()) return src_kind.error();
    const auto tgt_kind = region_extent_kind(graph, tgt);
    if (!tgt_kind.ok()) return tgt_kind.error();
    const bool src_nodal = src_kind.value() == RegionExtentKind::Vertex;
    const bool tgt_nodal = tgt_kind.value() == RegionExtentKind::Vertex;

    const RankSet* src_set =
        (src_component.valid() && src_component.value() < graph.component_count())
            ? &graph.component(src_component).ranks
            : nullptr;
    const RankSet* dst_set =
        (dst_component.valid() && dst_component.value() < graph.component_count())
            ? &graph.component(dst_component).ranks
            : nullptr;
    const std::size_t src_ranks = src_set != nullptr ? src_set->size() : 1U;
    const std::size_t dst_ranks = dst_set != nullptr ? dst_set->size() : 1U;
    // Round 5 (a real host MPI run again): `src_ranks > 1 || dst_ranks > 1`
    // was still not the condition. At exactly TWO ranks a disjoint pair is
    // one rank EACH -- neither side is partitioned, yet the source lives on
    // rank 0 and the target on rank 1, so no rank holds both and the
    // rank-local path below produced E-INTERP-NOGEOMETRY ("they hold 7 and 0
    // entities") on whichever rank was missing a side. What decides this is
    // not whether either placement spans several ranks but whether the two
    // TOGETHER are confined to one: only then is every entity the operator
    // needs local to the rank building it.
    const bool both_on_one_rank =
        src_ranks == 1U && dst_ranks == 1U &&
        (src_set == nullptr || dst_set == nullptr ||
         src_set->min_rank() == dst_set->min_rank());
    // "A single-rank placement on BOTH sides skips this whole path, since
    // nothing is shared" -- the lead's ruling, generalized from the
    // original's `dst_ranks.size() > 1` to look at whichever side (or
    // both) is genuinely partitioned: a nodal source replicated via
    // Allgatherv still needs the distributed path when only the EVAL side
    // is spread across ranks, since a rank holding zero local evaluation
    // entities cannot build ITS row from a local-only view either.
    if ((!both_on_one_rank || moves_placement || config.force_distributed_nodal_remap) &&
        (src_nodal || tgt_nodal)) {
      if (src_nodal && tgt_nodal) {
        // Neither `build_nodal_dist_remap` orientation has been run against
        // a nodal-to-nodal channel (both regions Vertex-kind); refuse by
        // name rather than silently picking one orientation's convention
        // (KNOWN_ISSUES.md D8) -- verified by
        // NodalRemapRunnerTierZero.BothSidesNodalOnMultipleRanksRefuses.
        return Failure{
            "E-RT-REMAP-NODAL-PARALLEL",
            label + " has BOTH a nodal source and a nodal target, and at "
                    "least one side's component is on more than one rank ("
                "source: " + std::to_string(src_ranks) + ", target: " +
                std::to_string(dst_ranks) +
                "). A nodal-to-nodal distributed Remap is not implemented "
                "(KNOWN_ISSUES.md D8). Route through a cell-located "
                "intermediate field, or place both components on one rank."};
      }
      // "consistent" needs a nodal SOURCE (H); "consistent-transpose" needs
      // a nodal TARGET (H^T D) -- build_nodal_dist_remap's own rule (it
      // mirrors build_consistent_weights_for_plan's non-distributed one),
      // checked here too since a method paired with the WRONG orientation
      // must be refused, not silently misapplied to the other side.
      const bool method_matches_orientation =
          (ins.attrs.method == "consistent" && src_nodal) ||
          (ins.attrs.method == "consistent-transpose" && tgt_nodal);
      if (method_matches_orientation) {
        // Round 2 (KNOWN_ISSUES.md D8): build distributed rows via
        // `build_nodal_dist_remap` instead of refusing --
        // `nodal_remap_distributed.hpp`'s Allgatherv-replicated primitive,
        // verified serially in
        // test/backend/nodal_remap_distributed_nullcomm_test.cpp and under
        // real MPI in test/parallel/nodal_remap_distributed_test.cpp;
        // round 4 extended it to the nodal-SOURCE orientation
        // (test/parallel/nodal_remap_runner_test.cpp's wall_velocity
        // channel). Any OTHER method (including the supermesh conservative
        // ones, which build FV<->FV weights only) still refuses:
        // per-placement supermesh weights for a nodal side remain
        // unimplemented (KNOWN_ISSUES D5 item 3), and this track's own
        // scope is the two consistent-interpolation methods specifically,
        // not every method.
        const auto built = build_nodal_dist_remap(ins, label, src, tgt, sf_ptr, tf_ptr);
        if (!built.ok()) return built.error();
        nodal_dist_idx = built.value();
      } else {
        // Whether a shared vertex's per-holder rows are partial or already
        // complete depends on the decomposition the weights are built on,
        // and no Remap on a genuinely partitioned nodal side through a
        // supermesh method (or the orientation-mismatched consistent
        // method) has been run: on a replicated source every holder
        // computes the whole row and an owner sum would multiply the load
        // by the holder count, with no diagnostic. The condition is read
        // from the declared graph and placement alone, so every rank
        // refuses together.
        const std::string& nodal_side_name = src_nodal ? sf_ptr->name : tf_ptr->name;
        return Failure{
            "E-RT-REMAP-NODAL-PARALLEL",
            label + " has a nodal " + std::string(src_nodal ? "source" : "target") +
                " field '" + nodal_side_name + "', and at least one side's "
                "component is on more than one rank (source: " +
                std::to_string(src_ranks) + ", target: " + std::to_string(dst_ranks) +
                "), via method '" + ins.attrs.method +
                "'. A Remap touching a nodal field on more than one rank is "
                "supported only for method 'consistent' with a nodal SOURCE, "
                "or 'consistent-transpose' with a nodal TARGET "
                "(nodal_remap_distributed.hpp); a supermesh conservative "
                "method's per-placement weights are not built yet "
                "(KNOWN_ISSUES.md D8, D5 item 3). Place both components on "
                "one rank, use a cell-located field on the multi-rank side, "
                "or set Channel::remap_method to the matching orientation"};
      }
    }
  }
  if (nodal_dist_idx >= 0) {
    // Mutually exclusive with the rest of resolve_remap: the distributed
    // rows ARE this instruction's whole operator, not a correction applied
    // after the normal (local, replicated-mesh) path below. `r.remap` stays
    // -1 -- execute()'s OpKind::Remap case reads `r.nodal_dist >= 0` FIRST
    // and calls apply_nodal_dist_remap() instead of the weights-apply, per
    // NodalDistRemap's own comment.
    r.nodal_dist = nodal_dist_idx;
    return Result<void>::success();
  }

  const Coincidence c = regions_coincide(graph, backend, src, tgt, placement);
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
  // The decomposition the weights index. Part of the plan's key, so the same
  // channel on two placements builds two matrices rather than sharing one.
  p.placement = placement;
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
  if (sf_ptr != nullptr && tf_ptr != nullptr) {
    p.src_disc = sf_ptr->disc;
    p.tgt_disc = tf_ptr->disc;
    p.src_order = sf_ptr->order;
    p.tgt_order = tf_ptr->order;
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

  // KNOWN_ISSUES.md D11 (closing): the CSR conversion of THIS weights matrix,
  // built here -- at plan time, once -- so execute()'s OpKind::Remap case
  // never converts a triplet list on the hot path. remap_csr stays parallel
  // to remap_weights: both are pushed together, indexed by the same
  // r.remap.
  const Result<detail::RemapCsr> csr = build_remap_csr(*weights);
  if (!csr.ok()) return csr.error();

  r.remap = static_cast<std::int32_t>(remap_weights.size());
  remap_weights.push_back(weights);
  remap_csr.push_back(csr.value());
  if (config.execution_space == RunnerConfig::RunnerExecutionSpace::Device) {
    remap_csr_device.push_back(
        detail::mirror_remap_csr<DeviceMemSpace>(remap_csr.back()));
  }
  return Result<void>::success();
}

// ---------------------------------------------------------------------------
// KNOWN_ISSUES.md D11 (closing): builds the CSR conversion of one
// RemapWeights' COO triplets, ONCE, at the point resolve_remap() is about
// to push a new remap_weights entry. See detail::RemapCsr's own comment
// (runner_impl.hpp) for the layout this produces and why the two views'
// memory space is always Kokkos::HostSpace regardless of which host
// execution space actually applies the matvec.
// ---------------------------------------------------------------------------
Result<detail::RemapCsr> Runner::Impl::build_remap_csr(
    const RemapWeights& weights) {
  const std::size_t nnz = weights.val.size();
  if (weights.row.size() != nnz || weights.col.size() != nnz) {
    return Failure{"E-REMAP-COO",
                 "build_remap_csr: the triplet arrays disagree in length "
                 "(row " + std::to_string(weights.row.size()) + ", col " +
                     std::to_string(weights.col.size()) + ", val " +
                     std::to_string(nnz) + ")"};
  }
  for (std::size_t k = 0; k < nnz; ++k) {
    if (weights.row[k] < 0 ||
        static_cast<std::size_t>(weights.row[k]) >= weights.n_rows ||
        weights.col[k] < 0 ||
        static_cast<std::size_t>(weights.col[k]) >= weights.n_cols) {
      return Failure{"E-REMAP-COO",
                   "build_remap_csr: triplet " + std::to_string(k) + " is (" +
                       std::to_string(weights.row[k]) + ", " +
                       std::to_string(weights.col[k]) + "), outside the " +
                       std::to_string(weights.n_rows) + "x" +
                       std::to_string(weights.n_cols) + " matrix"};
    }
  }

  // A STABLE sort by row, not an assumption that `row` already is ascending
  // (see detail::RemapCsr's own comment for why the latter is false for
  // consistent_interp.hpp's "consistent-transpose" H^T D). Stability is what
  // makes this exact rather than merely convenient: for two triplets of the
  // SAME row, it preserves their original relative order, which is all a
  // per-row sum's floating-point rounding depends on -- `apply_weights`
  // (fuse/backend/remap_plan.hpp) accumulates a row's own entries in
  // ascending-k order regardless of what other rows' entries sit between
  // them in the array, and a stable sort by row reproduces exactly that
  // relative order for each row's own entries. When `row` already IS
  // ascending (the conservative supermesh builder), a stable sort of an
  // already-sorted sequence is the identity permutation, so this costs
  // nothing extra in the common case and changes nothing about it.
  std::vector<std::size_t> perm(nnz);
  for (std::size_t k = 0; k < nnz; ++k) perm[k] = k;
  std::stable_sort(perm.begin(), perm.end(),
                   [&weights](std::size_t a, std::size_t b) {
                     return weights.row[a] < weights.row[b];
                   });

  detail::RemapCsr csr;
  csr.row_begin = Kokkos::View<std::size_t*, Kokkos::HostSpace>(
      Kokkos::view_alloc("fuse::remap_csr_row_begin",
                        Kokkos::WithoutInitializing),
      weights.n_rows + 1);
  csr.col = Kokkos::View<std::size_t*, Kokkos::HostSpace>(
      Kokkos::view_alloc("fuse::remap_csr_col", Kokkos::WithoutInitializing),
      nnz);
  csr.val = Kokkos::View<double*, Kokkos::HostSpace>(
      Kokkos::view_alloc("fuse::remap_csr_val", Kokkos::WithoutInitializing),
      nnz);

  std::size_t k = 0;
  for (std::size_t row = 0; row < weights.n_rows; ++row) {
    csr.row_begin(row) = k;
    while (k < nnz && static_cast<std::size_t>(weights.row[perm[k]]) == row) {
      csr.col(k) = static_cast<std::size_t>(weights.col[perm[k]]);
      csr.val(k) = weights.val[perm[k]];
      ++k;
    }
  }
  csr.row_begin(weights.n_rows) = k;
  // Cannot fail given the bounds check above (every row[] value is in
  // [0, n_rows), so the scan above visits every one of them); asserted as a
  // Result rather than FUSE_ASSERT so a future change to the sort or the
  // scan that broke this is a diagnosed refusal, not undefined behaviour
  // from an under-filled row_begin.
  if (k != nnz) {
    return Failure{"E-REMAP-COO",
                 "build_remap_csr: internal error -- the row-sorted scan "
                 "consumed " + std::to_string(k) + " of " +
                     std::to_string(nnz) + " triplets"};
  }
  return csr;
}

// ---------------------------------------------------------------------------
// Nodal-remap track, round 2 (KNOWN_ISSUES.md D8). Replaces the deleted
// reduce_nodal_remap_target()/NodalReduceInfo design (see NodalDistRemap's
// own comment in runner_impl.hpp for why): a distributed nodal Remap target
// now gets ITS OWN operator built here via nodal_remap_distributed.hpp,
// rather than a per-rank LOCAL Remap answer reconciled afterwards.
// ---------------------------------------------------------------------------

Result<std::int32_t> Runner::Impl::build_nodal_dist_remap(
    const Instruction& ins, const std::string& label, RegionId src, RegionId tgt,
    const FieldSpec* sf_ptr, const FieldSpec* tf_ptr) {
  // `is_nodal` mirrors consistent_interp_detail::is_nodal (consistent_interp.hpp)
  // exactly -- duplicated rather than shared because that function lives in
  // an internal-to-that-header namespace runner.cpp does not otherwise reach
  // into, and this is one line, not a maintenance-prone algorithm.
  const auto is_nodal = [](const FieldSpec* f) {
    return f != nullptr && f->disc == Discretization::ContinuousGLL && f->order <= 1;
  };

  const std::string& method = ins.attrs.method;
  const bool transpose = (method == "consistent-transpose");
  // Round 2 activates this function only when `tgt` is Vertex-located
  // (resolve_remap's own caller already checked); "consistent" additionally
  // needs `src` nodal (H interpolates FROM a nodal source), and
  // "consistent-transpose" needs `tgt` nodal (H^T D delivers a load TO a
  // nodal target) -- both exactly what build_consistent_weights_for_plan
  // itself requires, checked here too since this path never calls that
  // function.
  if (!transpose && !is_nodal(sf_ptr)) {
    return Failure{"E-INTERP-DISC",
                   label + ": method 'consistent' builds H, the interpolation FROM "
                           "the source field's nodal DOFs; the source field is not "
                           "a continuous field at order <= 1"};
  }
  if (transpose && !is_nodal(tf_ptr)) {
    return Failure{"E-INTERP-DISC",
                   label + ": method 'consistent-transpose' delivers H^T D, a "
                           "conservative load transfer TO the target field's nodal "
                           "DOFs; the target field is not a continuous field at "
                           "order <= 1"};
  }

  // The nodal side is `src` for "consistent", `tgt` for "consistent-transpose"
  // -- the SAME orientation build_consistent_weights_for_plan uses (see that
  // function's own comment: "Build H itself with the SAME orientation the
  // paired forward channel would use"). The eval side is the other one.
  const RegionId nodal_region = transpose ? tgt : src;
  const RegionId eval_region = transpose ? src : tgt;
  const bool eval_is_nodal = transpose ? is_nodal(sf_ptr) : is_nodal(tf_ptr);

  // Each side as seen by ITS OWN component: the input region by the input
  // arc's placement, the output region by the output arc's. Before 2026-09-25
  // both used the mesh's default view, which coincided with the placed view
  // on the backends that had been run; it is the placed view the buffers are
  // sized by (make_region_sizes), so it is the placed view the operator must
  // read and write.
  const auto placement_of = [&](ArcId arc) {
    const int rs = dag.arc(arc).type.rank_set;
    return rs >= 0 ? ComponentId{static_cast<std::uint32_t>(rs)} : ComponentId{};
  };
  const ComponentId src_placement = placement_of(ins.in[0]);
  const ComponentId tgt_placement = placement_of(ins.out);
  const auto resolve = [&](RegionId rid, ComponentId who) -> Result<const MeshView*> {
    const Region& rg = graph.region(rid);
    if (!rg.mesh.valid() || rg.mesh.value() >= graph.mesh_count()) {
      return Failure{"E-RT-REMAP-MESH", label + ": region '" + rg.name + "' names no mesh"};
    }
    const auto bid = backend.mesh_id(graph.mesh(rg.mesh).name);
    if (!bid.ok()) return bid.error();
    const auto view = backend.placed_view(bid.value(), who);
    if (!view.ok()) return view.error();
    if (view.value() == nullptr) {
      return Failure{"E-RT-REMAP-MESH",
                     label + ": the backend returned no view for region '" + rg.name + "'"};
    }
    return view.value();
  };
  const auto nodal_view_r = resolve(nodal_region, transpose ? tgt_placement : src_placement);
  if (!nodal_view_r.ok()) return nodal_view_r.error();
  const auto eval_view_r = resolve(eval_region, transpose ? src_placement : tgt_placement);
  if (!eval_view_r.ok()) return eval_view_r.error();

  const int eval_dim = eval_is_nodal ? 0 : graph.region(eval_region).dimension;

  search::CommHandle comm_handle = search::kNullComm;
#if FUSE_HAVE_MPI
  if (config.base_communicator.has_value()) {
    comm_handle = *config.base_communicator;
  }
#endif

  const auto rows_r = build_distributed_interp_rows(*nodal_view_r.value(), *eval_view_r.value(),
                                                     eval_dim, comm_handle);
  if (!rows_r.ok()) {
    return Failure{rows_r.error().code,
                   label + ": build_distributed_interp_rows failed: " + rows_r.error().message};
  }

  detail::NodalDistRemap plan;
  plan.rows = rows_r.value();
  plan.nodal_view = nodal_view_r.value();
  plan.eval_view = eval_view_r.value();
  plan.eval_dim = eval_dim;
  plan.transpose = transpose;
  plan.src_is_load = transpose ? is_nodal(sf_ptr) : false;
  plan.components = static_cast<std::size_t>(
      slots[ins.out.value()].components < 1 ? 1 : slots[ins.out.value()].components);
  plan.comm_handle = comm_handle;

  const auto idx = static_cast<std::int32_t>(nodal_dist_remaps.size());
  nodal_dist_remaps.push_back(std::move(plan));
  return idx;
}

Result<void> Runner::Impl::apply_nodal_dist_remap(const Instruction& ins, const Resolved& r) {
  detail::NodalDistRemap& plan = nodal_dist_remaps[static_cast<std::size_t>(r.nodal_dist)];
  const detail::Slot& in_slot = slots[ins.in[0].value()];
  const detail::Slot& out_slot = slots[ins.out.value()];
  const std::size_t components = plan.components;

  // Opt-in trace (FUSE_NODAL_DEBUG). Host run 2026-09-25: on a disjoint
  // two-component placement the H direction delivered exact values while
  // H^T D delivered zero to every target vertex, and the two paths read as
  // symmetric. What decides it is what each rank actually holds here: how
  // many rows it built, how many values its input slot has and whether they
  // are zero, and how many entities its two views hold. Printed per rank so
  // the next run answers that instead of another reading.
  const bool trace = std::getenv("FUSE_NODAL_DEBUG") != nullptr;
  if (trace) {
    double in_abs = 0.0;
    for (std::size_t k = 0; k < in_slot.elements * components; ++k) in_abs += std::abs(in_slot.data[k]);
    int me = -1;
#if FUSE_HAVE_MPI
    if (config.base_communicator.has_value()) {
      MPI_Comm_rank(MPI_Comm_f2c(static_cast<MPI_Fint>(*config.base_communicator)), &me);
    }
#endif
    fmt::print(stderr,
               "[nodal-dist rank {}] arc %{} {} rows={} in.elements={} sum|in|={:.6g} "
               "nodal.nv={} eval.n={} out.elements={} src_is_load={}\n",
               me, ins.out.value(), plan.transpose ? "H^T" : "H", plan.rows.rows.size(),
               in_slot.elements, in_abs, plan.nodal_view->entity_count(0),
               plan.eval_view->entity_count(plan.eval_dim), out_slot.elements,
               plan.src_is_load);
  }

  if (!plan.transpose) {
    // "consistent": H. in_slot is the nodal side's own LOCAL buffer
    // (entity-major, owned + ghost) -- exactly apply_distributed_H's
    // `local_src_values` contract.
    const std::vector<double> local_src(in_slot.data, in_slot.data + in_slot.elements * components);
    std::vector<double> owned_out;
    const auto ah = apply_distributed_H(plan.rows, *plan.nodal_view, local_src, components,
                                        owned_out, plan.comm_handle);
    if (!ah.ok()) return ah.error();
    std::fill(out_slot.data, out_slot.data + out_slot.elements * components, 0.0);
    for (std::size_t k = 0; k < plan.rows.rows.size(); ++k) {
      const std::size_t li = plan.rows.rows[k].local_index;
      for (std::size_t c = 0; c < components; ++c) {
        out_slot.data[li * components + c] = owned_out[k * components + c];
      }
    }
    if (trace) {
      double s = 0.0;
      for (std::size_t k = 0; k < out_slot.elements * components; ++k) s += std::abs(out_slot.data[k]);
      fmt::print(stderr, "[nodal-dist] arc %{} H sum|out|={:.6g}\n", ins.out.value(), s);
    }
    return Result<void>::success();
  }

  // "consistent-transpose": H^T D. in_slot is the eval (point/density) side's
  // own LOCAL buffer; only its rows this rank's OWNED evaluation entities
  // name are read (via plan.rows.rows[k].local_index), scaled by that
  // entity's own measure UNLESS the eval side is itself already a load
  // (plan.src_is_load) -- the same density-to-load rule
  // build_consistent_weights_for_plan's transpose branch bakes into its
  // weights at build time, applied here to the VALUES at apply time instead.
  std::vector<double> owned_loads(plan.rows.rows.size() * components, 0.0);
  for (std::size_t k = 0; k < plan.rows.rows.size(); ++k) {
    const std::size_t li = plan.rows.rows[k].local_index;
    const double scale = plan.src_is_load ? 1.0 : plan.eval_view->measure(plan.eval_dim, li);
    for (std::size_t c = 0; c < components; ++c) {
      owned_loads[k * components + c] = in_slot.data[li * components + c] * scale;
    }
  }
  std::vector<double> out_local;
  const auto aht = apply_distributed_HT(plan.rows, *plan.nodal_view, *plan.eval_view, plan.eval_dim,
                                        owned_loads, components, out_local, plan.comm_handle);
  if (!aht.ok()) return aht.error();
  if (out_local.size() != out_slot.elements * components) {
    return Failure{"E-RT-NODAL-DIST-SIZE",
                   "apply_nodal_dist_remap: apply_distributed_HT returned " +
                       std::to_string(out_local.size()) + " values, expected " +
                       std::to_string(out_slot.elements * components)};
  }
  std::copy(out_local.begin(), out_local.end(), out_slot.data);
  if (trace) {
    double s_in = 0.0, s_out = 0.0;
    for (const double v : owned_loads) s_in += std::abs(v);
    for (const double v : out_local) s_out += std::abs(v);
    fmt::print(stderr, "[nodal-dist] arc %{} H^T sum|owned_loads|={:.6g} sum|out|={:.6g}\n",
               ins.out.value(), s_in, s_out);
  }
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
  remap_csr.clear();  // parallel to remap_weights; see build_remap_csr()
  remap_csr_device.clear();
  stats.device_dispatches = 0;
  stats.device_staged_copies = 0;
  stats.device_resident_arcs = 0;
  stats.device_boundary_copies = 0;
  stats.device_transport_moves = 0;
  // Both are APPENDED to by the OpKind::Transport case below, so they must be
  // reset here like every other setup-owned container. Runner::initialize() is
  // re-callable (it clears `initialized` at entry), and without these a second
  // initialize() would carry the first call's entries forward and build a
  // Transport per stale plan as well as per real one --
  // RunnerTransport.InitializeIsRepeatableOnTheSameRunner is what notices.
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
    // `Remap` is exempt: transferring between two differently-sized meshes is
    // exactly what it is for. Its sizes are reconciled against the matrix it
    // is actually given, in resolve_remap() below, which is a stronger check
    // than this one -- equal lengths never implied the two ends corresponded.
    //
    // `Transport` is exempt too, since per-placement decomposition (KNOWN_ISSUES
    // D5): its two ends are the SAME region in two components' decompositions,
    // so on a rank in one placement and not the other one end legitimately
    // holds 12 entities and the other 0. Whether that is right depends on
    // which kind of plan it gets -- a whole-arc copy needs equal sizes, a
    // CommGraph plan needs its indices in range -- and that is not known until
    // attach_comm_graphs(), so check_transport_extents() reconciles it there.
    //
    // `ExternalDelivery` is exempt for a STRONGER reason than either, in its
    // OWN outer guard rather than folded into the condition just below (an
    // existing mutation row -- build-evidence/mutations/transport-runtime.
    // toml's own `transport-held-to-equal-extents` -- anchors that exact
    // line verbatim; wrapping it here, instead of editing it, is what keeps
    // that anchor valid unchanged). Its input and output are not just two
    // placements of the SAME region (D5) but two DIFFERENT components' own
    // decompositions by design (src/ir/rules_channel.cpp's own `lower_
    // channel()`: the input carries the EXPORTING component's own producer
    // arc, the output is typed with the IMPORTING component's own
    // `rank_set` -- "the value...lands on the CONSUMER's own placement, not
    // the producer's"). On a rank hosting exactly one end, D24's own
    // contract ("a placement is decomposed over its own RankSet... a
    // component that does not live on this rank gets an empty view") makes
    // the two sizes disagree by construction: 2816 values from the
    // exporter's own producer into 0 at the (unhosted) importer's own arc,
    // or the mirror -- KNOWN_ISSUES.md D38, measured on the host's first
    // LowFive-bound MPMD/accept run past D36's own launch-layout fix.
    // Unlike `Transport`, there is no replacement check to add here:
    // `execute()`'s own `OpKind::ExternalDelivery` case (this file) never
    // reads the input at all -- it is purely the ordering edge op.hpp's own
    // header comment on `ExternalDelivery` documents -- so no kernel ever
    // reads `out_total` doubles from it, and the "heap overread" hazard
    // this whole reconciliation exists to catch (this function's own
    // comment, two paragraphs up) cannot arise for this kind. The output
    // arc's own size (what the application's write through `fuse_
    // view_field` actually gets, src/abi/participant.cpp's `impl_view_
    // field` via `Runner::field_window_view()`) is sized by the SAME
    // `sizes(region, rank_set)` callback (src/schedule/buffers.cpp's
    // `plan_buffers()`) already relied on, untested by this entry, for
    // every other coupler-owned arc in the system -- there is no new
    // invariant this kind needs that a narrower check could still catch.
    if (ins.kind != OpKind::ExternalDelivery) {
      if (ins.kind != OpKind::Remap && ins.kind != OpKind::Transport) {
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

namespace {

/// Time windows track: every component that is a member of some
/// window-scoped iteration region (`fuse::CouplingWindowScope`,
/// `fuse/model/iteration.hpp`) -- the ones `build_ticks()` below must NOT
/// give an ordinary per-native-interval `Tick`. Such a member's entire
/// within-window advance() sequence (every one of its own substeps, not
/// just one call) is driven by its own `OpKind::WindowAdvance` instruction
/// (`Runner::Impl::advance_window()`, called from `run_region()`'s body
/// loop, src/runtime/iteration.cpp) -- unlike an ordinary iteration
/// region's member, which still gets exactly one ordinary `Tick` per
/// region pass (its FIRST advance, before the region is ever reached; only
/// its RE-advances for sweeps k=1,2,... are diverted into
/// `RegionRuntime::tick_indices`). Giving a window member a `Tick` as well
/// would advance it twice over the same interval: once here, and once
/// inside its own `WindowAdvance` instruction's `plan_substeps()`/
/// `advance_straddle_split()` sequence, which has no way to know the
/// ordinary tick loop got there first.
[[nodiscard]] std::unordered_set<std::uint32_t> window_member_components(
    const CouplingGraph& g) {
  std::unordered_set<std::uint32_t> out;
  for (std::uint32_t i = 0; i < g.iteration_region_count(); ++i) {
    const IterationRegionDecl& decl =
        g.iteration_region(IterationRegionIdx{i});
    if (!decl.window_scope.has_value()) continue;
    for (const ComponentId c : window_region_members(g, decl)) {
      out.insert(c.value());
    }
  }
  return out;
}

}  // namespace

Result<void> Runner::Impl::build_ticks(ComponentSet& components) {
  ticks.clear();

  // KNOWN_ISSUES.md D12: one entry per ComponentSet slot, consumed by
  // advance() the next time it is about to suspend for an externally
  // driven component. Sized here (components.size() is already fixed by
  // this point in setup()) rather than in resolve_regions(), which runs
  // afterwards and only knows about the SUBSET of slots a region
  // checkpoints -- every other slot must still read as "false" (0).
  slot_just_checkpointed.assign(components.size(), 0);
  slot_just_restored.assign(components.size(), 0);

  const Rational period = plan.period();
  if (!(period > Rational{0})) {
    return Failure{"E-RT-INTERVAL",
                 "the plan's common period is " + period.to_string() +
                     ", which is not positive"};
  }

  // Every graph component needs an implementation and vice versa. A component
  // present in only one of the two is never advanced and never exchanged,
  // which is a wrong-physics failure with no symptom.
  //
  // A `ComponentKind::Coupler` entry is exempt from BOTH directions of that
  // rule: it is a placement, not a physics participant, so it has no
  // `register_fields()`/`advance()` implementation to supply (there is no
  // `add_coupler_impl()`-style call on `ComponentSet`, and there should
  // never need to be one), and it gets no `Tick` -- it is never due, never
  // advanced, never clocked. Its ranks still get a communicator in
  // `attach_comm_graphs()` below, exactly like any physics component's,
  // which is what lets a channel's Transport/Epoch actually reach them.
  // Which process this is, when a base communicator is set: a component is
  // stepped only on the processes of its own processor set (the placement
  // guide's contract -- a component placed on no processes "does not run
  // here, this run"). Without a communicator every component runs in this
  // one process, exactly as before. Found by the D3 host run (2026-09-28): a
  // process outside a component's processor set was calling its advance(),
  // where the component's own communicator is MPI_COMM_NULL.
  // Unit 2b (D19): `this_process` is now `Runner::Impl::this_process`, a
  // member (its own declaration comment has the full account), not a
  // local -- reset here, at the top of the ONE place that computes it,
  // so a second `build_ticks()` call (if initialize() ever runs twice)
  // never sees a stale value from a previous call.
  this_process.reset();
  if (config.force_this_process.has_value()) {
    // RunnerConfig::force_this_process's own comment: a tier-0 test's
    // override, taking priority over base_communicator so a build with
    // no MPI compiled in at all can still exercise per-rank hosting.
    this_process = config.force_this_process;
  } else {
#if FUSE_HAVE_MPI
  if (config.base_communicator.has_value()) {
    const MPI_Comm base =
        MPI_Comm_f2c(static_cast<MPI_Fint>(*config.base_communicator));
    int rank = 0;
    if (base == MPI_COMM_NULL || MPI_Comm_rank(base, &rank) != MPI_SUCCESS) {
      return Failure{"E-RT-TRANSPORT-COMM",
                     "could not read this process's rank in the base "
                     "communicator while building the step list"};
    }
    this_process = rank;
  }
#endif
  }

  all_ticks.clear();
  std::vector<bool> hosted(components.size(), false);
  std::vector<bool> matched(components.size(), false);
  // Time windows track: computed once, up front -- see
  // window_member_components()'s own comment for why a member of this set
  // gets no Tick below, ever, regardless of its own ComponentDecl::interval.
  const std::unordered_set<std::uint32_t> window_members =
      window_member_components(graph);
  for (std::uint32_t i = 0; i < graph.component_count(); ++i) {
    const ComponentDecl& decl = graph.component(ComponentId{i});
    if (decl.kind == ComponentKind::Coupler) continue;
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

    // Time windows track: this component's own advance() calls are entirely
    // owned by its OpKind::WindowAdvance instruction (advance_window(),
    // src/runtime/iteration.cpp) -- no ordinary Tick, at any interval, is
    // built for it. Its own ComponentDecl::interval is still read directly
    // by advance_window() as this member's fine subcycling rate, and still
    // contributes to ClockSet::checked_common_period()'s LCM (why a 7:13
    // window naturally produces more than one window instance per period);
    // neither of those needs a Tick to exist.
    if (window_members.count(i)) continue;

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
    // Every step of every component goes into `all_ticks`, identical on every
    // process; `ticks`, filtered below, holds only this process's own.
    hosted[slot] = !this_process.has_value() || decl.ranks.contains(*this_process);
    for (std::int64_t k = 0; k < n_exact.num(); ++k) {
      all_ticks.push_back(Tick{Rational{k} * dt, dt, slot, i, k, n_exact.num()});
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

  std::stable_sort(all_ticks.begin(), all_ticks.end(),
                   [](const Tick& a, const Tick& b) {
                     if (a.time != b.time) return a.time < b.time;
                     return a.order < b.order;
                   });
  for (const Tick& tk : all_ticks) {
    if (hosted[tk.slot]) ticks.push_back(tk);
  }

  // Which components would consume a value a failed step never produced: one
  // edge per channel, from the component owning its source field to the
  // component owning its target, then a transitive closure. Component counts
  // here are tens, so the plain cubic closure costs nothing and reads better
  // than a worklist. Built here because this is where graph components are
  // already matched to ComponentSet slots.
  const std::size_t nc = components.size();
  downstream_of.assign(nc, std::vector<char>(nc, 0));
  const auto slot_of = [&](ComponentId cid) -> std::size_t {
    const std::string& want = graph.component(cid).name;
    for (std::size_t j = 0; j < nc; ++j) {
      if (components.at(j).name() == want) return j;
    }
    return nc;  // declared but not supplied; the loop above already refused
  };
  for (std::uint32_t ci = 0; ci < graph.channel_count(); ++ci) {
    const Channel&    ch = graph.channel_at(ChannelId{ci});
    const std::size_t a  = slot_of(graph.owner_of(ch.source));
    const std::size_t b  = slot_of(graph.owner_of(ch.target));
    if (a < nc && b < nc && a != b) downstream_of[a][b] = 1;
  }
  for (std::size_t k = 0; k < nc; ++k) {
    for (std::size_t a = 0; a < nc; ++a) {
      if (downstream_of[a][k] == 0) continue;
      for (std::size_t b = 0; b < nc; ++b) {
        if (downstream_of[k][b] != 0) downstream_of[a][b] = 1;
      }
    }
  }

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

  // Set from the rank graph when a pair's placements partially overlap, and
  // consumed by the plan build below. Local to this function rather than
  // members of TransportPlan: runner_impl.hpp deliberately holds no MPI-only
  // type, which is why the local-copy pairs there are two plain vectors and
  // not a CommGraph type.
  // clear()+resize(), not assign(n, nullptr): assign copy-constructs its
  // fill value and unique_ptr is move-only.
  graph_transports.clear();
  graph_transports.resize(transport_plans.size());

  bool                       peer_plan_needed = false;
  std::vector<CommGraphPeer> send_peers;
  std::vector<CommGraphPeer> recv_peers;

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
    const auto src_cid = static_cast<std::uint32_t>(p.spec.src_rank_set);
    const auto dst_cid = static_cast<std::uint32_t>(p.spec.dst_rank_set);

    // EACH SIDE ITS OWN COMPONENT'S DECOMPOSITION (KNOWN_ISSUES D5). This used
    // to be one `backend.view(mesh)` handed to both sides, so a rank in both
    // placements offered the same piece as source and as destination, and two
    // partially overlapping placements could not both cover the region. The
    // failures here are local and identical on every rank (the same graph,
    // the same backend contract), so breaking before the collective is safe.
    const auto src_view = backend.placed_view(mid.value(), ComponentId{src_cid});
    if (!src_view.ok()) { status = src_view.error(); break; }
    const auto dst_view = backend.placed_view(mid.value(), ComponentId{dst_cid});
    if (!dst_view.ok()) { status = dst_view.error(); break; }

    // (3) Which dimension this Transport matches entities at: the region's
    // own (cell) dimension, unless every field on it is nodal, in which case
    // it is the mesh's VERTEX dimension (mesh_view.hpp's field_extent /
    // region_extent_kind -- the same verdict make_region_sizes and
    // weights_for already reach for this region). Wrong either way is silent
    // corruption, not a crash: matching a nodal field's Transport at the
    // cell dimension would send/receive the wrong COUNT of entities outright
    // (checked below by CommGraph itself), but matching it at dimension 0
    // when the field is actually cell-based would silently swap in the
    // vertex global-id space instead -- a same-count, wrong-entities transfer
    // with no size mismatch to catch it. So this, like checks (1)/(2) above,
    // reads only the graph -- identical on every rank -- and is safe to
    // `break` on before the collective CommGraph::build() below.
    const auto extent_kind = region_extent_kind(graph, p.region);
    if (!extent_kind.ok()) { status = extent_kind.error(); break; }
    const int dim = extent_kind.value() == RegionExtentKind::Vertex
                        ? 0
                        : (reg_side.dimension >= 0 ? reg_side.dimension
                                                   : src_view.value()->dimension());
    const bool in_src =
        graph.component(ComponentId{src_cid}).ranks.contains(my_rank);
    const bool in_dst =
        graph.component(ComponentId{dst_cid}).ranks.contains(my_rank);

    // A side's mesh must be non-null exactly when its comm_handle names a real
    // communicator (comm_graph.hpp:46-47).
    CommGraphSide src_side;
    src_side.dim         = dim;
    src_side.comm_handle = MPI_Comm_c2f(comms[src_cid]);
    src_side.mesh        = in_src ? src_view.value() : nullptr;

    CommGraphSide dst_side;
    dst_side.dim         = dim;
    dst_side.comm_handle = MPI_Comm_c2f(comms[dst_cid]);
    dst_side.mesh        = in_dst ? dst_view.value() : nullptr;

    CommGraphOptions opts;
    // NOT a default: the only reachable mode. Guard (1) above proved Transport
    // preserves region, so both sides are the same mesh and ids match by
    // construction. NearestNeighbor becomes reachable only when Transport can
    // cross regions, which needs a per-channel tolerance the graph cannot yet
    // express -- see docs/plans/2026-09-11-comm-graph-runner-wiring.md.
    opts.mode         = MatchMode::GlobalId;
    opts.nn_tolerance = 0.0;

    auto built = CommGraph::build(MPI_Comm_c2f(base), src_side, dst_side, opts);
    if (!built.ok()) {
      // CommGraph reports the reconciliation in ITS own terms -- a global id,
      // a union rank -- because that is all it is given. Which exchange, and
      // what each side actually offered, is knowable only here, and without
      // it a failure says that some id is unowned while naming neither the
      // channel nor the side that should have owned it. Appended rather than
      // replacing: the id and union rank are what locate the entity.
      const auto& src_c = graph.component(ComponentId{src_cid});
      const auto& dst_c = graph.component(ComponentId{dst_cid});
      const auto side_extent = [dim](bool mine, const MeshView* v) {
        if (!mine) return std::string{"not hosted on this rank"};
        if (v == nullptr) return std::string{"hosted but no view"};
        return std::to_string(v->owned_count(dim)) + " owned of " +
               std::to_string(v->entity_count(dim));
      };
      status = Failure{
          built.error().code,
          built.error().message + " [exchange: region '" + reg.name +
              "' on mesh '" + graph.mesh(reg_side.mesh).name + "', dimension " +
              std::to_string(dim) + "; source '" + src_c.name + "' offers " +
              side_extent(in_src, src_side.mesh) + "; destination '" +
              dst_c.name + "' offers " + side_extent(in_dst, dst_side.mesh) +
              "; this rank is " + std::to_string(my_rank) + " of the base "
              "communicator]"};
      break;
    }
    const CommGraph& g = built.value();

    // Which of the two transport plan forms this exchange needs. Identical or
    // disjoint placements can be described by one global_ids list per rank, so
    // they take the matched path, which is the verified one. Partially shared
    // placements cannot -- a rank is then a source for some entities and a
    // destination for others -- and take the peer-wise path (R31).
    const int nc_planned = p.spec.components;
    peer_plan_needed = !g.specs_are_disjoint();

    if (peer_plan_needed) {
      send_peers = g.send_plan();
      recv_peers = g.recv_plan();
      // prepare() compares only the extents for a peer-built plan, so those
      // are the only spec fields that must be right. With per-placement
      // decomposition a rank in both placements holds TWO blocks -- its source
      // component's and its destination component's -- whose sizes need not
      // agree, and the peer plan indexes each by its own local numbering. The
      // extent is therefore the LARGER of the two: every source index and
      // every destination index lies below it, which is what
      // build_plan_from_peers checks. The hot loop stages the source block
      // into `send` and copies only the written destination entries back.
      const std::size_t src_n =
          in_src ? src_view.value()->entity_count(dim) : 0;
      const std::size_t dst_n =
          in_dst ? dst_view.value()->entity_count(dim) : 0;
      p.spec.elements   = std::max(src_n, dst_n);
      p.spec.components = nc_planned;
      p.from_peers      = true;
      p.peer_src_total =
          src_n * static_cast<std::size_t>(std::max(nc_planned, 1));
      p.peer_written.clear();
      for (const CommGraphPeer& peer : recv_peers) {
        p.peer_written.insert(p.peer_written.end(), peer.local_indices.begin(),
                              peer.local_indices.end());
      }
      for (const LocalCopyPair& lp : g.local_pairs()) {
        p.peer_written.push_back(lp.dest_local_index);
      }
      std::sort(p.peer_written.begin(), p.peer_written.end());
      p.peer_written.erase(
          std::unique(p.peer_written.begin(), p.peer_written.end()),
          p.peer_written.end());
      p.spec.global_ids.clear();
      p.spec.src_ranks.clear();
      p.spec.dst_ranks.clear();
      // The peer path addresses the field directly by local element index, so
      // there is no separate gather or scatter permutation to apply.
      p.gather.clear();
      p.scatter.clear();
    } else {
      // Reset explicitly: initialize() is re-callable, and a plan that was a
      // peer plan last time must not keep the peer staging this time.
      p.from_peers     = false;
      p.peer_src_total = 0;
      p.peer_written.clear();
      // CommGraph knows nothing of components-per-entity; carry the arc's
      // count across the assignment.
      p.spec            = in_src ? g.source_spec() : g.dest_spec();
      p.spec.components = nc_planned;

      p.gather  = in_src ? g.source_indices() : std::vector<std::size_t>{};
      p.scatter = in_dst ? g.dest_indices() : std::vector<std::size_t>{};
    }

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

    // This plan's own Transport, built here rather than after the loop
    // because `peer_plan_needed`, `send_peers` and `recv_peers` describe THIS
    // plan and are overwritten by the next iteration. Building it after the
    // loop is what limited the runner to a single cross-rank channel.
    //
    // COLLECTIVE, and every rank reaches it for every plan: the registry
    // creates from `config.transport`, `transport_setup` duplicates the
    // communicator, and build_plan agrees its verdict across ranks. All three
    // depend on the plan and the config, which are identical everywhere.
    Result<std::unique_ptr<Transport>> made =
        TransportRegistry::instance().create(config.transport);
    if (!made.ok()) {
      status = made.error();
      continue;
    }
    std::unique_ptr<Transport> own = std::move(made.value());
    if (config.transport_setup) {
      const Result<void> set_up = config.transport_setup(*own);
      if (!set_up.ok()) {
        status = set_up;
        continue;
      }
    }

    auto* mpi = dynamic_cast<MpiTransport*>(own.get());
    if (mpi == nullptr) {
      // A CommGraph-backed plan carries subset gather/scatter indices that
      // only MpiTransport::build_plan() knows how to turn into a real
      // cross-rank exchange -- a "copy" or other backend given this spec
      // would silently move the wrong data rather than fail. `has_graph`'s
      // own doc comment (runner_impl.hpp) already excludes this case; this is
      // that exclusion made a refusal instead of an assumption.
      status = Failure{"E-RT-TRANSPORT",
                     "a CommGraph-backed Transport plan requires the \"mpi\" "
                     "transport backend, but the configured transport is '" +
                         own->name() +
                         "'; set RunnerConfig::transport = \"mpi\" alongside "
                         "base_communicator"};
      continue;
    }

    if (peer_plan_needed) {
      std::vector<LocalCopyPair> local;
      local.reserve(p.local_src.size());
      for (std::size_t k = 0; k < p.local_src.size(); ++k) {
        local.push_back({p.local_src[k], p.local_dst[k]});
      }
      status = mpi->build_plan_from_peers(send_peers, recv_peers, local,
                                         p.spec.elements, p.spec.components);
    } else {
      status = mpi->build_plan(p.spec);
    }
    if (!status.ok()) continue;
    graph_transports[pi] = std::move(own);
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
  // A5: `config.execution_space` and `DeviceExecSpace` are resolved purely
  // from compile-time macros and this rank's own config struct, so every
  // rank of a job built from the same binary reaches the same verdict here
  // -- this check needs no collective agreement, unlike the resolution
  // checkpoint below.
  if (config.execution_space == RunnerConfig::RunnerExecutionSpace::Device &&
      std::is_same_v<DeviceExecSpace, HostExecSpace>) {
    return Failure{"E-RT-DEVICE",
                 "RunnerConfig::execution_space requested Device, but this "
                 "build's Kokkos has no CUDA/HIP/SYCL backend compiled in "
                 "(DeviceExecSpace falls back to HostExecSpace); configure "
                 "against a device-enabled Kokkos or leave execution_space "
                 "at its Host default"};
  }

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
  {
    Result<void> ext = check_transport_extents();
#if FUSE_HAVE_MPI
    // Agreed like the checkpoint above: the verdict is rank-local (it reads
    // this rank's buffers and plans), and a rank returning alone while its
    // peers go on would leave them in the next collective.
    if (config.base_communicator.has_value()) {
      const MPI_Comm base =
          MPI_Comm_f2c(static_cast<MPI_Fint>(*config.base_communicator));
      const int local_ok = ext.ok() ? 1 : 0;
      int       all_ok   = 0;
      if (MPI_Allreduce(&local_ok, &all_ok, 1, MPI_INT, MPI_MIN, base) !=
          MPI_SUCCESS) {
        return Failure{"E-RT-TRANSPORT-COMM",
                       "MPI_Allreduce failed while agreeing Transport extents "
                       "across the base communicator"};
      }
      if (all_ok == 0 && ext.ok()) {
        return Failure{"E-RT-SETUP-PEER",
                       "another rank's Transport buffers do not fit its plan; "
                       "see that rank's E-RT-SIZE for the cause"};
      }
    }
#endif
    if (!ext.ok()) return ext;
  }
  r = build_ticks(components);
  if (!r.ok()) return r;

  // A5: the device staging lanes (`Impl::device_staging`'s own comment has
  // the sizing argument), sized once here like `reduction_scratch` below and
  // for the same reason; grown, not shrunk, on a repeated initialize().
  // BEFORE resolve_regions(): a window-scoped region's lagged import chain
  // is executed once during region resolution (iteration.cpp, "chain"), and
  // with zero-length lanes that Interpolate reported E-RT-DEVICE
  // (verification.brusselator_runner.device, the first run of the twin).
  if (config.execution_space == RunnerConfig::RunnerExecutionSpace::Device) {
    std::size_t stride = 0;
    for (const Slot& s : slots) stride = std::max(stride, s.total());
    device_staging_stride = stride;
    if (block_host_scratch.size() < stride) block_host_scratch.resize(stride);
    const std::size_t needed = kDeviceStagingSlots * stride;
    if (device_staging.extent(0) < needed) {
      device_staging = Kokkos::View<double*, DeviceMemSpace>(
          Kokkos::view_alloc("fuse::dev_staging", Kokkos::WithoutInitializing),
          needed);
    }
  }

  // Last: resolve_regions() needs the ticks and the resolved buffers, and it
  // is the single place spec §5.3's preconditions are checked
  // (E-ITER-NOCONVERGE, E-ITER-MAXITER, E-ITER-NOROLLBACK).
  r = resolve_regions(components, registry);
  if (!r.ok()) return r;

  // docs/dev/device-resident-arcs.md, step 1: with every consumer of every
  // arc now known (instructions, regions, windows, constrains), move the
  // arcs that only Space-generic kernels touch into device memory.
  if (config.execution_space == RunnerConfig::RunnerExecutionSpace::Device) {
    place_arcs_on_device();
    // Phase 3: now that every Transport instruction's arcs have their
    // final placement, mirror the ones that landed fully on the device.
    build_transport_device_buffers();
  }

  // KNOWN_ISSUES.md D11 (closing): scratch for the fixed-chunk deterministic
  // reductions run_region() uses (fuse/num/space_kernels.hpp's
  // `deterministic_norm_squared`/`deterministic_norm_squared_diff`), sized
  // ONCE here from the largest block any region has -- never per sweep or
  // per period, per iteration.cpp's own header comment (point 4: nothing in
  // the sweep loop may allocate). Grown, not shrunk, on a repeated
  // initialize() call (Runner::initialize() is re-callable), so a second
  // call with a smaller plan does not lose headroom a still-live reference
  // might need -- though nothing currently keeps one across a call.
  std::size_t max_block_length = 0;
  for (const detail::RegionRuntime& region : regions) {
    for (const detail::RegionBlock& b : region.blocks) {
      max_block_length = std::max(max_block_length, b.length);
    }
  }
  const std::size_t needed_chunks = num::deterministic_chunks(max_block_length);
  if (reduction_scratch.extent(0) < needed_chunks) {
    reduction_scratch = Kokkos::View<double*, Kokkos::HostSpace>(
        Kokkos::view_alloc("fuse::reduction_scratch",
                          Kokkos::WithoutInitializing),
        needed_chunks);
  }

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

/// R-10 (REVIEW.md): folds `peer`'s contribution into `acc` in place, one
/// block at a time, via `num::combine()` -- the same undefined-count
/// exclusion `fuse::num::field_norm()`'s callers get for free, so a rank that
/// wrote `owned == 0` (no local entries, or a locally-failed sweep; see
/// reduce_region_residuals()) never reads as a converged zero. `status` is
/// combined by MIN, matching reconcile_epoch()'s status arithmetic: one
/// failed participant makes the combined status failed for everyone.
///
/// Euclidean is the kind used throughout: it is what run_region()'s
/// existing per-rank computation already was (REVIEW.md R-10's fix keeps the
/// serial iteration counts the recorded test fixtures were built with; see
/// docs -- num/residual_norm.hpp's own header comment says the same about
/// MeasureWeightedRms being a deliberate, opt-in departure). `combine()`
/// cannot fail for Euclidean (the E-NORM-UNITS check is
/// MeasureWeightedRms-only, and Euclidean's NormValue::units is always the
/// default), so the Result is unwrapped unconditionally.
void merge_region_payload(std::vector<double>& acc,
                          const std::vector<double>& peer,
                          std::size_t nblocks) {
  acc[0] = std::min(acc[0], peer[0]);
  for (std::size_t bi = 0; bi < nblocks; ++bi) {
    const std::size_t base = 1 + 4 * bi;
    for (const std::size_t off : {std::size_t{0}, std::size_t{2}}) {
      num::NormValue a;
      a.value = acc[base + off];
      a.owned_count = static_cast<std::size_t>(acc[base + off + 1]);
      a.total_measure = static_cast<double>(a.owned_count);
      num::NormValue b;
      b.value = peer[base + off];
      b.owned_count = static_cast<std::size_t>(peer[base + off + 1]);
      b.total_measure = static_cast<double>(b.owned_count);
      const num::NormValue c =
          num::combine(num::NormKind::Euclidean, {a, b}).value();
      acc[base + off]     = c.value;
      acc[base + off + 1] = static_cast<double>(c.owned_count);
    }
  }
}

}  // namespace

/// The tag the agreement reduction uses on the base communicator.
/// Defined with every other tag in fuse/transport/tags.hpp, which explains why
/// they have to be chosen together.
constexpr int kEpochTag = tags::kEpoch;

/// R-10 (REVIEW.md): the tag reduce_region_residuals() uses on the same base
/// communicator. Distinct from kEpochTag for the reason tags.hpp's own
/// comment on kIterationRegion gives.
constexpr int kRegionTag = tags::kIterationRegion;

Result<void> Runner::Impl::resolve_epoch_participants() {
#if FUSE_HAVE_MPI
  if (!config.base_communicator.has_value()) return Result<void>::success();

  // Checked once, here, rather than assumed: the guaranteed floor is 32767 and
  // `kEpochTag` is below it, so this cannot fire on a conforming MPI -- but a
  // refusal at setup is a better failure than "Invalid tag" from inside a
  // reduction on one rank while the others wait.
  //
  // R-32 (REVIEW.md): queried against `base`, the same communicator
  // reconcile_epoch() actually sends and receives on, not MPI_COMM_WORLD.
  // `MPI_TAG_UB` is an attribute of a communicator and nothing here
  // guarantees the two share a value -- only that they usually do, under the
  // spec §6 mode 1 usage `RunnerConfig::base_communicator` documents as
  // today's only integration mode. A caller that ever hands this something
  // other than MPI_COMM_WORLD itself would have this check silently pass on
  // MPI_COMM_WORLD's bound while reconcile_epoch()'s own communicator has a
  // lower one -- reproducing exactly the "Invalid tag" failure inside a
  // reduction, on one rank while its partners wait, that this check exists
  // to prevent.
  {
    const MPI_Comm base =
        MPI_Comm_f2c(static_cast<MPI_Fint>(*config.base_communicator));
    if (base == MPI_COMM_NULL) {
      return Failure{"E-RT-EPOCH-COMM",
                     "RunnerConfig::base_communicator is MPI_COMM_NULL"};
    }
    void* ub = nullptr;
    int   found = 0;
    if (MPI_Comm_get_attr(base, MPI_TAG_UB, &ub, &found) == MPI_SUCCESS &&
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
    // The union of the two placements. Iterated as a list, not a range: a
    // component may be placed on strided ranks, and the two may share some.
    // A shared rank appears once, which is what makes it a participant once.
    std::vector<int> ranks;
    for (const int component : {ins.attrs.src_rank_set, ins.attrs.dst_rank_set}) {
      if (component < 0) continue;
      const auto id = static_cast<std::uint32_t>(component);
      if (id >= graph.component_count()) continue;
      for (const int r : graph.component(ComponentId{id}).ranks) ranks.push_back(r);
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
    // This rank is fine and a partner is not. RECORD the failure; do not
    // return it. Every participant must leave this agreement point in the
    // same state: the failing rank already holds a step_failure and walks on
    // through the rest of the stream, executing only collective operators
    // (step_period/run_region's failed mode), so a rank that returned here
    // instead would leave its partners waiting in the next Transport. That
    // was the hang behind RunnerTransport.StillRunsASecondChannelAfterA
    // RegionFailsUnderMpi once each component is stepped only on its own
    // processes (3aae2ed): before it, every process stepped the failing
    // component, so every process failed and none returned. Recording it
    // puts this rank in the same failed mode, which is also what the
    // implicit coupling group's residual agreement does (E-RT-REGION-PEER).
    // The period still fails on every rank at its end; the failing rank
    // reports its own cause, which is the diagnostic worth reading.
    step_failure = Failure{"E-RT-EPOCH-PEER",
                           "a rank taking part in this exchange failed; this "
                           "rank ran only the collective operations for the "
                           "rest of the period. The failing rank reports the "
                           "cause."};
  }
#else
  (void)ins;
#endif
  return Result<void>::success();
}

Result<void> Runner::Impl::reduce_region_residuals(
    const detail::RegionRuntime& r, std::vector<double>& payload) {
#if FUSE_HAVE_MPI
  if (r.participants.size() <= 1) return Result<void>::success();
  const MPI_Comm base =
      MPI_Comm_f2c(static_cast<MPI_Fint>(*config.base_communicator));
  if (base == MPI_COMM_NULL) {
    return Failure{"E-RT-REGION-COMM",
                   "RunnerConfig::base_communicator is MPI_COMM_NULL at "
                   "iteration region '" + r.name + "'s residual agreement"};
  }
  int me = 0;
  if (MPI_Comm_rank(base, &me) != MPI_SUCCESS) {
    return Failure{"E-RT-REGION-COMM",
                   "MPI_Comm_rank failed at iteration region '" + r.name +
                       "'s residual agreement"};
  }
  const std::vector<int>& part = r.participants;
  const auto self = std::find(part.begin(), part.end(), me);
  if (self == part.end()) {
    // Not a participant of this region -- see RegionRuntime::participants.
    // Joining would make the real participants wait for a rank with nothing
    // to contribute, exactly as reconcile_epoch() reasons for an Epoch.
    return Result<void>::success();
  }
  const auto n = static_cast<int>(part.size());
  const auto my_index = static_cast<int>(std::distance(part.begin(), self));
  const auto count = static_cast<int>(payload.size());
  const EpochSchedule sched = epoch_schedule(n, my_index);
  const std::size_t nblocks = r.blocks.size();

  std::vector<double> peer(payload.size());
  for (int ri = 0; ri < sched.reduce_recv_count; ++ri) {
    const int from = sched.reduce_recv_from[static_cast<std::size_t>(ri)];
    if (MPI_Recv(peer.data(), count, MPI_DOUBLE,
                 part[static_cast<std::size_t>(from)], kRegionTag, base,
                 MPI_STATUS_IGNORE) != MPI_SUCCESS) {
      return Failure{"E-RT-REGION-COMM",
                     "MPI_Recv failed while reducing iteration region '" +
                         r.name + "'s residuals"};
    }
    merge_region_payload(payload, peer, nblocks);
  }
  if (sched.reduce_send_to >= 0) {
    if (MPI_Send(payload.data(), count, MPI_DOUBLE,
                 part[static_cast<std::size_t>(sched.reduce_send_to)],
                 kRegionTag, base) != MPI_SUCCESS) {
      return Failure{"E-RT-REGION-COMM",
                     "MPI_Send failed while reducing iteration region '" +
                         r.name + "'s residuals"};
    }
  }
  if (sched.result_recv_from >= 0) {
    if (MPI_Recv(payload.data(), count, MPI_DOUBLE,
                 part[static_cast<std::size_t>(sched.result_recv_from)],
                 kRegionTag, base, MPI_STATUS_IGNORE) != MPI_SUCCESS) {
      return Failure{"E-RT-REGION-COMM",
                     "MPI_Recv failed while receiving iteration region '" +
                         r.name + "'s agreed residual"};
    }
  }
  for (int to = 1; to <= sched.result_send_count; ++to) {
    if (MPI_Send(payload.data(), count, MPI_DOUBLE,
                 part[static_cast<std::size_t>(to)], kRegionTag,
                 base) != MPI_SUCCESS) {
      return Failure{"E-RT-REGION-COMM",
                     "MPI_Send failed while publishing iteration region '" +
                         r.name + "'s agreed residual"};
    }
  }
#else
  (void)r;
  (void)payload;
#endif
  return Result<void>::success();
}

Result<void> Runner::Impl::gather_window_block_samples(
    const detail::RegionRuntime& r, bool local_ok,
    const std::vector<WindowBlockSample>& mine,
    std::vector<std::vector<WindowBlockSample>>& gathered_out,
    bool& all_ok_out) {
  if (r.participants.size() <= 1) {
    gathered_out.assign(1, mine);
    all_ok_out = local_ok;
    return Result<void>::success();
  }
#if FUSE_HAVE_MPI
  const MPI_Comm base =
      MPI_Comm_f2c(static_cast<MPI_Fint>(*config.base_communicator));
  if (base == MPI_COMM_NULL) {
    return Failure{"E-RT-WINDOW-COMM",
                   "RunnerConfig::base_communicator is MPI_COMM_NULL at "
                   "iteration region '" + r.name +
                       "'s window convergence agreement"};
  }
  int me = 0;
  if (MPI_Comm_rank(base, &me) != MPI_SUCCESS) {
    return Failure{"E-RT-WINDOW-COMM",
                   "MPI_Comm_rank failed at iteration region '" + r.name +
                       "'s window convergence agreement"};
  }
  const std::vector<int>& part = r.participants;
  const auto self = std::find(part.begin(), part.end(), me);
  if (self == part.end()) {
    // Not a participant -- mirrors reduce_region_residuals()'s own early
    // return. Nothing to contribute, nothing to wait for.
    gathered_out.clear();
    all_ok_out = true;
    return Result<void>::success();
  }
  const auto n = static_cast<int>(part.size());
  const auto my_index = static_cast<int>(std::distance(part.begin(), self));
  const std::size_t nb = mine.size();
  const std::size_t width = 1 + 3 * nb;

  std::vector<double> mine_flat(width);
  mine_flat[0] = local_ok ? 1.0 : 0.0;
  for (std::size_t bi = 0; bi < nb; ++bi) {
    mine_flat[1 + 3 * bi + 0] = mine[bi].residual_sq;
    mine_flat[1 + 3 * bi + 1] = mine[bi].reference_sq;
    mine_flat[1 + 3 * bi + 2] = mine[bi].owned ? 1.0 : 0.0;
  }

  // All-to-all over `part`, point-to-point, exactly as reconcile_epoch()/
  // reduce_region_residuals() address peers over this SAME base
  // communicator -- never a sub-communicator built just for this (R.
  // participants is a subset of `base`'s own ranks; a TRUE MPI_Allgather
  // would need every rank of `base` to call it, which a non-participant
  // never does). Non-blocking sends so no participant can deadlock
  // waiting on a send before its own partners have posted their receives.
  std::vector<double> all_flat(static_cast<std::size_t>(n) * width);
  std::copy(mine_flat.begin(), mine_flat.end(),
            all_flat.begin() +
                static_cast<std::size_t>(my_index) * width);

  std::vector<MPI_Request> sends;
  sends.reserve(static_cast<std::size_t>(n > 0 ? n - 1 : 0));
  for (int j = 0; j < n; ++j) {
    if (j == my_index) continue;
    MPI_Request req;
    if (MPI_Isend(mine_flat.data(), static_cast<int>(width), MPI_DOUBLE,
                 part[static_cast<std::size_t>(j)], tags::kWindowConvergence,
                 base, &req) != MPI_SUCCESS) {
      return Failure{"E-RT-WINDOW-COMM",
                   "MPI_Isend failed while gathering iteration region '" +
                       r.name + "'s window convergence samples"};
    }
    sends.push_back(req);
  }
  for (int j = 0; j < n; ++j) {
    if (j == my_index) continue;
    if (MPI_Recv(all_flat.data() + static_cast<std::size_t>(j) * width,
                 static_cast<int>(width), MPI_DOUBLE,
                 part[static_cast<std::size_t>(j)], tags::kWindowConvergence,
                 base, MPI_STATUS_IGNORE) != MPI_SUCCESS) {
      return Failure{"E-RT-WINDOW-COMM",
                   "MPI_Recv failed while gathering iteration region '" +
                       r.name + "'s window convergence samples"};
    }
  }
  if (!sends.empty() &&
      MPI_Waitall(static_cast<int>(sends.size()), sends.data(),
                  MPI_STATUSES_IGNORE) != MPI_SUCCESS) {
    return Failure{"E-RT-WINDOW-COMM",
                 "MPI_Waitall failed while gathering iteration region '" +
                     r.name + "'s window convergence samples"};
  }

  gathered_out.assign(static_cast<std::size_t>(n), std::vector<WindowBlockSample>(nb));
  all_ok_out = true;
  for (int ri = 0; ri < n; ++ri) {
    const double* row = all_flat.data() + static_cast<std::size_t>(ri) * width;
    if (row[0] < 0.5) all_ok_out = false;
    for (std::size_t bi = 0; bi < nb; ++bi) {
      WindowBlockSample& s = gathered_out[static_cast<std::size_t>(ri)][bi];
      s.residual_sq = row[1 + 3 * bi + 0];
      s.reference_sq = row[1 + 3 * bi + 1];
      s.owned = row[1 + 3 * bi + 2] != 0.0;
    }
  }
#else
  (void)r;
  (void)local_ok;
  (void)mine;
  gathered_out.assign(1, mine);
  all_ok_out = local_ok;
#endif
  return Result<void>::success();
}

Result<bool> Runner::Impl::region_is_local_participant(const detail::RegionRuntime& r) const {
  if (r.participants.size() <= 1) return true;
#if FUSE_HAVE_MPI
  if (!config.base_communicator.has_value()) {
    return Failure{"E-RT-REGION-COMM",
                   "iteration region '" + r.name + "' has more than one "
                       "participant but RunnerConfig::base_communicator is "
                       "unset"};
  }
  const MPI_Comm base = MPI_Comm_f2c(static_cast<MPI_Fint>(*config.base_communicator));
  if (base == MPI_COMM_NULL) {
    return Failure{"E-RT-REGION-COMM",
                   "RunnerConfig::base_communicator is MPI_COMM_NULL at "
                       "iteration region '" + r.name + "'s participant check"};
  }
  int me = 0;
  if (MPI_Comm_rank(base, &me) != MPI_SUCCESS) {
    return Failure{"E-RT-REGION-COMM",
                   "MPI_Comm_rank failed at iteration region '" + r.name +
                       "'s participant check"};
  }
  return std::find(r.participants.begin(), r.participants.end(), me) !=
         r.participants.end();
#else
  return Failure{"E-RT-TRANSPORT-NOMPI",
                 "iteration region '" + r.name + "' has more than one "
                     "participant in a build without MPI"};
#endif
}

Result<void> Runner::Impl::broadcast_window_convergence(
    const detail::RegionRuntime& r, bool converged, bool& converged_out) {
  converged_out = converged;
  if (r.participants.size() <= 1) return Result<void>::success();
#if FUSE_HAVE_MPI
  const MPI_Comm base =
      MPI_Comm_f2c(static_cast<MPI_Fint>(*config.base_communicator));
  if (base == MPI_COMM_NULL) {
    return Failure{"E-RT-WINDOW-COMM",
                   "RunnerConfig::base_communicator is MPI_COMM_NULL at "
                   "iteration region '" + r.name +
                       "'s window convergence broadcast"};
  }
  int conv = converged ? 1 : 0;
  if (MPI_Bcast(&conv, 1, MPI_INT, r.participants.front(), base) !=
      MPI_SUCCESS) {
    return Failure{"E-RT-WINDOW-COMM",
                 "MPI_Bcast failed while broadcasting iteration region '" +
                     r.name + "'s window convergence decision"};
  }
  converged_out = conv != 0;
#else
  (void)converged;
#endif
  return Result<void>::success();
}

Result<void> Runner::Impl::execute(const Instruction& ins, const Resolved& r) {
  // A5: `config.execution_space` picks the instantiation; the body is
  // `execute_on<Space>` below, identical for both except where a kernel
  // launches and, on a non-host-accessible `Space`, how its operands get
  // there (the staging helpers at the top of that body).
  if (config.execution_space == RunnerConfig::RunnerExecutionSpace::Device) {
    return execute_on<DeviceExecSpace>(ins, r);
  }
  return execute_on<HostExecSpace>(ins, r);
}

template <class Space>
Result<void> Runner::Impl::execute_on(const Instruction& ins, const Resolved& r) {
  const auto in_ptr = [&](std::uint8_t k) -> const double* {
    return slots[ins.in[k].value()].data;
  };
  const auto out_ptr = [&]() -> double* { return slots[ins.out.value()].data; };
  const std::size_t n = r.out_total;

  // KNOWN_ISSUES.md D11 (closing) / matrix cell A-SPACES-TRANSPORT: the
  // elementwise operators below wrap their existing raw pointers as
  // UNMANAGED views (fuse/num/space_kernels.hpp's SpanOf/ConstSpanOf are
  // zero-cost for exactly this) and run on `Space` -- `HostExecSpace`
  // (OpenMP if the build enabled it, Serial otherwise) for every caller
  // that leaves `RunnerConfig::execution_space` at its default -- instead
  // of the hand-written serial loops this switch used to run directly.
  // Each output element's write depends on no other, so the answer cannot
  // depend on how many threads Kokkos happened to split the range across;
  // see space_kernels.hpp's own header comment. Nothing here allocates: a
  // SpanOf/ConstSpanOf is a view CONSTRUCTOR over an existing pointer, not a
  // new buffer.
  //
  // Only the cases that call `num::xxx<Space>(span_c(...), span_m(...))`
  // below are Space-generic. Source/ExternalDelivery (no-ops), Sink/
  // Accelerate/Epoch/Constrain/Diagnose (plain `num::copy(ptr, ptr, n)` plus
  // host-side bookkeeping) and the WindowAdvance/Map/Reduce/Closure
  // refusals are host-only whatever `Space` is, and read their raw
  // pointers directly. `Transport` is NOT in that list as of
  // docs/dev/device-resident-arcs.md Phase 3: its own case below reads
  // `Impl::transport_device` and `Transport::is_gpu_aware()` to decide,
  // per instruction, whether to pack/unpack on the device (handing MPI a
  // device pointer when the backend is GPU-aware, staging through host
  // memory when it is not) or to fall back to this same host pointer
  // arithmetic unchanged -- it is the one case here that is space-AWARE
  // rather than either generic or unconditionally host-only. Remap's
  // distributed-nodal branch (`r.nodal_dist >= 0`,
  // apply_nodal_dist_remap(): an MPI exchange over host buffers) remains
  // unconditionally host-only; it is a different track's scope.
  //
  // A5, the non-host-accessible `Space` (Kokkos::Cuda here): the same two
  // helpers instead hand back views over `device_staging` lanes
  // (`Impl::device_staging`'s own comment). `stage()` keys lanes by HOST
  // pointer, so an in-place call such as Accumulate's
  // `scale(inv, span_c(out, n), span_m(out, n))` aliases ONE lane exactly as
  // the host instantiation aliases one buffer, and a buffer read by one
  // kernel and written by the next in the same instruction (Relax's `prev`)
  // is copied in once and back out once, after the switch. A pointer's lane
  // is widened in place if a later call asks for more of it; the only
  // faults -- more distinct pointers than lanes, or a longer run than the
  // stride -- are impossible by setup()'s sizing and are reported as
  // `E-RT-DEVICE` after the switch rather than silently clamped.
  constexpr bool kHostAccessible = Kokkos::SpaceAccessibility<
      Kokkos::HostSpace, typename Space::memory_space>::accessible;
  struct Staged {
    const double* host = nullptr;
    double*       host_mut = nullptr;  // non-null once some kernel writes it
    std::size_t   len = 0;
  };
  std::array<Staged, kDeviceStagingSlots> staged{};
  std::size_t n_staged = 0;
  bool        staging_fault = false;
  // Device-resident arcs (docs/dev/device-resident-arcs.md): a pointer that
  // IS this instruction's input or output slot and is flagged `device` is
  // wrapped directly, no lane. Identity, not range lookup: every call site
  // below passes in_ptr(k)/out_ptr() (or `out`, the same pointer) for arcs,
  // and `previous[]`/`history` host buffers otherwise.
  [[maybe_unused]] bool       used_device_arc = false;
  // The slot a pointer belongs to, if it is one of this instruction's arcs.
  [[maybe_unused]] const auto slot_of = [&](const double* p) -> const Slot* {
    if (ins.out.valid() && p == slots[ins.out.value()].data) return &slots[ins.out.value()];
    for (std::uint8_t k = 0; k < ins.n_in; ++k) {
      if (ins.in[k].valid() && p == slots[ins.in[k].value()].data) {
        return &slots[ins.in[k].value()];
      }
    }
    return nullptr;
  };
  // Where a kernel READS an arc: its own device storage, or the device
  // mirror of a host-registered field (refreshed here if component code
  // ran since its last copy -- Impl::component_epoch), or nullptr when it
  // has to be staged (a host arc, `previous[]`, history).
  [[maybe_unused]] const auto device_read_ptr = [&](const double* p) -> const double* {
    const Slot* s = slot_of(p);
    if (s == nullptr) return nullptr;
    if (s->device) return p;
    const std::size_t idx = static_cast<std::size_t>(s - slots.data());
    if (device_mirror[idx].extent(0) < s->total()) return nullptr;
    if constexpr (!kHostAccessible) {
      if (device_mirror_epoch[idx] != component_epoch) {
        Kokkos::deep_copy(num::SpanOf<Space>(device_mirror[idx].data(), s->total()),
                          num::ConstSpanOf<HostExecSpace>(s->data, s->total()));
        device_mirror_epoch[idx] = component_epoch;
        ++stats.device_boundary_copies;
      }
    }
    return device_mirror[idx].data();
  };
  [[maybe_unused]] const auto is_device_arc = [&](const double* p) -> bool {
    const Slot* s = slot_of(p);
    return s != nullptr && s->device;
  };
  // A copy between two arcs of any spaces: the host-only cases that move a
  // value on (Sink, Accelerate, Epoch) stay correct whichever side a field
  // or arc lives on. Same-space copies are what they always were.
  [[maybe_unused]] const auto copy_arc = [&](const double* src, double* dst,
                                             std::size_t len) {
    if constexpr (kHostAccessible) {
      num::copy(src, dst, len);
    } else {
      const bool sd = is_device_arc(src), dd = is_device_arc(dst);
      if (!sd && !dd) { num::copy(src, dst, len); return; }
      if (sd && dd) {
        num::copy<Space>(num::ConstSpanOf<Space>(src, len), num::SpanOf<Space>(dst, len));
        return;
      }
      ++stats.device_boundary_copies;
      if (sd) {
        Kokkos::deep_copy(num::SpanOf<HostExecSpace>(dst, len),
                          num::ConstSpanOf<Space>(src, len));
      } else {
        Kokkos::deep_copy(num::SpanOf<Space>(dst, len),
                          num::ConstSpanOf<HostExecSpace>(src, len));
      }
    }
  };
  const auto  stage = [&](const double* p, std::size_t len,
                          double* writable) -> double* {
    double* const base = device_staging.data();
    for (std::size_t i = 0; i < n_staged; ++i) {
      if (staged[i].host != p) continue;
      double* const lane = base + i * device_staging_stride;
      if (len > staged[i].len) {
        if (len > device_staging_stride) {
          staging_fault = true;
          return base;
        }
        if (staged[i].host_mut != nullptr) {
          Kokkos::deep_copy(
              num::SpanOf<HostExecSpace>(staged[i].host_mut, staged[i].len),
              num::ConstSpanOf<Space>(lane, staged[i].len));
          ++stats.device_staged_copies;
        }
        Kokkos::deep_copy(num::SpanOf<Space>(lane, len),
                          num::ConstSpanOf<HostExecSpace>(p, len));
        ++stats.device_staged_copies;
        staged[i].len = len;
      }
      if (writable != nullptr) staged[i].host_mut = writable;
      return lane;
    }
    if (n_staged == kDeviceStagingSlots || len > device_staging_stride) {
      staging_fault = true;
      return base;
    }
    double* const lane = base + n_staged * device_staging_stride;
    Kokkos::deep_copy(num::SpanOf<Space>(lane, len),
                      num::ConstSpanOf<HostExecSpace>(p, len));
    ++stats.device_staged_copies;
    staged[n_staged++] = Staged{p, writable, len};
    return lane;
  };
  // After a staging fault the lane handed back is lane 0 and `len` may exceed
  // the stride (or the whole buffer), so every view built after the fault is
  // zero-length: the kernels between here and the post-switch `E-RT-DEVICE`
  // report become no-ops instead of writing past the lane. Nothing is
  // written back after a fault, so a no-op kernel costs nothing observable.
  const auto span_c = [&](const double* p, std::size_t len) {
    if constexpr (kHostAccessible) {
      return num::ConstSpanOf<Space>(p, len);
    } else {
      if (const double* d = device_read_ptr(p); d != nullptr) {
        used_device_arc = true;
        return num::ConstSpanOf<Space>(d, len);
      }
      double* const lane = stage(p, len, nullptr);
      return num::ConstSpanOf<Space>(lane, staging_fault ? 0 : len);
    }
  };
  const auto span_m = [&](double* p, std::size_t len) {
    if constexpr (kHostAccessible) {
      return num::SpanOf<Space>(p, len);
    } else {
      if (is_device_arc(p)) {
        used_device_arc = true;
        return num::SpanOf<Space>(p, len);
      }
      double* const lane = stage(p, len, p);
      return num::SpanOf<Space>(lane, staging_fault ? 0 : len);
    }
  };

  switch (ins.kind) {
    case OpKind::Source:
      // The arc already aliases component storage; nothing to do. In Device
      // mode a host-registered field that a kernel reads has a device mirror
      // (Impl::device_mirror) that the READ refreshes -- not this
      // instruction: Source is stamped BeforeTicks, and the `Sample` that
      // reads the field runs AfterTicks, after the producer advanced again.
      break;

    case OpKind::ExternalDelivery:
      // The application (or, for the tier-0 oracle, the test harness
      // standing in for it) fills this arc's buffer directly -- by the
      // point an H5Dread or its stand-in would occur -- before this
      // instruction runs; nothing to do here. `ins.in[0]` (lower_channel(),
      // src/ir/rules_channel.cpp) exists ONLY to give the scheduler the
      // same ordering constraint `Transport`'s own input gives it for an
      // ordinary channel -- deliberately never read here. See
      // OpKind::ExternalDelivery (include/fuse/ir/op.hpp) and
      // docs/user/lowfive.md, Unit 3. The application wrote the buffer, so
      // a device mirror of it is stale (Impl::component_epoch).
      note_component_ran();
      break;

    case OpKind::Sink:
    case OpKind::Accelerate: {  // Phase 7 replaces this with a real accelerator.
      if (ins.n_in == 0 || !ins.out.valid()) break;
      copy_arc(in_ptr(0), out_ptr(), n);
      break;
    }

    case OpKind::Interpolate: {
      if (ins.n_in == 0 || !ins.out.valid()) break;
      if (r.spacetime < 0) {
        // The `lag(k)` no-op of src/ir/rules_channel.cpp: it exists so the
        // TimeLevel shift has an arc to live on and moves no data.
        num::copy<Space>(span_c(in_ptr(0), n), span_m(out_ptr(), n));
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
      if (r.nodal_dist >= 0) {
        // Nodal-remap track, round 2 (KNOWN_ISSUES.md D8): a distributed
        // nodal Remap target via "consistent"/"consistent-transpose" has
        // its OWN operator (build_nodal_dist_remap() at setup); this is the
        // WHOLE answer, not a local Remap answer plus a correction, so it
        // takes over the case entirely -- `r.remap` is left at -1 for such
        // an instruction (see Resolved::nodal_dist's own comment). Host-only
        // whatever `Space` is: the operator is an MPI exchange over host
        // buffers, so this branch never stages and never counts as a
        // device dispatch (A5).
        const Result<void> nd = apply_nodal_dist_remap(ins, r);
        if (!nd.ok()) return nd;
        break;
      }
      if (r.remap < 0) {
        // The identity -- and it IS the conservative map here, not a stand-in
        // for one. resolve_remap() reached this branch only after checking
        // every cell of both regions for the same measure and the same
        // centroid at the same local index, so `dst[i] = src[i]` moves each
        // value to the cell it already belonged to. Any pair that failed that
        // check either carries real weights below or never got past setup.
        num::copy<Space>(span_c(in_ptr(0), n), span_m(out_ptr(), n));
      } else {
        // Spec §5.4: `g = W f`, with `W` the conservative FV<->FV matrix
        // built from the supermesh at SETUP (src/remap/intx_weights.cpp).
        // Nothing is allocated, looked up or rebuilt here -- `remap_weights`
        // AND `remap_csr` were both filled in resolve_remap()
        // (build_remap_csr(), KNOWN_ISSUES.md D11 closing), and the CSR
        // conversion, like the triplets themselves, is owned by setup, not
        // rebuilt per period. `remap_apply`'s multi-component overload
        // (fuse/num/space_kernels.hpp) reproduces `apply_weights`' own
        // per-row accumulation order exactly -- see detail::RemapCsr's own
        // comment for why the stable sort at plan time guarantees that.
        const auto ri = static_cast<std::size_t>(r.remap);
        const RemapWeights& w = *remap_weights[ri];
        const int components = slots[ins.out.value()].components;
        if (components < 1) {
          return Failure{"E-REMAP-ARG",
                       "apply_weights: components must be >= 1, not " +
                           std::to_string(components)};
        }
        const std::size_t nc = static_cast<std::size_t>(components);
        // The CSR already lives in `Space`'s memory: `remap_csr` for the
        // host, its `deep_copy` twin `remap_csr_device` otherwise (A5) --
        // nothing about the matrix is staged per instruction.
        num::IndexSpanOf<Space> row_begin;
        num::IndexSpanOf<Space> col;
        num::ConstSpanOf<Space> val;
        if constexpr (kHostAccessible) {
          const detail::RemapCsr& csr = remap_csr[ri];
          row_begin = num::IndexSpanOf<Space>(csr.row_begin.data(),
                                              csr.row_begin.extent(0));
          col = num::IndexSpanOf<Space>(csr.col.data(), csr.col.extent(0));
          val = num::ConstSpanOf<Space>(csr.val.data(), csr.val.extent(0));
        } else {
          if (ri >= remap_csr_device.size()) {
            return Failure{"E-RT-DEVICE",
                         "Remap instruction has no device-resident CSR; "
                         "resolve_remap() must build one for every weights "
                         "entry when execution_space is Device"};
          }
          const auto& csr = remap_csr_device[ri];
          row_begin = num::IndexSpanOf<Space>(csr.row_begin.data(),
                                              csr.row_begin.extent(0));
          col = num::IndexSpanOf<Space>(csr.col.data(), csr.col.extent(0));
          val = num::ConstSpanOf<Space>(csr.val.data(), csr.val.extent(0));
        }
        num::remap_apply<Space>(row_begin, col, val,
                                span_c(in_ptr(0), w.n_cols * nc),
                                span_m(out_ptr(), w.n_rows * nc), nc);
      }
      break;
    }

    case OpKind::Sample: {
      if (ins.n_in == 0 || !ins.out.valid()) break;
      if (r.spacetime >= 0) {
        // Spec §7.3: file this source step's value in the temporal history.
        // The arc still carries the latest sample so that the SSA value is
        // defined and inspectable; the projection reads `history`.
        SpacetimeInfo& st = spacetimes[static_cast<std::size_t>(r.spacetime)];
        const auto j = static_cast<std::size_t>(ins.src_step);
        num::copy<Space>(span_c(in_ptr(0), st.width),
                         span_m(st.history.data() + j * st.width, st.width));
        num::copy<Space>(span_c(in_ptr(0), n), span_m(out_ptr(), n));
        break;
      }
      const std::size_t idx = sample_count[ins.out.value()];
      // r.accum, not ins.attrs.accum: see resolve_instructions(). `first`
      // mirrors `num::accumulate_into`'s own `sample_index == 0` rule.
      num::accumulate<Space>(r.accum, span_c(in_ptr(0), n),
                             span_m(out_ptr(), n), idx == 0);
      // After a staging fault the kernel above was a no-op and the
      // instruction fails after the switch; do not record a sample that was
      // never accumulated. Always false in the host instantiation.
      if (!staging_fault) sample_count[ins.out.value()] = idx + 1;
      break;
    }

    case OpKind::Accumulate: {
      if (ins.n_in == 0 || !ins.out.valid()) break;
      double* out = out_ptr();
      num::copy<Space>(span_c(in_ptr(0), n), span_m(out, n));
      // The sample counter belongs to the accumulator arc this reduction
      // reads, not to the arc it writes. `num::accumulate_finalize`
      // (kernels.hpp) does nothing outside AccumOp::Mean with a nonzero
      // count; the elementwise equivalent is exactly that scale, in place.
      const std::size_t count = sample_count[ins.in[0].value()];
      if (ins.attrs.accum == AccumOp::Mean && count != 0) {
        const double inv = 1.0 / static_cast<double>(count);
        num::scale<Space>(inv, span_c(out, n), span_m(out, n));
      }
      sample_count[ins.in[0].value()] = 0;  // reset for the next window
      break;
    }

    case OpKind::Extrapolate: {
      if (ins.n_in == 0 || !ins.out.valid()) break;
      // Phase 4 keeps no history, so order 0 -- `out[i] = history[i]`,
      // exactly `copy` -- is the only behaviour `num::extrapolate` (still
      // used, and still the oracle for a future order > 0) ever actually
      // takes here.
      num::copy<Space>(span_c(in_ptr(0), n), span_m(out_ptr(), n));
      break;
    }

    case OpKind::Merge: {
      if (!ins.out.valid() || ins.n_in == 0) break;
      double* out = out_ptr();
      switch (ins.attrs.merge) {
        case MergeOp::First:
          // The first contributing channel in declaration order wins; the
          // others are not read at all. An override, not a combination.
          num::copy<Space>(span_c(in_ptr(0), n), span_m(out, n));
          break;
        case MergeOp::Sum:
        case MergeOp::Mean: {
          // `accumulate<HostExecSpace>(AccumOp::Sum, ..., first)` (already
          // verified for every AccumOp by
          // SpaceKernels.AccumulateMatchesTheSerialOracleForEveryOp) sets
          // `out <- sample` when `first` and `out <- out + sample`
          // otherwise -- called once per input channel, in declaration
          // order, this is the SAME zero-then-accumulate-in-order the
          // hand-written loop it replaces did, just spelled as the kernel
          // Sample's own accumulation already uses.
          for (std::uint8_t k = 0; k < ins.n_in; ++k) {
            num::accumulate<Space>(AccumOp::Sum, span_c(in_ptr(k), n),
                                   span_m(out, n), k == 0);
          }
          if (ins.attrs.merge == MergeOp::Mean) {
            // Unweighted, as declared: the model carries no coverage
            // information that could weight the contributions, and inventing
            // weights here would be inventing physics.
            const double inv = 1.0 / static_cast<double>(ins.n_in);
            num::scale<Space>(inv, span_c(out, n), span_m(out, n));
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
      num::scale<Space>(f, span_c(in_ptr(0), n), span_m(out_ptr(), n));
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
      copy_arc(in_ptr(0), out_ptr(), n);
      break;
    }

    case OpKind::WindowAdvance:
      // Time windows track: a plan CAN contain this op now (Unit B of this
      // track compiles a window-scoped region's own field transfers
      // through the ordinary rules, with a WindowAdvance node per
      // exporting member; see src/ir/window_lowering.cpp), and it does
      // reach the interpreter -- but never through THIS switch. Unlike
      // every other instruction here, subcycling is not a stateless
      // function of its resolved input/output buffers -- it calls the
      // owning component's advance()/save_state()/restore_state()
      // directly and carries state across sweeps -- so it cannot be
      // executed through this switch's uniform
      // in_ptr()/out_ptr()/span_c()/span_m() convention, which has no
      // `ComponentSet&` to call any of those on. `run_region()`'s own
      // body loop (src/runtime/iteration.cpp) dispatches a
      // WindowAdvance instruction to `advance_window()` INSTEAD OF
      // calling this function at all -- reachable because a
      // WindowAdvance instruction only ever appears inside some window-
      // scoped region's body (build_iteration_regions(), src/ir/
      // regions.cpp, includes it by origin like any other node with that
      // channel's origin). This case is therefore a defensive catch-all
      // for a WindowAdvance instruction reaching `execute()` from
      // anywhere else -- which would be a scheduler defect, not a
      // reachable path today -- and refuses rather than falling through
      // to a copy, matching Map/Reduce/Closure immediately below: a
      // reserved operator that behaved as a pass-through would silently
      // run a time window as though it had been fully executed.
      return Failure{"E-RT-WINDOW-UNWIRED",
                     "operator 'WindowAdvance' reached execute() directly, "
                     "which has no ComponentSet& to advance/checkpoint the "
                     "owning member with; it should have been dispatched "
                     "through run_region()'s own advance_window() instead"};

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
      num::relax<Space>(ins.attrs.omega, span_c(prev.data(), n),
                        span_c(in_ptr(0), n), span_m(out, n));
      num::copy<Space>(span_c(out, n), span_m(prev.data(), n));
      break;
    }

    case OpKind::Transport: {
      if (ins.n_in == 0 || !ins.out.valid()) break;
      if (r.transport < 0) {
        return Failure{"E-RT-TRANSPORT",
                     "Transport instruction resolved to no plan"};
      }
      const std::size_t plan_idx = static_cast<std::size_t>(r.transport);
      detail::TransportPlan& p = transport_plans[plan_idx];
      const Slot& in_slot  = slots[ins.in[0].value()];
      const Slot& out_slot = slots[ins.out.value()];
      // docs/dev/device-resident-arcs.md Phase 3: build_transport_device_buffers()
      // mirrors a plan only when BOTH its arcs landed on the device; a
      // mixed or all-host pair leaves this entry empty and the code below
      // falls back to per-arc host staging (mixed) or the unchanged host
      // pointer arithmetic (all-host), exactly as every Transport
      // instruction ran before this track.
      detail::TransportDeviceOf<DeviceMemSpace>* d =
          (plan_idx < transport_device.size() &&
           transport_device[plan_idx].send.extent(0) > 0)
              ? &transport_device[plan_idx]
              : nullptr;

      if (!p.has_graph) {
        // Same-process move (the "copy" backend: no CommGraph was ever
        // attached -- attach_comm_graphs() builds one only for an MPI
        // exchange). A device-resident arc here is both ends of the SAME
        // process's memory, never a network exchange, so the move is one
        // Kokkos::deep_copy -- which bridges any combination of host and
        // device pointers by itself, unlike CopyTransport::move's
        // std::memcpy (host address on both sides, by contract), so EVERY
        // combination below calls it directly rather than only the
        // both-device one.
        if (in_slot.device && out_slot.device) {
          ++stats.device_transport_moves;
          Kokkos::deep_copy(num::SpanOf<Space>(out_ptr(), n),
                            num::ConstSpanOf<Space>(in_ptr(0), n));
          return Result<void>::success();
        }
        if (in_slot.device || out_slot.device) {
          ++stats.device_boundary_copies;
          if (in_slot.device) {
            Kokkos::deep_copy(num::SpanOf<HostExecSpace>(out_ptr(), n),
                              num::ConstSpanOf<Space>(in_ptr(0), n));
          } else {
            Kokkos::deep_copy(num::SpanOf<Space>(out_ptr(), n),
                              num::ConstSpanOf<HostExecSpace>(in_ptr(0), n));
          }
          return Result<void>::success();
        }
        const Result<void> prep = transport->prepare(p.spec);
        if (!prep.ok()) return prep;
        return transport->move(in_ptr(0), out_ptr(), n);
      }

      // This plan's own instance, not the shared one: see
      // Impl::graph_transports.
      Transport* tp = graph_transports[plan_idx].get();
      if (tp == nullptr) {
        return Failure{"E-RT-TRANSPORT",
                     "the CommGraph-backed plan for this Transport "
                     "instruction has no Transport instance; initialize() "
                     "must have failed and been ignored"};
      }

      const std::size_t nc =
          static_cast<std::size_t>(p.spec.components < 1 ? 1 : p.spec.components);

      if (d != nullptr) {
        // Both arcs are device-resident: pack and unpack with Kokkos
        // kernels on `Space` (fuse::num::gather/scatter -- the same
        // primitives the elementwise operators and the remap matvec
        // already use), and hand MPI a device pointer when the backend is
        // GPU-aware right now (`Transport::is_gpu_aware()`, a run-time
        // probe -- not a configure-time constant); otherwise deep_copy the
        // packed buffer to the plan's existing host `send`/`recv` and
        // exchange through those, exactly as the host path always has,
        // then deep_copy the result back before unpacking on the device.
        // `FUSE_TRANSPORT_HOST_STAGING=1` forces this second branch even
        // on a GPU-aware MPI, so the fallback has a test on every machine.
        const double* in  = in_ptr(0);
        double*       out = out_ptr();
        const bool gpu_aware = tp->is_gpu_aware();

        if (p.from_peers) {
          // A peer plan indexes both buffers by local entity index and
          // performs the same-rank copies itself (MpiTransport::move,
          // local_copy), so the matched path's gather/scatter -- empty
          // here -- must not run. Stage the whole source block, on the
          // device.
          if (p.peer_src_total > 0) {
            Kokkos::deep_copy(num::SpanOf<Space>(d->send.data(), p.peer_src_total),
                              num::ConstSpanOf<Space>(in, p.peer_src_total));
          }
          const Result<void> prep = tp->prepare(p.spec);
          if (!prep.ok()) return prep;
          Result<void> mv = Result<void>::success();
          if (gpu_aware) {
            ++stats.device_transport_moves;
            mv = tp->move(d->send.data(), d->recv.data(), p.total());
          } else {
            Kokkos::deep_copy(num::SpanOf<HostExecSpace>(p.send.data(), p.total()),
                              num::ConstSpanOf<Space>(d->send.data(), p.total()));
            ++stats.device_boundary_copies;
            mv = tp->move(p.send.data(), p.recv.data(), p.total());
            Kokkos::deep_copy(num::SpanOf<Space>(d->recv.data(), p.total()),
                              num::ConstSpanOf<HostExecSpace>(p.recv.data(), p.total()));
            ++stats.device_boundary_copies;
          }
          if (!mv.ok()) return mv;
          // Only what the plan wrote: an entry it did not write holds the
          // staging zero, not a value, and must not overwrite the field.
          // Expressed as gather-then-scatter at the SAME positions (a
          // masked copy) rather than a third bespoke kernel.
          if (d->peer_written.extent(0) > 0) {
            num::gather<Space>(
                num::ConstSpanOf<Space>(d->recv.data(), d->recv.extent(0)),
                num::IndexSpanOf<Space>(d->peer_written.data(), d->peer_written.extent(0)),
                num::SpanOf<Space>(d->peer_scratch.data(), d->peer_scratch.extent(0)));
            num::scatter<Space>(
                num::ConstSpanOf<Space>(d->peer_scratch.data(), d->peer_scratch.extent(0)),
                num::IndexSpanOf<Space>(d->peer_written.data(), d->peer_written.extent(0)),
                num::SpanOf<Space>(out, out_slot.total()));
          }
          return Result<void>::success();
        }

        // Matches whose two ends are on this rank never reach MPI at all
        // (comm_graph.hpp:75-86): gather FROM `in` at `local_src` into a
        // scratch buffer, then scatter the scratch INTO `out` at
        // `local_dst` -- two kernels standing in for the one permutation
        // neither `gather` nor `scatter` alone can express (each varies
        // only one of its two indices per call).
        if (d->local_src.extent(0) > 0) {
          num::gather<Space>(
              num::ConstSpanOf<Space>(in, in_slot.total()),
              num::IndexSpanOf<Space>(d->local_src.data(), d->local_src.extent(0)),
              num::SpanOf<Space>(d->local_scratch.data(), d->local_scratch.extent(0)));
          num::scatter<Space>(
              num::ConstSpanOf<Space>(d->local_scratch.data(), d->local_scratch.extent(0)),
              num::IndexSpanOf<Space>(d->local_dst.data(), d->local_dst.extent(0)),
              num::SpanOf<Space>(out, out_slot.total()));
        }

        // Gather. `global_ids` is a SUBSET of this rank's entities, so it
        // does not index the field buffer; `d->gather` (built from
        // `CommGraph::source_indices()`) is the map that does.
        if (d->gather.extent(0) > 0) {
          num::gather<Space>(
              num::ConstSpanOf<Space>(in, in_slot.total()),
              num::IndexSpanOf<Space>(d->gather.data(), d->gather.extent(0)),
              num::SpanOf<Space>(d->send.data(), d->gather.extent(0)));
        }

        const Result<void> prep = tp->prepare(p.spec);
        if (!prep.ok()) return prep;
        Result<void> mv = Result<void>::success();
        // p.total(), NOT the arc's n: move() requires exactly elements *
        // components OF THE PLANNED SPEC, and the planned spec is the
        // subset this rank actually exchanges.
        if (gpu_aware) {
          ++stats.device_transport_moves;
          mv = tp->move(d->send.data(), d->recv.data(), p.total());
        } else {
          Kokkos::deep_copy(num::SpanOf<HostExecSpace>(p.send.data(), p.total()),
                            num::ConstSpanOf<Space>(d->send.data(), p.total()));
          ++stats.device_boundary_copies;
          mv = tp->move(p.send.data(), p.recv.data(), p.total());
          Kokkos::deep_copy(num::SpanOf<Space>(d->recv.data(), p.total()),
                            num::ConstSpanOf<HostExecSpace>(p.recv.data(), p.total()));
          ++stats.device_boundary_copies;
        }
        if (!mv.ok()) return mv;

        if (d->scatter.extent(0) > 0) {
          num::scatter<Space>(
              num::ConstSpanOf<Space>(d->recv.data(), d->scatter.extent(0)),
              num::IndexSpanOf<Space>(d->scatter.data(), d->scatter.extent(0)),
              num::SpanOf<Space>(out, out_slot.total()));
        }
        return Result<void>::success();
      }

      // Host path: byte-for-byte what every Transport instruction ran
      // before this track, EXCEPT that `in`/`out` are staged through a
      // per-plan host scratch buffer first when the arc they name is
      // device-resident -- a MIXED pair, which build_transport_device_
      // buffers() leaves with no `transport_device` entry precisely so
      // this staging runs (a Remap needing conservation checking pins the
      // Constrain's before_arc, frequently this Transport's own output,
      // while the OTHER arc, untouched by any other host-only
      // instruction, landed on the device like everything around it --
      // RunnerDevice.DeviceAndHostAgreeToRoundingOnRelaxAndWeightedRemap
      // and .OrdinaryIterationRegionConvergesOnTheDeviceLikeTheHost are
      // both no-graph examples of exactly this; a CommGraph-backed one
      // reaches here the same way). Reached under Host unconditionally
      // (every slot's `device` flag is false there, so both stagings
      // below are no-ops and `in`/`out` are the arcs' own pointers,
      // unchanged).
      const double* in = in_ptr(0);
      double* out = out_ptr();
      if (in_slot.device) {
        std::vector<double>& scratch = transport_in_scratch[plan_idx];
        Kokkos::deep_copy(num::SpanOf<HostExecSpace>(scratch.data(), in_slot.total()),
                          num::ConstSpanOf<Space>(in, in_slot.total()));
        in = scratch.data();
        ++stats.device_boundary_copies;
      }
      double* out_device_ptr = nullptr;
      if (out_slot.device) {
        out_device_ptr = out;
        out = transport_out_scratch[plan_idx].data();
      }

      Result<void> result = Result<void>::success();
      if (p.from_peers) {
        // A peer plan indexes both buffers by local entity index and performs
        // the same-rank copies itself (MpiTransport::move, local_copy), so the
        // matched path's gather/scatter -- empty here -- must not run. Stage
        // the whole source block; `send` is at least that large because the
        // plan's extent is the larger of the two blocks.
        for (std::size_t k = 0; k < p.peer_src_total; ++k) p.send[k] = in[k];
        result = tp->prepare(p.spec);
        if (result.ok()) {
          result = tp->move(p.send.data(), p.recv.data(), p.total());
        }
        if (result.ok()) {
          // Only what the plan wrote: an entry it did not write holds the
          // staging zero, not a value, and must not overwrite the field.
          for (const std::size_t d2 : p.peer_written) {
            for (std::size_t c = 0; c < nc; ++c) {
              out[d2 * nc + c] = p.recv[d2 * nc + c];
            }
          }
        }
      } else {
        // Matches whose two ends are on this rank never reach the Transport at
        // all -- comm_graph.hpp:75-86. One global_ids list cannot describe a
        // rank that is both source and destination, so splitting these out is
        // what makes the exchange expressible, not an optimization.
        for (std::size_t k = 0; k < p.local_src.size(); ++k) {
          const std::size_t s = p.local_src[k] * nc;
          const std::size_t d2 = p.local_dst[k] * nc;
          for (std::size_t c = 0; c < nc; ++c) out[d2 + c] = in[s + c];
        }

        // Gather. `global_ids` is a SUBSET of this rank's entities, so it does
        // not index the field buffer; source_indices() is the map that does.
        for (std::size_t k = 0; k < p.gather.size(); ++k) {
          const std::size_t s = p.gather[k] * nc;
          for (std::size_t c = 0; c < nc; ++c) p.send[k * nc + c] = in[s + c];
        }

        result = tp->prepare(p.spec);
        if (result.ok()) {
          // p.total(), NOT the arc's n: move() requires exactly elements *
          // components OF THE PLANNED SPEC, and the planned spec is the
          // subset this rank actually exchanges.
          result = tp->move(p.send.data(), p.recv.data(), p.total());
        }
        if (result.ok()) {
          for (std::size_t k = 0; k < p.scatter.size(); ++k) {
            const std::size_t d2 = p.scatter[k] * nc;
            for (std::size_t c = 0; c < nc; ++c) out[d2 + c] = p.recv[k * nc + c];
          }
        }
      }

      if (result.ok() && out_device_ptr != nullptr) {
        Kokkos::deep_copy(num::SpanOf<Space>(out_device_ptr, out_slot.total()),
                          num::ConstSpanOf<HostExecSpace>(out, out_slot.total()));
        ++stats.device_boundary_copies;
      }
      return result;
    }
    case OpKind::Residual: {
      if (ins.n_in < 1 || !ins.out.valid()) break;
      double* out = out_ptr();
      const double* a = in_ptr(0);
      if (ins.n_in > 1) {
        const double* b = in_ptr(1);
        num::residual<Space>(span_c(a, n), span_c(b, n), span_m(out, n));
      } else {
        std::vector<double>& prev = previous[ins.out.value()];  // pre-sized
        num::residual<Space>(span_c(a, n), span_c(prev.data(), n),
                             span_m(out, n));
        num::copy<Space>(span_c(a, n), span_m(prev.data(), n));
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
      double before = num::weighted_sum(src.data, wb.data(), wb.size());
      double after = num::weighted_sum(out, wa.data(), wa.size());
      if (ci.reduce) {
#if FUSE_HAVE_MPI
        // KNOWN_ISSUES.md D10: the owned shares summed over every rank, plus
        // a flag from any rank whose period has already failed -- its buffers
        // are whatever the abandoned step left, and summing them would turn
        // one real failure into a spurious conservation error on every
        // partner. All three ride one Allreduce; all ranks see identical
        // totals, so the verdict below is the same everywhere without any
        // further agreement.
        const double local[3] = {before, after,
                                 step_failure.has_value() ? 1.0 : 0.0};
        double total[3] = {0.0, 0.0, 0.0};
        const MPI_Comm base =
            MPI_Comm_f2c(static_cast<MPI_Fint>(*config.base_communicator));
        if (MPI_Allreduce(local, total, 3, MPI_DOUBLE, MPI_SUM, base) !=
            MPI_SUCCESS) {
          return Failure{"E-RT-CONSERVATION-REDUCE",
                       "MPI_Allreduce of the conservation integrals for arc %" +
                           std::to_string(ins.out.value()) + " failed"};
        }
        if (total[2] > 0.0) break;  // a partner's period is already lost
        before = total[0];
        after = total[1];
#else
        return Failure{"E-RT-CONSERVATION-REDUCE",
                     "a base communicator is configured but this build has no MPI"};
#endif
      }
      // Opt-in, like FUSE_TRANSPORT_DEBUG: one line per check per process,
      // and a getenv per check otherwise. Prints this process's own shares
      // (before reduction) next to the totals the verdict uses, plus how
      // many weights are non-zero on each side, so a wrong ownership or a
      // wrong weight is visible without a debugger.
      if (std::getenv("FUSE_CONSERVATION_DEBUG") != nullptr) {
        const double own_before =
            num::weighted_sum(src.data, wb.data(), wb.size());
        const double own_after = num::weighted_sum(out, wa.data(), wa.size());
        std::size_t nzb = 0, nza = 0;
        for (const double w : wb) nzb += w != 0.0 ? 1 : 0;
        for (const double w : wa) nza += w != 0.0 ? 1 : 0;
        int me = -1;
#if FUSE_HAVE_MPI
        if (config.base_communicator.has_value()) {
          MPI_Comm_rank(MPI_Comm_f2c(static_cast<MPI_Fint>(*config.base_communicator)), &me);
        }
#endif
        fmt::print(stderr,
                   "[conservation rank {} period {} arc %{}<-%{}] own before={:.17g} "
                   "({} of {} weights non-zero) own after={:.17g} ({} of {}) "
                   "total before={:.17g} after={:.17g} reduced={}\n",
                   me, period_index, ins.out.value(), ci.before_arc, own_before,
                   nzb, wb.size(), own_after, nza, wa.size(), before, after,
                   ci.reduce ? 1 : 0);
      }
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

  // A5: every Space-generic case above ends in `break`, so this is the one
  // place a device-staged output returns to its host buffer. The host-only
  // cases that `return` early staged nothing and never reach here with
  // `n_staged > 0`.
  if constexpr (!kHostAccessible) {
    if (staging_fault) {
      return Failure{"E-RT-DEVICE",
                   std::string{"instruction '"} + to_cstring(ins.kind) +
                       "' needed more device staging than setup() "
                       "provisioned (" +
                       std::to_string(kDeviceStagingSlots) + " lanes of " +
                       std::to_string(device_staging_stride) + " values)"};
    }
    for (std::size_t i = 0; i < n_staged; ++i) {
      if (staged[i].host_mut == nullptr) continue;
      Kokkos::deep_copy(
          num::SpanOf<HostExecSpace>(staged[i].host_mut, staged[i].len),
          num::ConstSpanOf<Space>(device_staging.data() +
                                      i * device_staging_stride,
                                  staged[i].len));
      ++stats.device_staged_copies;
    }
    if (n_staged > 0 || used_device_arc) ++stats.device_dispatches;
  }
  return Result<void>::success();
}

void Runner::Impl::block_advances_from(std::size_t origin) {
  if (downstream_of.empty()) return;
  if (advance_blocked.empty()) advance_blocked.assign(downstream_of.size(), 0);
  if (origin >= downstream_of.size()) {
    // An unattributable failure blocks everything, which is what this rule
    // replaced. Being unable to name the origin is not a licence to advance
    // components that might be downstream of it.
    advance_blocked.assign(downstream_of.size(), 1);
    return;
  }
  advance_blocked[origin] = 1;
  for (std::size_t b = 0; b < downstream_of.size(); ++b) {
    if (downstream_of[origin][b] != 0) advance_blocked[b] = 1;
  }
}

bool Runner::Impl::may_advance(std::size_t slot) const {
  if (advance_blocked.empty()) return true;
  return slot < advance_blocked.size() && advance_blocked[slot] == 0;
}

detail::Task<Result<void>> Runner::Impl::advance(ComponentSet& components,
                                                  const Tick& tick,
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
    co_return Failure{"E-RT-STEP",
                 "step context [" + ctx.t_begin.to_string() + ", " +
                     ctx.t_end.to_string() + ") for component '" +
                     components.at(tick.slot).name() + "' is not valid"};
  }
  Component& comp = components.at(tick.slot);
  // KNOWN_ISSUES.md D12: an in-process component takes this branch never --
  // `is_externally_driven()` is false for every one of them -- so the
  // statement immediately below (the unconditional `comp.advance(ctx)` this
  // function has always ended with) runs at exactly the same point in
  // exactly the same order it always has, with no suspension anywhere in
  // between. Only `abi_detail::ExternalStepComponent` (src/abi/
  // external_step_component.hpp) takes it.
  if (comp.is_externally_driven()) {
    ExternalAdvanceRequest req;
    req.component = comp.name();
    req.ctx = ctx;
    // Consumed, not just read: each flag must be reported true at most
    // once, on the first advance() resumed to after the save_state()/
    // restore_state() that set it (see runner_impl.hpp's own comment on
    // these two vectors).
    if (tick.slot < slot_just_checkpointed.size()) {
      req.requires_checkpoint = slot_just_checkpointed[tick.slot] != 0;
      slot_just_checkpointed[tick.slot] = 0;
    }
    if (tick.slot < slot_just_restored.size()) {
      req.requires_restore = slot_just_restored[tick.slot] != 0;
      slot_just_restored[tick.slot] = 0;
    }
    pending_external_request = std::move(req);
    co_await detail::ExternalSuspend{&active_handle};
  }
  const Result<void> adv = comp.advance(ctx);
  note_component_ran();
  co_return adv;
}

detail::Task<Result<void>> Runner::Impl::step_period(ComponentSet& components) {
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
    if (!c.ok()) co_return c;
  }

  advance_blocked.clear();

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
      // A component refusing its step no longer stops components that read
      // nothing it provides: the origin and everything downstream of it are
      // blocked (Impl::downstream_of), the rest still step. Their values are
      // as valid as in a period that succeeded, and the period still reports
      // the failure at its end.
      if (may_advance(ticks[ti].slot)) {
        const Result<void> a = co_await advance(components, ticks[ti], 0);
        if (!a.ok()) {
          if (!step_failure.has_value()) step_failure = a.error();
          block_advances_from(ticks[ti].slot);
        }
      }
      ++ti;
    }
    if (ins.region_enter >= 0) {
      const std::int32_t region_idx = ins.region_enter;
      detail::RegionRuntime& r =
          regions.at(static_cast<std::size_t>(region_idx));
      // Time windows track: THIS run's own exit position, not r.exit --
      // a window-scoped region can have more than one pass per period
      // (RegionRuntime::passes' own comment), so a stale single r.exit
      // (fixed at resolve_regions() time to the FIRST pass only) would
      // desynchronise the walk on the second and later window instances.
      // `i` is already this pass's own enter (mark_regions() stamped it
      // there); scan forward for the matching exit, the same technique
      // mark_regions() itself uses to find one run's own boundary.
      std::size_t exit_pos = i;
      while (exit_pos < stream.size() &&
             stream[exit_pos].region_exit != region_idx) {
        ++exit_pos;
      }
      const Result<void> rr = co_await run_region(components, r, i, exit_pos);
      // R-09 (REVIEW.md): a region failure carries forward exactly like an
      // ordinary instruction failure two branches below -- record it and
      // keep walking the stream -- instead of returning straight out of
      // step_period(). Returning here abandoned every later Epoch/Transport
      // in the same period's instruction stream, including ones for a
      // channel that has nothing to do with this region; a partner rank
      // waiting at one of those later agreement points blocked forever.
      if (!rr.ok()) {
        if (!step_failure.has_value()) step_failure = rr.error();
        // The region's members are the origin: a region failure is not
        // attributable to one component (a convergence failure is
        // attributable to none), so everything it re-advances is suspect.
        for (const std::size_t slot : r.component_slots) {
          block_advances_from(slot);
        }
      }
      i = exit_pos + 1;
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
      if (ins.kind == OpKind::Epoch) co_return *step_failure;
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
    // Not `break` on any failure -- see the tick loop above. A period's ticks
    // routinely ALL land here, so breaking made one component's refusal skip
    // every other component's step, unrelated ones included.
    if (!may_advance(ticks[ti].slot)) continue;
    const Result<void> a = co_await advance(components, ticks[ti], 0);
    if (!a.ok()) {
      if (!step_failure.has_value()) step_failure = a.error();
      block_advances_from(ticks[ti].slot);
    }
  }

  if (step_failure.has_value()) {
    // Reported at the END of the period, not at the point of failure: by here
    // every agreement point after the failure has been reached, so the
    // partners know too. `periods` is deliberately not incremented -- a period
    // that failed did not happen.
    const Failure f = *step_failure;
    step_failure.reset();
    co_return f;
  }

  ++stats.periods;
  ++period_index;
  co_return Result<void>::success();
}

// ---------------------------------------------------------------------------
// Public surface
// ---------------------------------------------------------------------------

Runner::Runner(const CouplingGraph& graph, const OperatorDAG& dag,
               ExecutionPlan plan, const MeshBackend& backend)
    : impl_(std::make_unique<Impl>(graph, dag, std::move(plan), backend)) {}
Runner::~Runner() = default;

std::string_view Runner::device_execution_space_name() noexcept {
  return Impl::DeviceExecSpace::name();
}
Runner::Runner(Runner&&) noexcept = default;
Runner& Runner::operator=(Runner&&) noexcept = default;


// Every Transport's buffers against the plan it was actually given -- run
// after attach_comm_graphs(), the first point at which that is known.
//
// A whole-arc copy (no CommGraph) moves `n` doubles from one buffer to the
// other, so the two must be the same size. A CommGraph plan reads and writes
// by index, and since each end is its own component's decomposition the two
// buffers may differ; what must hold is that every index it uses lies inside
// the buffer it uses it on. An out-of-range index here is a heap overrun in
// the hot loop, which checks nothing, so it is refused before the first step.
Result<void> Runner::Impl::check_transport_extents() const {
  const std::vector<Instruction>& stream = plan.instructions();
  auto size_failure = [](std::size_t i, const std::string& what) {
    return Failure{"E-RT-SIZE", "Transport instruction " + std::to_string(i) +
                                    ": " + what};
  };
  for (std::size_t i = 0; i < stream.size(); ++i) {
    const Instruction& ins = stream[i];
    if (ins.kind != OpKind::Transport || ins.n_in == 0 || !ins.out.valid()) {
      continue;
    }
    const std::int32_t ti = resolved[i].transport;
    if (ti < 0 || static_cast<std::size_t>(ti) >= transport_plans.size()) {
      continue;  // refused at execution with E-RT-TRANSPORT
    }
    const detail::TransportPlan& p = transport_plans[static_cast<std::size_t>(ti)];
    const Slot& in = slots[ins.in[0].value()];
    const Slot& out = slots[ins.out.value()];
    const std::size_t in_n = in.total();
    const std::size_t out_n = out.total();

    if (!p.has_graph) {
      if (in_n != out_n) {
        return size_failure(
            i, "reads " + std::to_string(in_n) + " values from arc %" +
                   std::to_string(ins.in[0].value()) + " into " +
                   std::to_string(out_n) + " at arc %" +
                   std::to_string(ins.out.value()) +
                   "; without a CommGraph the whole arc is moved, so the two "
                   "ends must be the same size -- set base_communicator and "
                   "the \"mpi\" transport for placements whose "
                   "decompositions differ");
      }
      continue;
    }

    const std::size_t nc =
        static_cast<std::size_t>(p.spec.components < 1 ? 1 : p.spec.components);
    auto fits = [nc](std::size_t idx, std::size_t n) {
      return (idx + 1) * nc <= n;
    };

    if (p.from_peers) {
      if (p.peer_src_total != in_n || p.peer_src_total > p.send.size()) {
        return size_failure(
            i, "the peer plan stages " + std::to_string(p.peer_src_total) +
                   " source values but arc %" + std::to_string(ins.in[0].value()) +
                   " holds " + std::to_string(in_n));
      }
      for (const std::size_t d : p.peer_written) {
        if (!fits(d, out_n) || !fits(d, p.recv.size())) {
          return size_failure(
              i, "the peer plan writes destination entity " + std::to_string(d) +
                     ", outside the " + std::to_string(out_n) +
                     " values of arc %" + std::to_string(ins.out.value()));
        }
      }
      continue;
    }

    for (std::size_t k = 0; k < p.gather.size(); ++k) {
      if (!fits(p.gather[k], in_n) || !fits(k, p.send.size())) {
        return size_failure(i, "gather index " + std::to_string(p.gather[k]) +
                                   " is outside the " + std::to_string(in_n) +
                                   " source values");
      }
    }
    for (std::size_t k = 0; k < p.scatter.size(); ++k) {
      if (!fits(p.scatter[k], out_n) || !fits(k, p.recv.size())) {
        return size_failure(i, "scatter index " + std::to_string(p.scatter[k]) +
                                   " is outside the " + std::to_string(out_n) +
                                   " destination values");
      }
    }
    for (std::size_t k = 0; k < p.local_src.size(); ++k) {
      if (!fits(p.local_src[k], in_n) || !fits(p.local_dst[k], out_n)) {
        return size_failure(i, "a same-rank copy names an entity outside "
                               "one of the two buffers");
      }
    }
  }
  return Result<void>::success();
}

Result<RunnerConfig::RunnerExecutionSpace>
RunnerConfig::resolved_execution_space() const {
  if (execution_space != RunnerExecutionSpace::Default) return execution_space;
  const char* env = std::getenv("FUSE_EXECUTION_SPACE");
  std::string v = env == nullptr ? std::string{} : std::string{env};
  for (char& ch : v) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  if (v.empty() || v == "host") return RunnerExecutionSpace::Host;
  if (v == "device") return RunnerExecutionSpace::Device;
  return Failure{"E-RT-SPACE",
               "FUSE_EXECUTION_SPACE='" + std::string{env} +
                   "' is not a Runner execution space; use 'host' or 'device' "
                   "(or unset it), or set RunnerConfig::execution_space"};
}

Result<void> Runner::initialize(ComponentSet& components,
                                FieldRegistry& registry,
                                const RunnerConfig& config) {
  impl_->initialized = false;
  impl_->config = config;
  {
    // RunnerConfig::execution_space's own comment: `Default` is the
    // environment's choice, decided once here so every later test of
    // `config.execution_space` sees Host or Device.
    const Result<RunnerConfig::RunnerExecutionSpace> space =
        config.resolved_execution_space();
    if (!space.ok()) return space.error();
    impl_->config.execution_space = space.value();
  }

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
    // A fresh, self-contained coroutine: unlike resume_period() below, this
    // task is driven to completion in this one call and never stored in
    // impl_->period_task -- the two entry points are mutually exclusive on
    // a given Runner (runner.hpp's own doc comment on resume_period()).
    detail::Task<Result<void>> task = impl_->step_period(components);
    const bool finished = task.resume();
    if (!finished) {
      // KNOWN_ISSUES.md D12: step_period() suspends only at a component
      // whose is_externally_driven() is true (component.hpp) -- nothing a
      // run_period()/run() caller's own ComponentSet should ever contain,
      // since that caller has no begin_step()/end_step() to resume past
      // the suspension with. Refusing by name, rather than deadlocking on
      // a Task this function then abandons mid-period, is deliberate:
      // Runner itself cannot tell "a stray external component" from "the
      // caller meant to use resume_period() and reached for the wrong
      // method", so it reports the fact rather than guessing which.
      const std::string name = impl_->pending_external_request.has_value()
                                   ? impl_->pending_external_request->component
                                   : std::string{"<unknown>"};
      impl_->pending_external_request.reset();
      return Failure{
          "E-RT-EXTERNAL-UNSUPPORTED",
          "run_period(): component '" + name +
              "' is driven from outside the Runner (Component::"
              "is_externally_driven()); call Runner::resume_period() "
              "instead of run_period()/run() for a graph with an "
              "externally driven component"};
    }
    return task.take_result();
  } catch (const std::exception& e) {
    return Failure{"E-RT-RUN", std::string{"period failed: "} + e.what()};
  } catch (...) {
    return Failure{"E-RT-RUN",
                 "period threw an exception that does not derive from "
                 "std::exception"};
  }
}

Result<std::optional<ExternalAdvanceRequest>> Runner::resume_period(
    ComponentSet& components) {
  if (!impl_->initialized) {
    return Failure{"E-RT-INIT",
                 "resume_period() called before a successful initialize()"};
  }
  try {
    return impl_->resume_period(components);
  } catch (const std::exception& e) {
    return Failure{"E-RT-RUN", std::string{"period failed: "} + e.what()};
  } catch (...) {
    return Failure{"E-RT-RUN",
                 "period threw an exception that does not derive from "
                 "std::exception"};
  }
}

std::vector<ComponentId> Runner::components_due_at_period_start() const {
  return impl_->components_due_at_period_start();
}

std::vector<ComponentId> Runner::Impl::components_due_at_period_start() const {
  std::vector<ComponentId> out;
  const Rational zero{0};
  // Ordinary (non-window) ticks ALWAYS include a k=0 entry, at time 0,
  // for every ordinary component (build_ticks()'s own "Rational{k}*dt"
  // for k = 0 .. n-1) -- and a window region's own first-pass block is
  // NEVER stamped earlier than its own (strictly positive) window
  // length: a WindowAdvance instruction is a chain node, stamped at the
  // SLAB END per plan.hpp's own documented convention ("chain nodes at
  // the slab end"), not at its start -- MEASURED directly against this
  // exact confusion (an earlier version of this function filtered
  // WindowAdvance instructions by `.time == 0` and found every one of
  // them excluded, since their `.time` is the window's own end, e.g.
  // 1/500 for a 1/500 window, never 0). So whenever `ticks` is non-empty
  // at all, its own time-0 prefix -- already sorted ascending by
  // (time, order), build_ticks()'s own stable_sort -- is unconditionally
  // the period's own first due point; no window region can tie with it,
  // let alone beat it.
  for (const detail::Tick& tk : ticks) {
    if (tk.time != zero) break;  // sorted ascending: nothing past here ties
    out.push_back(ComponentId{tk.order});
  }
  if (!out.empty()) return out;

  // No ordinary component at all in this graph (every one is a window
  // member, or there are truly none) -- the period's own first due point
  // is the FIRST window region's own first pass, found by ITS position
  // in the (already time-sorted) instruction stream: the earliest
  // `region_enter` reached walking from the front. Within that block,
  // WindowAdvance instructions appear in the SAME order
  // members_in_first_seen_order() (src/ir/window_lowering.cpp) created
  // their nodes in -- a WindowAdvance node has no formal DAG inputs at
  // all (op.hpp's own comment on OpAttrs::window_import explains why),
  // so NodeId/creation order, not a dependency edge, is what the
  // scheduler's own tie-break (time, phase, topological rank, NodeId)
  // actually uses to order two members against each other.
  const std::vector<Instruction>& stream = plan.instructions();
  std::size_t i = 0;
  while (i < stream.size() && stream[i].region_enter < 0) ++i;
  if (i >= stream.size()) return out;  // no region either: truly nothing due
  std::size_t exit_pos = i;
  while (exit_pos < stream.size() &&
         stream[exit_pos].region_exit != stream[i].region_enter) {
    ++exit_pos;
  }
  for (std::size_t k = i; k <= exit_pos && k < stream.size(); ++k) {
    const Instruction& wi = stream[k];
    if (wi.kind != OpKind::WindowAdvance) continue;
    // Mirrors resolve_regions()'s own lookup (src/runtime/iteration.cpp):
    // a WindowAdvance instruction's own output arc carries the owning
    // member's graph ComponentId as `rank_set`.
    const int rank_set = dag.arc(wi.out).type.rank_set;
    if (rank_set >= 0) {
      out.push_back(ComponentId{static_cast<std::uint32_t>(rank_set)});
    }
  }
  return out;
}

std::optional<ComponentId> Runner::next_real_advance_after(
    const ComponentSet& components, std::size_t instruction_index) const {
  return impl_->next_real_advance_after(components, instruction_index);
}

std::optional<ComponentId> Runner::Impl::next_real_advance_after(
    const ComponentSet& components, std::size_t instruction_index) const {
  const std::vector<Instruction>& stream = plan.instructions();
  if (instruction_index >= stream.size()) return std::nullopt;
  const Rational from_time = stream[instruction_index].time;

  // The SAME hosted formula deliver_external() and build_ticks()'s own
  // hosted[] use, plus the externally-driven check deliver_external()
  // also makes before suspending -- a candidate failing either is
  // skipped (not returned), because the scan keeps going: the next
  // eligible one further out is still a real candidate, unlike
  // components_due_at_period_start()'s own "return everything, caller
  // picks the first eligible one", which only works for a genuine tie
  // at time 0 (this function's own declaration, runner_impl.hpp, has
  // the full account of the difference).
  const auto eligible = [&](const ComponentId cid) -> bool {
    const ComponentDecl& decl = graph.component(cid);
    const bool hosted =
        !this_process.has_value() || decl.ranks.contains(*this_process);
    if (!hosted) return false;
    const Component* comp = components.find(decl.name);
    return comp != nullptr && comp->is_externally_driven();
  };

  // Ordinary ticks strictly after this instruction's own time -- `ticks`,
  // not `all_ticks`, is already filtered to this process's own hosted
  // components (build_ticks()'s own comment on the two), but NOT to
  // externally-driven ones, so `eligible()` still has work to do.
  std::optional<ComponentId> tick_candidate;
  Rational tick_candidate_time{0};
  for (const detail::Tick& tk : ticks) {
    if (!(tk.time > from_time)) continue;
    const ComponentId cid{tk.order};
    if (!eligible(cid)) continue;
    if (!tick_candidate.has_value() || tk.time < tick_candidate_time ||
        (tk.time == tick_candidate_time && tk.order < tick_candidate->value())) {
      tick_candidate = cid;
      tick_candidate_time = tk.time;
    }
  }

  // WindowAdvance instructions later in the SAME compiled stream --
  // components_due_at_period_start()'s own region scan, starting from
  // `instruction_index + 1` instead of the stream's front; stream order
  // is already this period's own schedule order, so the first eligible
  // one found is the earliest.
  //
  // The lead's own ruling (2026-10-04, correcting the SAME day's earlier
  // one): a plain forward scan to `stream.size()` is only correct when
  // `instruction_index` lies outside every iteration region. Inside one,
  // whether the scan may walk past that region's own `exit` depends on a
  // convergence decision the region's sweep loop makes AFTER this
  // instruction runs -- not yet taken, and not knowable from the
  // compiled plan alone. So: if `instruction_index` falls inside some
  // region's own (enter, exit) pass, the forward scan is bounded to that
  // pass's own `exit`, and -- finding nothing there -- WRAPS to that same
  // pass's own `enter` and continues up to (not including)
  // `instruction_index`, because that is the instruction sequence the
  // sweep loop actually runs if it loops back for another sweep. The
  // result is therefore a CONDITIONAL answer -- "the component due IF
  // the iteration continues" (ExternalAdvanceRequest::is_delivery_due's
  // own comment, runner.hpp, states this plainly) -- never a promise:
  // whether the iteration really continues is resolved for real only by
  // actually resuming the Runner (prime_next_request()'s own rollover,
  // src/abi/participant.cpp), which is free to find the region converged
  // and the period over instead. Only when `instruction_index` lies
  // outside every region does "found nothing" legitimately mean "nothing
  // remains in this compiled period", the case the caller's own fallback
  // to the next period's start (components_due_at_period_start()) is
  // for.
  std::optional<std::pair<std::size_t, std::size_t>> containing_pass;
  for (const detail::RegionRuntime& region : regions) {
    for (const auto& [enter, exit] : region.passes) {
      if (instruction_index >= enter && instruction_index <= exit) {
        containing_pass = std::make_pair(enter, exit);
        break;
      }
    }
    if (containing_pass.has_value()) break;
  }
  const std::size_t forward_limit =
      containing_pass.has_value() ? containing_pass->second + 1 : stream.size();
  std::optional<ComponentId> window_candidate;
  Rational window_candidate_time{0};
  for (std::size_t k = instruction_index + 1; k < forward_limit; ++k) {
    if (stream[k].kind != OpKind::WindowAdvance) continue;
    const int rank_set = dag.arc(stream[k].out).type.rank_set;
    if (rank_set < 0) continue;
    const ComponentId cid{static_cast<std::uint32_t>(rank_set)};
    if (!eligible(cid)) continue;
    window_candidate = cid;
    window_candidate_time = stream[k].time;
    break;
  }
  if (!window_candidate.has_value() && containing_pass.has_value()) {
    // Not found forward to this pass's own exit: wrap to its own entry,
    // the instruction sequence the sweep loop actually resumes at if it
    // loops back for another sweep (this function's own header comment,
    // runner.hpp -- NOT a fall-through to the next period, since we are
    // still inside a region).
    for (std::size_t k = containing_pass->first; k < instruction_index; ++k) {
      if (stream[k].kind != OpKind::WindowAdvance) continue;
      const int rank_set = dag.arc(stream[k].out).type.rank_set;
      if (rank_set < 0) continue;
      const ComponentId cid{static_cast<std::uint32_t>(rank_set)};
      if (!eligible(cid)) continue;
      window_candidate = cid;
      window_candidate_time = stream[k].time;
      break;
    }
  }

  if (tick_candidate.has_value() && window_candidate.has_value()) {
    // A genuine tie between an ordinary tick and a window boundary is
    // not exercised by any graph this codebase tests today; broken
    // toward the window candidate, arbitrarily but deterministically
    // (this function's own declaration, runner_impl.hpp).
    return (tick_candidate_time < window_candidate_time) ? tick_candidate
                                                          : window_candidate;
  }
  if (tick_candidate.has_value()) return tick_candidate;
  if (window_candidate.has_value()) return window_candidate;
  return std::nullopt;
}

Result<std::optional<ExternalAdvanceRequest>> Runner::Impl::resume_period(
    ComponentSet& components) {
  if (!period_task.has_value() || period_task->done()) {
    // Either the very first call, or the previous period_task ran to
    // completion and was never read out (resume_period() always reads it
    // out itself, below, the same call it finishes on) -- either way,
    // start the next period's coroutine fresh. initial_suspend() is
    // suspend_always, so nothing in step_period() has run yet, and its
    // handle is exactly what the next line below must resume.
    period_task = step_period(components);
    active_handle = period_task->raw_handle();
  }
  pending_external_request.reset();
  // resume.hpp's own comment on ExternalSuspend: NEVER
  // `period_task->resume()` here. Once a period has suspended even once,
  // step_period()'s own handle is no longer the frame actually waiting to
  // run -- active_handle is, wherever `advance()`/`advance_window()` last
  // recorded it, possibly several co_awaits deep. Resuming the wrong
  // handle a second time does not re-enter that frame; it jumps straight
  // to the awaiting coroutine's OWN resume point and reads a result that
  // was never produced.
  active_handle.resume();
  const bool finished = period_task->done();
  if (!finished) {
    if (!pending_external_request.has_value()) {
      // Invariant violated: Impl::advance() always sets
      // pending_external_request immediately before co_await'ing
      // ExternalSuspend{}, which is the ONLY awaiter anywhere in this
      // file that leaves a resumed Task not done(). Reaching here means
      // some OTHER suspend point was added without updating this
      // contract -- a programmer error, not a runtime one.
      return Failure{"E-RT-RESUME-INVARIANT",
                   "resume_period(): the period coroutine suspended "
                   "without recording an ExternalAdvanceRequest"};
    }
    ExternalAdvanceRequest req = std::move(*pending_external_request);
    pending_external_request.reset();
    return std::optional<ExternalAdvanceRequest>{std::move(req)};
  }
  Result<void> r = period_task->take_result();
  period_task.reset();
  if (!r.ok()) return r.error();
  return std::optional<ExternalAdvanceRequest>{};
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

std::optional<bool> Runner::test_only_window_block_hosted(
    const std::string& channel_name) const {
  for (const detail::RegionRuntime& region : impl_->regions) {
    for (const detail::RegionBlock& b : region.blocks) {
      if (b.channel_name == channel_name) return b.window_hosted;
    }
  }
  return std::nullopt;
}

Result<const double*> Runner::buffer(ArcId arc) const {
  if (!arc.valid() || arc.value() >= impl_->slots.size()) {
    return Failure{"E-RT-ARC", "no buffer for arc " + std::to_string(arc.value())};
  }
  if (impl_->slots[arc.value()].device) {
    return Failure{"E-RT-SPACE",
                 "arc " + std::to_string(arc.value()) +
                     " is resident in device memory (RunnerConfig::execution_space "
                     "== Device); its values reach the host only through the "
                     "component field a Sink delivers to"};
  }
  return static_cast<const double*>(impl_->slots[arc.value()].data);
}

Result<Runner::FieldWindowView> Runner::field_window_view(FieldId field) const {
  if (!field.valid() || field.value() >= impl_->graph.field_count()) {
    return Failure{"E-RT-FIELD", "no such field"};
  }
  ArcId arc;
  const auto exp_it = impl_->window_export_field_arc.find(field.value());
  if (exp_it != impl_->window_export_field_arc.end()) {
    arc = exp_it->second;
  } else {
    const auto imp_it = impl_->window_import_field_arc.find(field.value());
    if (imp_it != impl_->window_import_field_arc.end()) arc = imp_it->second;
  }
  if (!arc.valid()) {
    // Not window-scoped at all -- the ABI layer's own registered storage
    // answers this field, not the Runner.
    return FieldWindowView{};
  }
  if (arc.value() >= impl_->slots.size()) {
    return Failure{"E-RT-ARC", "window arc has no resolved buffer"};
  }
  const detail::Slot& slot = impl_->slots[arc.value()];
  const int field_components = impl_->graph.field(field).components;
  const int arc_components = impl_->dag.arc(arc).type.components;
  const std::int32_t window_samples =
      field_components > 0
          ? static_cast<std::int32_t>(arc_components / field_components)
          : 0;
  return FieldWindowView{static_cast<const double*>(slot.data),
                         static_cast<std::uint64_t>(slot.total()),
                         window_samples};
}

bool Runner::hosts_component(ComponentId component) const {
  // The IDENTICAL formula build_ticks()'s own `hosted[]` and Unit 2b's
  // own `WindowMemberConfig::hosted` already use, read from the SAME
  // `this_process` (set once, in build_ticks(), called from
  // initialize() -- std::nullopt beforehand, which this treats the same
  // as "unset": every component hosted, exactly as every existing
  // caller that never configures a per-rank identity already expects).
  // An invalid or out-of-range id has nothing to disagree about hosting
  // with the graph, so it defaults to `true` rather than refusing --
  // the caller's own validation (E-ABI-RUN-* and friends) is a better
  // place to reject a bad ComponentId than a hosting query is.
  if (!impl_->this_process.has_value()) return true;
  if (!component.valid() ||
      component.value() >= impl_->graph.component_count()) {
    return true;
  }
  return impl_->graph.component(component).ranks.contains(
      *impl_->this_process);
}

}  // namespace fuse
