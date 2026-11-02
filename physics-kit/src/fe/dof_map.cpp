// fuse_physics::fe::DofMap -- adapted from KARMA's karma_dof_map.h/.C.
// See dof_map.hpp's own header comment for how this port's identity and
// numbering diverge from KARMA's own MOAB/PETSc-based construction.

#include "fuse_physics/fe/dof_map.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>

// Explicit, not transitive: the feature macro FUSE_HAVE_MPI is read below
// (same posture as src/field/collective_agreement.cpp's own comment).
#include "fuse/base/config.hpp"

#if FUSE_HAVE_MPI
#include "fuse/linalg/comm.hpp"
#endif

namespace fuse::physics::fe {

namespace {

[[noreturn]] void refuse(const char* message) { throw std::invalid_argument(message); }

unsigned cell_dim(const spatial::Mesh& mesh, spatial::EntityHandle element) {
  return cell_dimension(cell_family(mesh.element_kind(element)));
}

/// The lexicographically smallest of the n rotations and n reflected
/// rotations of `local` -- the GLOBALLY-agreed canonical cyclic order for
/// a shared edge/face's corner GLOBAL ids, ported from KARMA's own
/// canonical_cycle (karma_dof_map.C) applied here to exact integer ids
/// rather than a coordinate-bit identity. Two elements sharing this
/// sub-entity see its corners in cyclic orders that are themselves
/// related by one of these 2n images of each other (a conforming mesh's
/// own guarantee), so both independently compute the IDENTICAL result.
std::vector<spatial::GlobalId> canonical_cyclic_order(const std::vector<spatial::GlobalId>& local) {
  const std::size_t n = local.size();
  std::vector<spatial::GlobalId> best;
  for (int reflect = 0; reflect < 2; ++reflect) {
    std::vector<spatial::GlobalId> base = local;
    if (reflect) std::reverse(base.begin(), base.end());
    for (std::size_t rotation = 0; rotation < n; ++rotation) {
      std::vector<spatial::GlobalId> candidate(n);
      for (std::size_t i = 0; i < n; ++i) candidate[i] = base[(i + rotation) % n];
      if (best.empty() || candidate < best) best = std::move(candidate);
    }
  }
  return best;
}

struct Classified {
  DofKey key;
  bool owned;
  // Meaningful only when !owned: the rank classify() believes owns
  // this key, so DofMap's own ghost
  // resolution (pass 3 below) can route each ghost key's resolution
  // request to this ONE participant instead of broadcasting to every
  // one of them. Left at -1 (never read) when owned, since an owned
  // key is never a ghost-resolution request's origin.
  int candidate_owner = -1;
};

/// The RANK that `shared_sub_entity_owned_here` below would need to own
/// a vertex among `corners` for the sub-entity to be owned HERE: the
/// minimum of `mesh.vertex_owner(corner)` across every corner -- the
/// SAME quantity that function computes internally to decide whether
/// THIS rank is that owner, exposed here under its own name because
/// `classify()` now needs it even when the answer is "no, somebody
/// else" (that "somebody else" is exactly the
/// ghost key's candidate owner, derivable locally from the corner
/// vertices this rank already has handles for -- no global-id lookup
/// or broadcast required). A single corner's own `vertex_owner` is the
/// vertex-dof case's own candidate owner directly, since the minimum
/// of one value is itself.
int min_corner_owner_rank(const spatial::Mesh& mesh, const std::vector<spatial::EntityHandle>& corners) {
  int minimum_owner = std::numeric_limits<int>::max();
  for (const auto corner : corners) minimum_owner = std::min(minimum_owner, mesh.vertex_owner(corner));
  return minimum_owner;
}

/// Builds `element`'s (variable_index, component)-th scalar dof's DofKey
/// and this-rank ownership (dof_map.hpp's own header comment states the
/// convention), for the sub-entity `association` names. Refuses
/// (E-PHYSICS-DOFMAP-SUBENTITY-UNSUPPORTED) for a 3-D cell's interior
/// edge (entity_dimension 1, cell dimension 3) -- this port exposes no
/// corner-vertex table for that case (dof_map.hpp's own header comment);
/// every other entity_dimension (vertex, codimension-1 face/edge, cell
/// interior) is supported.
Classified classify(const spatial::Mesh& mesh, spatial::EntityHandle element, unsigned variable_index,
                     unsigned component, const DofAssociation& association) {
  const unsigned top_dim = cell_dim(mesh, element);
  DofKey key;
  key.variable = variable_index;
  key.component = component;
  key.entity_dimension = association.entity_dimension;
  key.slot = association.slot;
  bool owned = false;
  int candidate_owner = -1;

  if (association.entity_dimension == 0) {
    const auto& connectivity = mesh.element_connectivity(element);
    if (association.local_entity_index >= connectivity.size())
      refuse("DofMap: vertex DofAssociation local_entity_index out of range");
    const spatial::EntityHandle vertex = connectivity[association.local_entity_index];
    key.corners = {mesh.vertex_global_id(vertex)};
    owned = mesh.is_vertex_owned(vertex);
    if (!owned) candidate_owner = mesh.vertex_owner(vertex);
  } else if (association.entity_dimension == top_dim) {
    key.corners = {mesh.element_global_id(element)};
    owned = mesh.is_element_owned(element);
  } else if (association.entity_dimension + 1 == top_dim) {
    const unsigned side = association.local_entity_index;
    key.corners = mesh.canonical_side_vertex_ids(element, side);  // sorted -- identity only, see dof_map.hpp
    const auto side_corners = mesh.side_vertices(element, side);
    owned = shared_sub_entity_owned_here(mesh, side_corners);
    if (!owned) candidate_owner = min_corner_owner_rank(mesh, side_corners);
  } else if (association.entity_dimension == 1 && top_dim == 3) {
    // A 3-D cell's own INTERIOR edge (e.g. a
    // Tet10/Hex27 edge midpoint), identified through
    // Mesh::canonical_edge_vertex_ids/edge_vertices
    // (ReferenceTopology's own edge-to-corner table, reference_
    // topology.hpp's own header comment).
    const unsigned edge = association.local_entity_index;
    key.corners = mesh.canonical_edge_vertex_ids(element, edge);  // sorted -- identity only
    const auto edge_corners = mesh.edge_vertices(element, edge);
    owned = shared_sub_entity_owned_here(mesh, edge_corners);
    if (!owned) candidate_owner = min_corner_owner_rank(mesh, edge_corners);
  } else {
    refuse(
        "DofMap: a sub-entity dimension this port does not recognize for a cell of this "
        "dimension (E-PHYSICS-DOFMAP-SUBENTITY-UNSUPPORTED) -- unreachable for the four cell "
        "dimensions (0, 1, 2, 3) this port's ReferenceTopology implements, kept as a named "
        "defensive refusal rather than an unchecked assumption");
  }
  return {std::move(key), owned, candidate_owner};
}

}  // namespace

bool shared_sub_entity_owned_here(const spatial::Mesh& mesh, const std::vector<spatial::EntityHandle>& corners) {
  if (corners.empty()) return false;
  const int minimum_owner = min_corner_owner_rank(mesh, corners);
  for (const auto corner : corners)
    if (mesh.is_vertex_owned(corner) && mesh.vertex_owner(corner) == minimum_owner) return true;
  return false;
}

bool DofKey::operator<(const DofKey& rhs) const noexcept {
  if (variable != rhs.variable) return variable < rhs.variable;
  if (component != rhs.component) return component < rhs.component;
  if (entity_dimension != rhs.entity_dimension) return entity_dimension < rhs.entity_dimension;
  if (corners != rhs.corners) return corners < rhs.corners;
  return slot < rhs.slot;
}

DofMap::DofMap(const spatial::Mesh& mesh, std::vector<Variable> variables, const BasisRegistry& registry,
               std::optional<int> fortran_communicator)
    : variables_(std::move(variables)), registry_(&registry), mesh_(&mesh) {
  if (variables_.empty()) refuse("DofMap: at least one variable is required");
  for (const auto& variable : variables_)
    if (variable.components == 0) refuse("DofMap: a variable must have at least one component");

  // ---- Pass 1: classify every dof touched by an OWNED element ---------
  struct LocalKeyInfo {
    bool owned;           // dof_map.hpp's own ownership convention
    int candidate_owner;  // meaningful only when !owned (Classified's own field, same meaning)
  };
  std::map<DofKey, LocalKeyInfo> local_keys;
  for (spatial::EntityHandle element = 1; element <= mesh.n_elements(); ++element) {
    if (!mesh.is_element_owned(element)) continue;
    const auto cell = cell_family(mesh.element_kind(element));
    for (unsigned vi = 0; vi < variables_.size(); ++vi) {
      const auto& variable = variables_[vi];
      const BasisKey basis_key{cell, variable.family, variable.order, variable.continuity};
      if (!registry.supports(basis_key))
        refuse("DofMap: BasisRegistry has no basis for one variable's (cell, family, order, continuity)");
      const Basis basis = registry.create(basis_key);
      for (const auto& association : basis.dof_associations()) {
        for (unsigned component = 0; component < variable.components; ++component) {
          auto classified = classify(mesh, element, vi, component, association);
          const auto found = local_keys.find(classified.key);
          if (found == local_keys.end()) {
            local_keys.emplace(std::move(classified.key),
                                LocalKeyInfo{classified.owned, classified.candidate_owner});
          } else if (found->second.owned != classified.owned) {
            refuse(
                "DofMap: the same dof key was classified owned on one element and not-owned on "
                "another (E-PHYSICS-DOFMAP-OWNERSHIP-CONFLICT) -- the mesh partition leaves this "
                "sub-entity's ownership ambiguous under this port's convention (dof_map.hpp's own "
                "header comment names the known edge case)");
          }
        }
      }
    }
  }

  std::vector<DofKey> owned_keys;
  std::vector<DofKey> ghost_keys;
  std::vector<int> ghost_candidate_owner;  // parallel to ghost_keys
  for (auto& [key, info] : local_keys) {
    if (info.owned) {
      owned_keys.push_back(key);
    } else {
      ghost_keys.push_back(key);
      ghost_candidate_owner.push_back(info.candidate_owner);
    }
  }
  const auto n_owned_here = static_cast<std::int64_t>(owned_keys.size());

  if (!fortran_communicator.has_value()) {
    if (!ghost_keys.empty())
      refuse(
          "DofMap: elements reference a dof this rank does not own, but no communicator was given "
          "to resolve it (E-PHYSICS-DOFMAP-GHOST-WITHOUT-COMMUNICATOR) -- the single-rank "
          "convenience constructor path assumes every dof an owned element touches is owned here");
    first_owned_ = 0;
    n_owned_ = n_owned_here;
    n_dofs_ = n_owned_here;
    row_lo_by_rank_ = {0};
    for (std::int64_t i = 0; i < n_owned_here; ++i) {
      key_to_global_.emplace(owned_keys[static_cast<std::size_t>(i)], i);
      local_to_global_.push_back(i);
    }
    return;
  }

#if FUSE_HAVE_MPI
  const auto reducer = linalg::make_mpi_reducer(*fortran_communicator);
  const int rank = reducer->rank();
  const int n_ranks = reducer->size();

  // ---- Pass 2: contiguous per-rank offsets, one allgather -------------
  std::vector<double> counts(static_cast<std::size_t>(n_ranks), 0.0);
  {
    const std::array<double, 1> send{static_cast<double>(n_owned_here)};
    reducer->allgather(send.data(), 1, counts.data());
  }
  first_owned_ = 0;
  n_dofs_ = 0;
  row_lo_by_rank_.assign(static_cast<std::size_t>(n_ranks), 0);
  for (int r = 0; r < n_ranks; ++r) {
    row_lo_by_rank_[static_cast<std::size_t>(r)] = n_dofs_;
    const auto count_r = static_cast<std::int64_t>(std::llround(counts[static_cast<std::size_t>(r)]));
    if (r < rank) first_owned_ += count_r;
    n_dofs_ += count_r;
  }
  n_owned_ = n_owned_here;
  for (std::int64_t i = 0; i < n_owned_here; ++i) {
    key_to_global_.emplace(owned_keys[static_cast<std::size_t>(i)], first_owned_ + i);
    local_to_global_.push_back(first_owned_ + i);
  }

  // ---- Pass 3: ghost resolution, point-to-point via two alltoallv's ---
  // Ghost resolution is a point-to-point exchange rather than a padded
  // allgather (every rank's own owned-dof table broadcast to every other
  // rank, O(global_dof_count) memory and communication per rank, and
  // prone to mis-packing when ranks own unequal counts), sized to what each
  // rank actually needs: each ghost key is routed ONLY to its CANDIDATE owner
  // (`ghost_candidate_owner`, computed locally by `classify()` above --
  // the minimum-rank corner owner `shared_sub_entity_owned_here` itself
  // reaches for) rather than resolved against a table every participant
  // was sent. Two round trips through `Reducer::alltoallv` (comm.hpp):
  // REQUEST (every ghost key's identity fields, grouped by candidate
  // owner) then ANSWER (that owner's own `key_to_global_` lookup --
  // already built in pass 2 above for every key this rank owns --
  // routed straight back to the asker), mirroring the two-phase
  // request/answer shape `src/linalg/comm_mpi.cpp`'s own
  // `make_mpi_exchanger` builds once and reuses; this ghost-resolution
  // pass runs the two phases freshly, once, at DofMap construction
  // (not a pattern meant to be replayed every call the way a
  // `GhostExchanger`'s own `update()` is).
  //
  // MESSAGE VOLUME, versus a padded allgather (R ranks, this rank holding
  // `n_ghost` ghost keys of width `kRequestWidth` int64 fields each,
  // `max_owned` the largest owned-key count any rank holds): the
  // allgather version moved `max_owned * kRecordWidth * R` int64
  // values on EVERY rank, independent of how many ghosts anyone
  // actually had; this version moves exactly `n_ghost * kRequestWidth`
  // int64 values out and one int64 answer back per ghost, with no
  // `max()` collective needed to size a shared buffer, and no
  // participant ever touches another rank's unrelated owned keys.
  constexpr std::size_t kMaxCorners = 4;  // the widest built-in sub-entity is a quadrilateral face/side
  constexpr std::size_t kRequestWidth = 4 + kMaxCorners + 1;  // variable,component,entity_dim,n_corners + corners + slot
  constexpr std::int64_t kKeyNotOwnedHere = -1;  // the candidate owner does not, after all, own this key

  std::vector<std::vector<std::size_t>> ghost_by_rank(static_cast<std::size_t>(n_ranks));
  for (std::size_t i = 0; i < ghost_keys.size(); ++i) {
    const int dest = ghost_candidate_owner[i];
    FUSE_ASSERT(dest >= 0 && dest < n_ranks,
               "DofMap: a ghost key's candidate owner rank {} is outside [0, {}) (classify()'s own "
               "min_corner_owner_rank/vertex_owner contract)",
               dest, n_ranks);
    ghost_by_rank[static_cast<std::size_t>(dest)].push_back(i);
  }

  std::vector<int> request_int_counts(static_cast<std::size_t>(n_ranks), 0);
  for (int r = 0; r < n_ranks; ++r)
    request_int_counts[static_cast<std::size_t>(r)] =
        static_cast<int>(ghost_by_rank[static_cast<std::size_t>(r)].size()) * static_cast<int>(kRequestWidth);

  // request_payload[k*kRequestWidth .. (k+1)*kRequestWidth) <-> the
  // ghost key at ghost_keys[request_origin_index[k]], grouped by
  // candidate-owner rank ascending.
  std::vector<std::int64_t> request_payload;
  std::vector<std::size_t> request_origin_index;
  request_payload.reserve(ghost_keys.size() * kRequestWidth);
  request_origin_index.reserve(ghost_keys.size());
  for (int r = 0; r < n_ranks; ++r) {
    for (const auto i : ghost_by_rank[static_cast<std::size_t>(r)]) {
      const auto& key = ghost_keys[i];
      if (key.corners.size() > kMaxCorners)
        refuse(
            "DofMap: a dof key with more than 4 corner vertices cannot be packed for ghost "
            "resolution (E-PHYSICS-DOFMAP-KEY-TOO-WIDE) -- no built-in basis places a dof "
            "on a sub-entity with more than 4 corners (a quadrilateral face)");
      request_payload.push_back(static_cast<std::int64_t>(key.variable));
      request_payload.push_back(static_cast<std::int64_t>(key.component));
      request_payload.push_back(static_cast<std::int64_t>(key.entity_dimension));
      request_payload.push_back(static_cast<std::int64_t>(key.corners.size()));
      for (std::size_t c = 0; c < kMaxCorners; ++c)
        request_payload.push_back(
            static_cast<std::int64_t>(c < key.corners.size() ? key.corners[c] : spatial::GlobalId{0}));
      request_payload.push_back(static_cast<std::int64_t>(key.slot));
      request_origin_index.push_back(i);
    }
  }

  std::vector<int> recv_request_int_counts;
  const auto recv_request_payload = reducer->alltoallv(request_payload, request_int_counts, recv_request_int_counts);
  FUSE_ASSERT(recv_request_payload.size() % kRequestWidth == 0,
             "DofMap: ghost-resolution request payload ({} int64 values) is not a multiple of the "
             "record width ({}) -- every rank must send whole kRequestWidth-wide records",
             recv_request_payload.size(), kRequestWidth);
  const std::size_t n_requests_received = recv_request_payload.size() / kRequestWidth;

  // ANSWER each received request from this rank's OWN key_to_global_
  // (already populated with every key this rank owns, in pass 2 above)
  // -- in the SAME order received, which is grouped by asking rank
  // ascending (Reducer::alltoallv's own recv ordering contract), so the
  // reply below can reuse recv_request_int_counts directly as its own
  // per-destination send counts with no regrouping.
  std::vector<std::int64_t> answer_payload(n_requests_received, kKeyNotOwnedHere);
  for (std::size_t k = 0; k < n_requests_received; ++k) {
    const std::int64_t* record = &recv_request_payload[k * kRequestWidth];
    DofKey key;
    key.variable = static_cast<unsigned>(record[0]);
    key.component = static_cast<unsigned>(record[1]);
    key.entity_dimension = static_cast<unsigned>(record[2]);
    const auto n_corners = static_cast<std::size_t>(record[3]);
    key.corners.resize(n_corners);
    for (std::size_t c = 0; c < n_corners; ++c) key.corners[c] = static_cast<spatial::GlobalId>(record[4 + c]);
    key.slot = static_cast<unsigned>(record[4 + kMaxCorners]);
    const auto found = key_to_global_.find(key);
    if (found != key_to_global_.end()) answer_payload[k] = found->second;
  }

  std::vector<int> answer_int_counts(static_cast<std::size_t>(n_ranks), 0);
  for (int r = 0; r < n_ranks; ++r)
    answer_int_counts[static_cast<std::size_t>(r)] = recv_request_int_counts[static_cast<std::size_t>(r)] /
                                                       static_cast<int>(kRequestWidth);

  std::vector<int> recv_answer_counts;
  const auto recv_answers = reducer->alltoallv(answer_payload, answer_int_counts, recv_answer_counts);
  FUSE_ASSERT(recv_answers.size() == request_origin_index.size(),
             "DofMap: ghost-resolution received {} answers but sent {} requests -- every request must "
             "get exactly one answer back",
             recv_answers.size(), request_origin_index.size());

  // Agree before refusing: an unresolved
  // ghost is a LOCAL finding, and a local refusal here would leave the
  // other ranks numbering a dof map this rank never finishes. One max
  // reduction carries the worst local count to every rank, and every rank
  // then refuses with the same code; the message names how many ghosts
  // went unresolved in total.
  std::size_t unresolved_here = 0;
  for (const auto global_index : recv_answers)
    if (global_index == kKeyNotOwnedHere) ++unresolved_here;
  const double unresolved_total = reducer->sum(static_cast<double>(unresolved_here));
  if (unresolved_total > 0.0) {
    const std::string message =
        "DofMap: " + std::to_string(static_cast<std::size_t>(unresolved_total)) +
        " ghost dof(s) across all ranks were not reported as OWNED by their candidate owner "
        "(E-PHYSICS-DOFMAP-GHOST-UNRESOLVED; this rank's own share: " + std::to_string(unresolved_here) +
        ") -- the sub-entity-ownership convention (dof_map.hpp's own header comment) leaves them "
        "unowned anywhere, the partition topology that convention's own known edge case names";
    refuse(message.c_str());
  }
  for (std::size_t k = 0; k < recv_answers.size(); ++k) {
    const auto global_index = recv_answers[k];
    const auto& key = ghost_keys[request_origin_index[k]];
    key_to_global_.emplace(key, global_index);
    ghost_indices_.push_back(global_index);
    local_to_global_.push_back(global_index);
  }
#else
  refuse(
      "DofMap: a communicator was given but this build has no MPI (FUSE_HAVE_MPI=0); build with "
      "FUSE_ENABLE_MPI=ON to use the collective numbering path (E-PHYSICS-DOFMAP-MPI-REQUIRED)");
#endif
}

void DofMap::dof_indices(spatial::EntityHandle element, std::vector<std::int64_t>& indices) const {
  indices.clear();
  for (unsigned vi = 0; vi < variables_.size(); ++vi) {
    std::vector<std::int64_t> one_variable;
    dof_indices(element, one_variable, vi);
    indices.insert(indices.end(), one_variable.begin(), one_variable.end());
  }
}

void DofMap::dof_indices(spatial::EntityHandle element, std::vector<std::int64_t>& indices,
                          unsigned variable) const {
  indices.clear();
  if (variable >= variables_.size()) refuse("DofMap::dof_indices: variable index out of range");
  const auto cell = cell_family(mesh_->element_kind(element));
  const auto& v = variables_[variable];
  const Basis basis = registry_->create(BasisKey{cell, v.family, v.order, v.continuity});
  for (const auto& association : basis.dof_associations()) {
    for (unsigned component = 0; component < v.components; ++component) {
      const auto classified = classify(*mesh_, element, variable, component, association);
      const auto found = key_to_global_.find(classified.key);
      if (found == key_to_global_.end())
        refuse(
            "DofMap::dof_indices: element references a dof outside this rank's visible set "
            "(E-PHYSICS-DOFMAP-DOF-NOT-VISIBLE)");
      indices.push_back(found->second);
    }
  }
}

std::vector<DofRecord> DofMap::storage(spatial::EntityHandle element, unsigned variable,
                                        const Basis& basis) const {
  if (variable >= variables_.size()) refuse("DofMap::storage: variable index out of range");
  const auto& v = variables_[variable];
  const unsigned top_dim = cell_dim(*mesh_, element);
  std::vector<DofRecord> result;
  for (const auto& association : basis.dof_associations()) {
    for (unsigned component = 0; component < v.components; ++component) {
      const auto classified = classify(*mesh_, element, variable, component, association);
      const auto found = key_to_global_.find(classified.key);
      if (found == key_to_global_.end())
        refuse(
            "DofMap::storage: element references a dof outside this rank's visible set "
            "(E-PHYSICS-DOFMAP-DOF-NOT-VISIBLE)");
      const bool element_local = association.entity_dimension == top_dim;
      result.push_back(DofRecord{found->second, component, association, association.slot, element_local});
    }
  }
  return result;
}

bool DofMap::owns_element_for_assembly(spatial::EntityHandle element) const noexcept {
  return mesh_->is_element_owned(element);
}

unsigned DofMap::orientation_code(spatial::EntityHandle element, unsigned entity_dimension,
                                   unsigned local_entity_index) const {
  const unsigned top_dim = cell_dim(*mesh_, element);
  // A vertex or cell-interior dof is never shared between two elements, so
  // "which local order did the OTHER element see this in" does not arise;
  // 0 is the only code BuiltinBasis::orientation_transforms() ever emits
  // for such a sub-entity (basis.cpp: orientation_count returns 1).
  if (entity_dimension == 0 || entity_dimension == top_dim) return 0;
  if (entity_dimension + 1 != top_dim)
    refuse("DofMap::orientation_code: unsupported entity_dimension (E-PHYSICS-DOFMAP-SUBENTITY-UNSUPPORTED)");

  const auto local_handles = mesh_->side_vertices(element, local_entity_index);
  std::vector<spatial::GlobalId> local_order;
  local_order.reserve(local_handles.size());
  for (const auto handle : local_handles) local_order.push_back(mesh_->vertex_global_id(handle));
  const auto canonical = canonical_cyclic_order(local_order);
  const std::size_t n = local_order.size();

  auto try_codes = [&](auto codes) -> std::optional<unsigned> {
    for (const auto code : codes) {
      const auto permutation = local_to_canonical_vertices(code);
      std::vector<spatial::GlobalId> candidate(permutation.size());
      for (std::size_t i = 0; i < permutation.size(); ++i) candidate[permutation[i]] = local_order[i];
      if (candidate == canonical) return static_cast<unsigned>(code);
    }
    return std::nullopt;
  };

  std::optional<unsigned> code;
  if (n == 2) {
    code = try_codes(std::array{EdgeOrientation::Direct, EdgeOrientation::Reversed});
  } else if (n == 3) {
    code = try_codes(std::array{TriangleOrientation::Rotate0, TriangleOrientation::Rotate1,
                                 TriangleOrientation::Rotate2, TriangleOrientation::ReflectRotate0,
                                 TriangleOrientation::ReflectRotate1, TriangleOrientation::ReflectRotate2});
  } else if (n == 4) {
    code = try_codes(std::array{QuadrilateralOrientation::Rotate0, QuadrilateralOrientation::Rotate1,
                                 QuadrilateralOrientation::Rotate2, QuadrilateralOrientation::Rotate3,
                                 QuadrilateralOrientation::ReflectRotate0, QuadrilateralOrientation::ReflectRotate1,
                                 QuadrilateralOrientation::ReflectRotate2, QuadrilateralOrientation::ReflectRotate3});
  } else {
    refuse("DofMap::orientation_code: a side with neither 2, 3, nor 4 corners has no orientation table");
  }
  if (!code.has_value())
    refuse(
        "DofMap::orientation_code: no orientation code reproduced the canonical cyclic order "
        "(E-PHYSICS-DOFMAP-ORIENTATION-NOT-FOUND) -- unreachable for a conforming mesh, refused "
        "by name rather than silently returning a wrong code");
  return *code;
}

}  // namespace fuse::physics::fe
