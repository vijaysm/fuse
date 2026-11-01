// fuse_physics::fe::DofMap -- adapted from KARMA's karma_dof_map.h/.C.
// See dof_map.hpp's own header comment for how this port's identity and
// numbering diverge from KARMA's own MOAB/PETSc-based construction.

#include "fuse_physics/fe/dof_map.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
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
};

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

  if (association.entity_dimension == 0) {
    const auto& connectivity = mesh.element_connectivity(element);
    if (association.local_entity_index >= connectivity.size())
      refuse("DofMap: vertex DofAssociation local_entity_index out of range");
    const spatial::EntityHandle vertex = connectivity[association.local_entity_index];
    key.corners = {mesh.vertex_global_id(vertex)};
    owned = mesh.is_vertex_owned(vertex);
  } else if (association.entity_dimension == top_dim) {
    key.corners = {mesh.element_global_id(element)};
    owned = mesh.is_element_owned(element);
  } else if (association.entity_dimension + 1 == top_dim) {
    const unsigned side = association.local_entity_index;
    key.corners = mesh.canonical_side_vertex_ids(element, side);  // sorted -- identity only, see dof_map.hpp
    const auto local_corners = mesh.side_vertices(element, side);
    owned = !local_corners.empty();
    for (const auto corner : local_corners) owned = owned && mesh.is_vertex_owned(corner);
  } else {
    refuse(
        "DofMap: a shared sub-entity strictly between a vertex and a codimension-1 side (an "
        "interior edge of a 3-D cell, e.g. a Tet10/Hex27 edge midpoint) has no corner-vertex "
        "identity this port exposes (E-PHYSICS-DOFMAP-SUBENTITY-UNSUPPORTED); only vertex dofs, "
        "codimension-1 face/edge dofs, and cell-interior dofs are numbered this wave");
  }
  return {std::move(key), owned};
}

}  // namespace

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
  std::map<DofKey, bool> local_keys;  // true = owned here (dof_map.hpp's own ownership convention)
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
            local_keys.emplace(std::move(classified.key), classified.owned);
          } else if (found->second != classified.owned) {
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
  for (auto& [key, owned] : local_keys) {
    if (owned) owned_keys.push_back(key);
    else ghost_keys.push_back(key);
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

  // ---- Pass 3: ghost resolution, one more allgather --------------------
  // Fixed-width record, each field an EXACT bit-for-bit transport of one
  // std::uint64_t/std::int64_t through std::bit_cast<double> (never a
  // numeric double conversion, which would lose bits above 2^53) --
  // KARMA's own coordinate_bits() trick (karma_dof_map.C), ported for the
  // same reason: Reducer::allgather's channel is doubles, the payload
  // here is integers that must survive the round trip exactly. A
  // STATED, non-hidden scalability limitation (open_questions): this
  // broadcasts every rank's own owned-dof table to every other rank
  // (O(global_dof_count) memory and communication per rank), rather than
  // a point-to-point exchange that would send a rank only the ghost
  // records it actually needs (the scale `fuse::backend`'s own
  // native_halo.hpp pattern reaches for); correct, not scalable, named
  // as a follow-up rather than built this round.
  constexpr std::size_t kMaxCorners = 4;  // the widest built-in sub-entity is a quadrilateral face/side
  constexpr std::size_t kRecordWidth = 4 + kMaxCorners + 2;  // variable,component,entity_dim,n_corners + corners + slot,index
  constexpr std::uint64_t kSentinelVariable = ~std::uint64_t{0};

  const auto pack_u64 = [](std::uint64_t value) { return std::bit_cast<double>(value); };
  const auto unpack_u64 = [](double word) { return std::bit_cast<std::uint64_t>(word); };

  const auto max_owned =
      static_cast<std::size_t>(std::llround(reducer->max(static_cast<double>(n_owned_here))));

  std::vector<double> send(max_owned * kRecordWidth, pack_u64(0));
  for (std::size_t i = 0; i < owned_keys.size(); ++i) send[i * kRecordWidth] = pack_u64(kSentinelVariable);
  for (std::size_t i = 0; i < owned_keys.size(); ++i) {
    const auto& key = owned_keys[i];
    if (key.corners.size() > kMaxCorners)
      refuse(
          "DofMap: a dof key with more than 4 corner vertices cannot be packed for ghost "
          "resolution (E-PHYSICS-DOFMAP-KEY-TOO-WIDE) -- no built-in basis this wave places a dof "
          "on a sub-entity with more than 4 corners (a quadrilateral face)");
    double* record = &send[i * kRecordWidth];
    record[0] = pack_u64(key.variable);
    record[1] = pack_u64(key.component);
    record[2] = pack_u64(key.entity_dimension);
    record[3] = pack_u64(key.corners.size());
    for (std::size_t c = 0; c < kMaxCorners; ++c)
      record[4 + c] = pack_u64(c < key.corners.size() ? key.corners[c] : std::uint64_t{0});
    record[4 + kMaxCorners] = pack_u64(key.slot);
    record[4 + kMaxCorners + 1] =
        pack_u64(static_cast<std::uint64_t>(first_owned_ + static_cast<std::int64_t>(i)));
  }

  std::vector<double> recv(max_owned * kRecordWidth * static_cast<std::size_t>(n_ranks), 0.0);
  reducer->allgather(send.data(), max_owned * kRecordWidth, recv.data());

  std::map<DofKey, std::int64_t> global_table;
  const std::size_t n_slots = max_owned * static_cast<std::size_t>(n_ranks);
  for (std::size_t slot = 0; slot < n_slots; ++slot) {
    const double* record = &recv[slot * kRecordWidth];
    const auto variable_word = unpack_u64(record[0]);
    if (variable_word == kSentinelVariable) continue;
    DofKey key;
    key.variable = static_cast<unsigned>(variable_word);
    key.component = static_cast<unsigned>(unpack_u64(record[1]));
    key.entity_dimension = static_cast<unsigned>(unpack_u64(record[2]));
    const auto n_corners = static_cast<std::size_t>(unpack_u64(record[3]));
    key.corners.resize(n_corners);
    for (std::size_t c = 0; c < n_corners; ++c) key.corners[c] = unpack_u64(record[4 + c]);
    key.slot = static_cast<unsigned>(unpack_u64(record[4 + kMaxCorners]));
    const auto global_index = static_cast<std::int64_t>(unpack_u64(record[4 + kMaxCorners + 1]));
    global_table.emplace(std::move(key), global_index);
  }

  for (const auto& key : ghost_keys) {
    const auto found = global_table.find(key);
    if (found == global_table.end())
      refuse(
          "DofMap: a dof this rank sees as a ghost was not reported as OWNED by any rank "
          "(E-PHYSICS-DOFMAP-GHOST-UNRESOLVED) -- the sub-entity-ownership convention "
          "(dof_map.hpp's own header comment) leaves this one unowned anywhere, the partition "
          "topology that convention's own known edge case names");
    key_to_global_.emplace(key, found->second);
    ghost_indices_.push_back(found->second);
    local_to_global_.push_back(found->second);
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
