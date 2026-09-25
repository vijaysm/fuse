#pragma once

/// Which remaps move data between placements themselves.
///
/// The lowering's default (src/ir/rules_channel.cpp) is "Transport, then
/// Remap on the target's ranks": a remap is rank-local, so the values must be
/// brought to where the operator runs first. That is wrong for the two
/// consistent-interpolation methods on a nodal field once the two components
/// are placed on different ranks. `build_distributed_interp_rows` gathers the
/// nodal side's geometry over the whole communicator and `apply_distributed_H`
/// / `_HT` exchange the VALUES themselves, reading each side where it lives
/// and writing the result where it belongs -- the operator IS the data move.
/// A Transport in front of it delivers the source values onto the target
/// placement's view of the SOURCE mesh, which on a disjoint placement holds
/// no entities: the values land in a zero-length buffer, and the operator,
/// reading its input on the target placement, sums nothing. Host run
/// 2026-09-25: H exact, H^T D delivering zero to every beam vertex, both
/// "correct" by their own bookkeeping.
///
/// So the lowering asks this ONE question and, when the answer is yes, emits
/// a Remap whose output type changes `rank_set` and no Transport; the runtime
/// (`Runner::Impl::resolve_remap`) asks the same question to pick the
/// distributed operator. One predicate, two callers, so they cannot disagree
/// -- the same argument RunnerConfig::base_communicator's comment makes for
/// reading the placement from `ComponentDecl::ranks` in both places.

#include <string_view>

#include "fuse/model/field_spec.hpp"

namespace fuse {

/// A P1/Q1 continuous field: the discretisation whose DOFs are vertex values,
/// which is what H interpolates from and H^T D delivers loads to. Mirrors
/// `consistent_interp_detail::is_nodal`.
[[nodiscard]] constexpr bool is_nodal_field(const FieldSpec& f) noexcept {
  return f.disc == Discretization::ContinuousGLL && f.order <= 1;
}

/// True when `method` names a consistent-interpolation remap whose nodal side
/// matches its orientation: "consistent" interpolates FROM a nodal source,
/// "consistent-transpose" delivers TO a nodal target. Such a remap is built
/// and applied by the distributed operator whenever the two ends' placements
/// are not one and the same rank, and it moves the data itself.
[[nodiscard]] constexpr bool remap_moves_placement(std::string_view method,
                                                   const FieldSpec& source,
                                                   const FieldSpec& target) noexcept {
  return (method == "consistent" && is_nodal_field(source)) ||
         (method == "consistent-transpose" && is_nodal_field(target));
}

}  // namespace fuse
