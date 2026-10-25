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
/// reading its input on the target placement, sums nothing. Measured on a host run:
/// H exact, H^T D delivering zero to every beam vertex, both
/// "correct" by their own bookkeeping.
///
/// So the lowering asks this ONE question and, when the answer is yes, emits
/// a Remap whose output type changes `rank_set` and no Transport; the runtime
/// (`Runner::Impl::resolve_remap`) asks the same question to pick the
/// distributed operator. One predicate, two callers, so they cannot disagree
/// -- the same argument RunnerConfig::base_communicator's comment makes for
/// reading the placement from `ComponentDecl::ranks` in both places.

#include <string_view>

#include "fuse/model/channel.hpp"
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

/// Whether it is even POSSIBLE for a channel with these flags, these two
/// fields, and this remap method to emit a Remap operator: explicit flags,
/// differing regions, or a method that moves placement itself (above)
/// whenever its two ends' placements differ.
///
/// The coupler-placement shared gate. Deliberately independent of any
/// particular placement -- the channel's own source/target ranks, or a
/// declared `RemapSpec::remap_on` -- because flags and region don't depend
/// on ranks, and a non-moving method never moves ranks by itself. If this is
/// false, no choice of `remap_on` could ever change what this channel
/// lowers to, which is exactly what `validate()`'s `W-CHANNEL-REMAP-ON-
/// UNUSED` reports and what `src/ir/rules_channel.cpp` uses to decide
/// whether a declared `remap_on` is honoured at all. One predicate, two
/// callers, so a channel the warning calls unused cannot be one whose
/// lowering silently routes through the coupler anyway.
/// True for a remap method that is a consistent (pointwise) interpolation:
/// target values are interpolated from the source, which reproduces fields
/// in the method's approximation space exactly but does NOT preserve the
/// field's integral. "consistent" (nodal interpolation, the matrix H) and
/// "mls" / "mls:..." (moving least squares) are. "consistent-transpose"
/// (H^T, which conserves the total load by construction), "conservative",
/// "intx", "tempest" and the default (empty: the supermesh conservative
/// remap) are not. Compilation emits no conservation check after a
/// pointwise interpolation: there is no conserved integral to compare, and
/// for a nodal source the two "integrals" would not even be the same
/// quantity (a vertex sum against a cell integral).
[[nodiscard]] constexpr bool is_pointwise_interpolation(std::string_view method) noexcept {
  return method == "consistent" || method == "mls" || method.substr(0, 4) == "mls:";
}

[[nodiscard]] constexpr bool could_ever_emit_remap(RemapFlags flags,
                                                   const FieldSpec& source,
                                                   const FieldSpec& target,
                                                   std::string_view method) noexcept {
  if (flags != RemapFlags::None) return true;
  if (!(source.region == target.region)) return true;
  return remap_moves_placement(method, source, target);
}

}  // namespace fuse
