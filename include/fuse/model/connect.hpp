#pragma once

/// The bridge between the two parents' halves of channel checking, and the
/// substance of S2.4.
///
/// The coupler validated a channel with direct comparisons and reported one
/// `E-UNITS` error for any unit disagreement. The spine evaluated a connection in
/// seven graded stages, inferred an operator chain, and distinguished "needs a
/// conversion nobody asked for" from "cannot be expressed at all". This header
/// maps a graph `FieldSpec` onto the spine's `FieldSemantics` so a channel is
/// checked by that framework, and translates its verdict back into the
/// accumulating `Diagnostics` the graph's `validate()` returns.
///
/// **The mapping is deliberately partial and says so.** A `FieldSpec` carries
/// seven properties; `FieldSemantics` carries fourteen. The five axes a graph
/// cannot yet state (continuity, temporal interpretation, ownership, access,
/// update) are left `Unspecified`, which the spine's stages treat as "not
/// claimed" rather than as a default -- an important difference, because a
/// defaulted axis would silently pass a check the user never made. Teaching
/// `FieldSpec` to carry them is the next merge step (it changes the descriptor,
/// so it waits for the waist freeze); until then a channel is checked on what it
/// actually declares.

#include <string>
#include <vector>

#include "fuse/base/result.hpp"
#include "fuse/model/compatibility.hpp"
#include "fuse/model/coupling_graph.hpp"
#include "fuse/model/field_spec.hpp"

namespace fuse {

/// Site policy for a whole graph. One flag today, named rather than passed as a
/// bare bool, because the next two policies (allow lossy operators, require a
/// declared transfer) belong beside it.
struct ConnectPolicy {
  /// An exact, zero-offset unit scale may be inserted without being requested:
  /// kilopascal to pascal is a change of spelling, not of meaning. An OFFSET
  /// change (Celsius to Kelvin) moves the zero of the scale and is never
  /// inserted on the caller's behalf, whatever this says.
  bool allow_automatic_unit_scale = true;
};

/// `FieldSpec` -> `FieldSemantics`, stating only what the graph declared.
[[nodiscard]] inline FieldSemantics to_semantics(const FieldSpec& f) {
  FieldSemantics s;
  s.id = f.name;
  s.units = f.units;
  s.components = f.components;
  s.shape = f.components == 1   ? SemanticShape::Scalar
            : f.components == 9 ? SemanticShape::Tensor2
                                : SemanticShape::Vector;
  // Discretization is where a value LIVES, which is exactly the spine's
  // `location` axis; the two vocabularies agree on the cases both have.
  switch (f.disc) {
    // A finite-volume value lives on the element; a GLL value is a finite-element
    // coefficient, NOT a vertex value -- they coincide only at order 1, and
    // calling a high-order coefficient a vertex value is exactly the confusion
    // that makes a transfer silently wrong.
    case Discretization::FV:               s.location = SemanticLocation::Element; break;
    case Discretization::ContinuousGLL:
    case Discretization::DiscontinuousGLL:
      s.location = f.order <= 1 ? SemanticLocation::Vertex
                                : SemanticLocation::FiniteElementCoefficient;
      break;
    // A point cloud is not any of the three; saying nothing is correct.
    case Discretization::PointCloud:       s.location = SemanticLocation::Unspecified; break;
  }
  // Continuity IS implied by the discretization for the GLL cases, and only
  // those: a finite-volume field is neither, and a point cloud says nothing.
  if (f.disc == Discretization::ContinuousGLL) s.continuity = SemanticContinuity::Continuous;
  if (f.disc == Discretization::DiscontinuousGLL) s.continuity = SemanticContinuity::Discontinuous;
  s.role = f.intent == Intent::Requires ? SemanticRole::Input : SemanticRole::Prognostic;
  return s;
}

/// What the graph requires of a channel, independent of what any transfer
/// happens to provide. An extensive quantity must be conserved globally; an
/// intensive one carries no conservation claim.
[[nodiscard]] inline ConservationGuarantee required_conservation(const FieldSpec& target) {
  return target.conservation == Conservation::Extensive ? ConservationGuarantee::GlobalConservative
                                                        : ConservationGuarantee::None;
}

/// Review 2026-10-11, BM-1: the transfer a channel actually declares, read
/// from its own `RemapSpec` rather than assumed. `RemapFlags::Conservative`
/// (`ChannelBuilder::conservative()`) is the one statement this layer can
/// trust as globally conservative -- it is the authored claim the project's
/// own conservation tests (`Compatibility.ConservationRules`) exist to hold a
/// channel to, not a property this header may infer from the mesh. Any other
/// declared remap, or a channel whose target field lives in a different
/// region than its source -- `lower()`'s own `region_changes` condition
/// (src/ir/rules_channel.cpp) emits a real spatial transfer for exactly that
/// case even with no remap flags at all -- moves values between locations
/// without conserving by construction. Only when neither holds is the
/// channel an exact copy, which is where `TransferSemantics`'s own defaults
/// (`Copy`, `None`) already point.
[[nodiscard]] inline TransferSemantics declared_transfer(const CouplingGraph& g, const Channel& ch) {
  TransferSemantics transfer;
  if (has(ch.remap.flags, RemapFlags::Conservative)) {
    transfer.kind = TransferKind::ConservativeRemap;
    transfer.conservation = ConservationGuarantee::GlobalConservative;
    return transfer;
  }
  const bool region_changes = g.field(ch.source).region != g.field(ch.target).region;
  if (ch.remap.flags != RemapFlags::None || region_changes) {
    transfer.kind = TransferKind::Interpolation;
    return transfer;  // conservation stays at TransferSemantics's own None default
  }
  return transfer;  // Copy, None -- the struct's own defaults
}

/// `ConservationGuarantee` has no stringifier of its own (`semantics.hpp`
/// declares the enum but not a `to_cstring`); `validate()` needs one to name
/// what a channel required against what its declared transfer provided.
[[nodiscard]] inline std::string_view conservation_guarantee_name(ConservationGuarantee g) noexcept {
  switch (g) {
    case ConservationGuarantee::None: return "no conservation guarantee";
    case ConservationGuarantee::LocalConservative: return "a local conservation guarantee";
    case ConservationGuarantee::GlobalConservative: return "a global conservation guarantee";
  }
  return "an unrecognized conservation guarantee";
}

[[nodiscard]] inline std::string_view transfer_kind_name(TransferKind k) noexcept {
  switch (k) {
    case TransferKind::Copy: return "Copy";
    case TransferKind::Interpolation: return "Interpolation";
    case TransferKind::ConservativeRemap: return "ConservativeRemap";
  }
  return "an unrecognized transfer kind";
}

/// Evaluate one channel through the spine's seven stages.
[[nodiscard]] inline ConnectionCompatibility check_channel(const CouplingGraph& g, ChannelId c,
                                                           const ConnectPolicy& policy = {}) {
  const Channel& ch = g.channel_at(c);
  ConnectionRequest request;
  request.source = to_semantics(g.field(ch.source));
  request.target = to_semantics(g.field(ch.target));
  request.required_conservation = required_conservation(g.field(ch.target));
  request.allow_automatic_unit_scale = policy.allow_automatic_unit_scale;
  // BM-1 (review 2026-10-11): this used to hardcode `TransferKind::Copy`
  // unconditionally, which made the Conservation stage below refuse EVERY
  // channel into a `Conservation::Extensive` field (no declared transfer can
  // ever satisfy `required_conservation`'s `GlobalConservative`) and made the
  // Location stage refuse every region-changing channel with no remap flags
  // (`declared_transfer` there is this same `request.transfer.kind != Copy`
  // test) -- see test/verification/interface_transfer_test.cpp's own account
  // of the second effect. Deriving it from the channel's own declaration
  // lets a channel that really does name a conservative transfer pass, and
  // lets a plain copy or an undeclared-conservation remap still be refused.
  request.transfer = declared_transfer(g, ch);
  return check_connection(request);
}

}  // namespace fuse
