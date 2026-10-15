#pragma once

/// KNOWN_ISSUES.md B3. The named bit assignments for the ONE 64-bit
/// capability space `fuse::ComponentSemantics::capabilities`,
/// `fuse::FieldRequirement::required_capabilities` (both fuse/model/
/// semantics.hpp) and `fuse::Component::capabilities()` (fuse/runtime/
/// component.hpp) all carry. Before this file the space existed -- three
/// `std::uint64_t` fields already passed it around -- but nothing named a
/// single bit in it: `test/model/compatibility_test.cpp` sets it with
/// literal patterns like `0b1011` because there was nothing else to set.
/// This file is where a caller who wants to ask "does the far side support
/// rollback" (rather than "are bits 0 and 3 set") gets a name for it.
///
/// WHY ONE HEADER AND NOT TWO. `Component::capabilities()` is a runtime
/// property (a C++ vtable call); `ComponentSemantics::capabilities` is a
/// model-layer declaration serialized into the canonical descriptor
/// (`serialize_component_semantics`, src/model/semantics.cpp). They are
/// different TYPES answering different questions, but B3 asks for the
/// bitmask a component returns to be checked against what a declared
/// `SurrogateContract` (fuse/model/multiscale.hpp) asks about -- which only
/// works if a bit means the same thing on both sides of that comparison.
/// Two separately-numbered spaces could disagree with each other silently;
/// one shared space cannot.
///
/// WHY THIS IS `fuse/model/`, NOT `fuse/runtime/`. `tools/check-layering.py`
/// ranks `model` below `runtime`, so `runtime` may depend on `model` but not
/// the reverse. Putting the constants here lets `component.hpp` (runtime)
/// use them; putting them in `runtime/component.hpp` instead would make
/// `semantics.hpp` (model) reach up for them, which the gate refuses.
///
/// WHY A NEW FILE AND NOT `semantics.hpp` ITSELF. The determinism field's
/// encode/decode helpers need `fuse::DeterminismLevel`
/// (fuse/model/multiscale.hpp). `multiscale.hpp` already includes
/// `semantics.hpp`; having `semantics.hpp` include `multiscale.hpp` back
/// would be a cycle. `capabilities.hpp` sits beside both, in the same
/// layer (`model` may include its own layer, tools/check-layering.py's rank
/// check is `>`, not `>=`), and depends only on `multiscale.hpp`.
///
/// BIT LAYOUT.
///
///   bit 0       kCapabilityRollback     -- `save_state()`/`restore_state()`
///               are a genuine, verifiable pair (see
///               `Component::supports_rollback()`).
///   bits 1-2    the determinism field, `kCapabilityDeterminismMask`
///               (`kCapabilityDeterminismShift` = 1, `kCapabilityDeterminismBits`
///               = 2). 0b00 is UNDECLARED (`decode_capability_determinism()`
///               returns `std::nullopt`); the enumerators are stored at raw
///               value + 1: `Bitwise` = 0b01, `RunToRun` = 0b10,
///               `Nondeterministic` = 0b11. Silence is not a claim -- see
///               `encode_capability_determinism()` for why zero must not
///               mean `Bitwise`.
///   bit 3       kCapabilityHiddenState   -- mirrors
///               `SurrogateContract::carries_hidden_state`: state
///               `advance()` mutates that was never bound through
///               `register_fields()`, so the byte-for-byte rollback
///               verification in `src/runtime/iteration.cpp`
///               (`R.verify_arcs`) cannot see it. NOT itself a refusal --
///               `Component::save_state()`'s own contract already requires
///               an implementer to save such state anyway, FUSE simply
///               cannot check that they did (see the R16 caveat there). The
///               bit exists so that fact can be READ rather than assumed.
///   bit 4       kCapabilityResidual     -- S7.1's residual/linearisation
///               seam: `Component::implicit()` returns a non-null
///               `fuse::ImplicitComponent` (fuse/runtime/component.hpp), so
///               `evaluate_residual()` may be called. Checked, not trusted,
///               the same way `kCapabilityRollback` is: see
///               `fuse::verify_newton_capability_support()`.
///   bit 5       kCapabilityLinearization -- the `LinearizationToken` the
///               `ImplicitComponent` returns after an `evaluate_residual()`
///               call is usable for this component's own block. Independent
///               of bit 4: a component may answer the residual question
///               without ever linearising it.
///   bit 6       kCapabilityRestartFormA -- Restart track (docs/dev/
///               restart.md), Stage B Slice 1, owner decision 1: the
///               component offers `write_restart()`/`read_restart()`
///               (`fuse::Component`, fuse/runtime/component.hpp) keyed by
///               its OWN global ids, re-partitionable across a rank-count
///               change. A component that lives on one rank numbers its
///               own entries `0..n-1` -- trivially satisfiable, so every
///               one of the four worked examples sets this bit.
///   bit 7       kCapabilityRestartFormB -- the component offers
///               `save_restart_blob()`/`load_restart_blob()`, an opaque,
///               versioned per-rank byte record, same-decomposition only.
///               Independent of bit 6: a component may set EITHER, or
///               BOTH (checked at restart-open time, not inferred from one
///               implying the other) -- owner decision 1's "capability"
///               is this one 64-bit space carrying two independent bits,
///               not a three-way enum, so a future third restart form
///               does not have to redefine what the first two already mean.
///               A restart on a different rank count is refused BY NAME,
///               collectively, if ANY component in the graph sets bit 7
///               without bit 6 (owner decision 1).
///   bits 8-63   reserved. Must be zero; no current caller sets or checks
///               them. A future bit here is additive (an old binary reading
///               a newer one sees zero and treats it as absent), so no
///               version bump accompanies adding one -- the descriptor
///               version note on this exact point lives in
///               fuse/model/descriptor.hpp.
///
/// DEFAULT (`capabilities() == 0`, i.e. no override at all): no rollback
/// (correct -- see `Component::capabilities()`'s own doc comment for why),
/// determinism UNDECLARED -- NOT `DeterminismLevel::Bitwise` -- and no
/// hidden state. BM-9 (review 2026-10-11): this paragraph used to read
/// "`DeterminismLevel::Bitwise` (the determinism field's zero bit
/// pattern)", which contradicts this file's own encode/decode pair just
/// above (raw 0 decodes to `std::nullopt`, i.e. undeclared, by design --
/// see `decode_capability_determinism()`'s own comment for why silence
/// must not read as the strongest claim in `DeterminismLevel`), the test
/// that pins it (`Capabilities.ZeroDecodesToUndeclaredNotBitwise`), and
/// `src/runtime/checkpoint_support.cpp`'s own undeclared-with-warning
/// handling. A reader of this paragraph alone would assume silence means
/// `Bitwise` -- the exact assumption KNOWN_ISSUES.md B3 was closed to
/// forbid. A component that is genuinely nondeterministic (wraps an RNG,
/// calls into a library with unspecified reduction order, etc.) MUST set
/// `kCapabilityDeterminismMask` to `encode_capability_determinism(
/// DeterminismLevel::Nondeterministic)` explicitly, or the check in
/// `src/runtime/checkpoint_support.cpp` that refuses a nondeterministic
/// component inside an iteration region will not see it.

#include <cstdint>
#include <optional>

#include "fuse/model/multiscale.hpp"

namespace fuse {

/// `save_state()`/`restore_state()` are a genuine, verifiable pair.
inline constexpr std::uint64_t kCapabilityRollback = std::uint64_t{1} << 0;

/// Where the 2-bit determinism field starts, and how wide it is.
inline constexpr std::uint64_t kCapabilityDeterminismShift = 1;
inline constexpr std::uint64_t kCapabilityDeterminismBits  = 2;

/// `0b110`: bits 1-2.
inline constexpr std::uint64_t kCapabilityDeterminismMask =
    ((std::uint64_t{1} << kCapabilityDeterminismBits) - 1) << kCapabilityDeterminismShift;

/// The component carries state `advance()` mutates that was never bound
/// through `register_fields()`, so FUSE's byte-for-byte rollback
/// verification cannot see it (mirrors `SurrogateContract::carries_hidden_state`).
inline constexpr std::uint64_t kCapabilityHiddenState = std::uint64_t{1} << 3;

/// S7.1's residual/linearisation seam. The component additionally
/// implements `fuse::ImplicitComponent` (fuse/runtime/component.hpp):
/// `Component::implicit()` returns non-null, and `evaluate_residual()` can
/// be called to get this component's own contribution to the coupled
/// residual at a frozen time level without advancing its state. A Newton-
/// solved iteration region whose `IterationRegionDecl::requires_residual_
/// capability` is set refuses a member that does not set this bit
/// (E-NEWTON-CAPABILITY, `fuse::verify_newton_capability_support`,
/// src/runtime/checkpoint_support.cpp) -- the same checked-not-trusted
/// pattern `kCapabilityRollback` already follows for `supports_rollback()`.
inline constexpr std::uint64_t kCapabilityResidual = std::uint64_t{1} << 4;

/// S7.1's residual/linearisation seam. The `LinearizationToken` returned by
/// `ImplicitComponent::linearization()` after an `evaluate_residual()` call
/// is usable for this component's own block -- in particular its
/// `apply_preconditioner()`, which a Newton region's block-Jacobi
/// physics-based preconditioner (S7.2) composes from every member's own
/// token. A region whose `IterationRegionDecl::requires_linearization_
/// capability` is set refuses a member that does not set this bit
/// (E-NEWTON-CAPABILITY). Independent of `kCapabilityResidual`: a component
/// may evaluate its residual without ever linearising it, in which case
/// the Newton coupler falls back to a purely matrix-free directional
/// derivative for that component's own block.
inline constexpr std::uint64_t kCapabilityLinearization = std::uint64_t{1} << 5;

/// Restart track (docs/dev/restart.md), owner decision 1: Form A, keyed by
/// the component's own global ids, re-partitionable. See the file header's
/// BIT LAYOUT entry for bit 6.
inline constexpr std::uint64_t kCapabilityRestartFormA = std::uint64_t{1} << 6;

/// Restart track (docs/dev/restart.md), owner decision 1: Form B, an
/// opaque versioned per-rank blob, same-decomposition only. See the file
/// header's BIT LAYOUT entry for bit 7.
inline constexpr std::uint64_t kCapabilityRestartFormB = std::uint64_t{1} << 7;

/// Bits 8-63: reserved, must be zero. Named so a future bit is added by
/// narrowing this constant, not by grepping for a magic number.
inline constexpr std::uint64_t kCapabilityReservedMask =
    ~(kCapabilityRollback | kCapabilityDeterminismMask | kCapabilityHiddenState |
      kCapabilityResidual | kCapabilityLinearization | kCapabilityRestartFormA |
      kCapabilityRestartFormB);

/// Packs `level` into the 2-bit determinism field, zero elsewhere. The
/// encoding is the enumerator's own raw value, left-shifted into place --
/// `DeterminismLevel::Bitwise` (0) therefore encodes as 0, which is also
/// `capabilities()`'s untouched default; see the file header for why that
/// is deliberate rather than a coincidence of the shift.
/// The 2-bit field's raw value 0 is UNDECLARED, not `Bitwise`. `Bitwise` is
/// the strongest claim in `DeterminismLevel` ("bit-identical ... across the
/// rank counts the coupler may use"), and a component that never mentioned
/// determinism has not made it; reading silence as that claim would let the
/// iteration-region check below pass a component nobody vouched for. So the
/// three enumerators are stored at raw 1, 2, 3 and `decode` returns
/// `std::nullopt` for 0. (The first version stored the enumerator's own
/// value, making 0 mean `Bitwise`; changed the same day, before anything
/// serialised it.)
[[nodiscard]] constexpr std::uint64_t encode_capability_determinism(
    DeterminismLevel level) noexcept {
  return ((static_cast<std::uint64_t>(level) + 1) << kCapabilityDeterminismShift) &
         kCapabilityDeterminismMask;
}

/// The inverse of `encode_capability_determinism()` for the three valid
/// enumerators. The fourth bit pattern the 2-bit field can hold (0b11) is
/// not a valid `DeterminismLevel` and is read as `Nondeterministic` -- the
/// conservative interpretation of a value nothing in this codebase encodes,
/// rather than a `static_cast` past the enum's declared range.
[[nodiscard]] constexpr std::optional<DeterminismLevel> decode_capability_determinism(
    std::uint64_t capabilities) noexcept {
  const std::uint64_t raw =
      (capabilities & kCapabilityDeterminismMask) >> kCapabilityDeterminismShift;
  switch (raw) {
    case 0: return std::nullopt;  // undeclared
    case 1: return DeterminismLevel::Bitwise;
    case 2: return DeterminismLevel::RunToRun;
    default: return DeterminismLevel::Nondeterministic;  // 3
  }
}

/// Convenience for an override site that wants to state all three axes at
/// once: `return fuse::make_capabilities(true, fuse::DeterminismLevel::Bitwise,
/// false);` reads as the three questions `capabilities()`'s doc comment
/// poses, rather than as a hand-assembled bitmask.
[[nodiscard]] constexpr std::uint64_t make_capabilities(
    bool rollback, DeterminismLevel determinism, bool hidden_state) noexcept {
  return (rollback ? kCapabilityRollback : std::uint64_t{0}) |
         encode_capability_determinism(determinism) |
         (hidden_state ? kCapabilityHiddenState : std::uint64_t{0});
}

}  // namespace fuse
