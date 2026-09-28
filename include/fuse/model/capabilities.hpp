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
///   bits 4-63   reserved. Must be zero; no current caller sets or checks
///               them. A future bit here is additive (an old binary reading
///               a newer one sees zero and treats it as absent), so no
///               version bump accompanies adding one -- the descriptor
///               version note on this exact point lives in
///               fuse/model/descriptor.hpp.
///
/// DEFAULT (`capabilities() == 0`, i.e. no override at all): no rollback
/// (correct -- see `Component::capabilities()`'s own doc comment for why),
/// `DeterminismLevel::Bitwise` (the determinism field's zero bit pattern),
/// no hidden state. Bitwise is an ASSUMPTION for a component that says
/// nothing, not a proof: it is the correct default for the overwhelming
/// common case FUSE already ships (hand-written numerical components with
/// no RNG and no external nondeterministic dependency), and it is also the
/// ONLY default that keeps every existing `capabilities()` override --
/// which sets `kCapabilityRollback` alone and leaves the determinism field
/// at zero -- describing the same component it described before this file
/// existed. A component that is genuinely nondeterministic (wraps an RNG,
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

/// Bits 4-63: reserved, must be zero. Named so a future bit is added by
/// narrowing this constant, not by grepping for a magic number.
inline constexpr std::uint64_t kCapabilityReservedMask =
    ~(kCapabilityRollback | kCapabilityDeterminismMask | kCapabilityHiddenState);

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
