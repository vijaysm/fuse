# SOMA correspondence baseline

This document records the interface correspondence between KARMA's native
coupling contracts and the adjacent SOMA coupler. It is a technical reference
for maintainers deciding whether a future capability can share an interface;
it does not define runtime behavior by itself. The source comparison was read
on 2026-09-11 from SOMA commit `400c46397cf90102dd4ae910ba02e9e5f1698dac` on
branch `soma-v1-implementation`. Re-read the source before relying on a row.

## Exact time

Both libraries represent time as a normalized rational with a positive
denominator and use 128-bit intermediates for comparison and arithmetic. KARMA
returns an explicit overflow result; SOMA throws, so an adapter must translate
the error at the component boundary. KARMA also retains bounded dyadic helpers
for legacy floating-point callers. The two libraries agree on interval and
time-slab meaning.

## Field semantics

The seven SI unit exponents are directly compatible. Both models describe the
field location and representation, although SOMA combines several KARMA
properties into its discretization record. KARMA carries exact affine scale
and offset, shape, role, ownership, access, update policy, and opaque metadata;
SOMA's descriptor lacks some of those fields or places them on a channel. An
adapter must preserve these distinctions instead of silently narrowing them.

Temporal semantics are complementary: KARMA describes whether a value is an
instant, interval average, or interval integral, while SOMA identifies the
time anchor and fixed-point iterate. A shared descriptor needs both.

## Component, transform, and transfer records

KARMA ports identify a direction, field semantics, version interval, and
required capability bits. SOMA channels express endpoint intent and richer
remap, relaxation, extrapolation, and space-time records. The concepts overlap
but are not interchangeable. In particular, KARMA refuses an implicit affine
conversion or transfer operator unless it was authorized, while SOMA's channel
specification is the explicit source of those operations.

KARMA stores conservation and accumulation on transfer semantics and field
semantics respectively; SOMA puts the corresponding properties on the field
and channel. A bridge must select one authoritative location and validate that
the other representation agrees.

## Serialization and identity

KARMA sorts identifier-keyed containers before canonical serialization and
uses length-prefixed strings, fixed-width little-endian integers, and an
FNV-1a hash. SOMA's descriptor preserves declaration order and does not expose
a canonical graph hash. A cross-runtime identity therefore requires KARMA's
sort rule or an equivalent normalization before comparing descriptors.

The existing KARMA transfer-map and checkpoint formats use the canonical
`FieldEntryKey` encoding. SOMA's compact integer handles cannot replace that
identity without a file-format migration; a bridge must retain the MOAB global
identity and the complete algebraic DoF key.

## ABI direction

KARMA's candidate C ABI is provider-oriented: the host asks a registry for a
component and drives its table. SOMA's participant-oriented ABI exposes a
component-owned loop with `begin_step` and `end_step`, graph loading, and a
Fortran communicator handle. Existing applications may need both directions.
Neither ABI is a subset of the other, so the adapter should keep the call
direction explicit and negotiate a shared record version before exchanging
data.

## Implementation implications

The current KARMA runtime intentionally supports in-process MPI components on
one communicator. General graph ownership, dynamic loading, foreign-language
adapters, canonical configuration, conservative remapping, and richer
transport remain separate capabilities. Any future bridge should preserve
collective failure reconciliation, exact rational time, deterministic identity,
buffer lifetime rules, and explicit conservation diagnostics.
