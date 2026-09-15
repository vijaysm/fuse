# Semantic model mapping to the SOMA descriptor

This document maps every record of `karma_semantics.h` onto SOMA's graph
descriptor (`coupe/modules/coupler`). It is a technical correspondence record
for future adapters; runtime behavior remains defined by KARMA's own headers.

**Baseline read:** SOMA `HEAD 9902f82`, branch `soma-v1-implementation`,
2026-09-09. SOMA is under active development and moved commit during the session
in which this was written, so re-read before relying on any row. Every row
carries a disposition; no row is left unmarked.

Dispositions: **same** (equivalent field, same meaning - a port is a rename);
**same in substance** (both express the fact with a different spelling);
**narrower** (SOMA carries something that cannot express every value this model
can); **absent** (no counterpart; a port must add it); **different axis** (each
side carries a fact the other does not); **local** (specific to KARMA's oracle
or storage and not expected to survive a port).

## Exact time

| This model | SOMA | Disposition | Consequence |
| --- | --- | --- | --- |
| `Rational{int64 numerator, int64 denominator}`, normalized, `denominator > 0` | `soma::Rational` (`base/rational.hpp`), normalized, `den_ > 0` | same | Both use exact rational time and neither uses `double`. This is the strongest single argument that a port is feasible. |
| `rational_compare` forms cross products in 128 bits and never narrows | `operator<=>` compares 128-bit products and deliberately never reduces | same, **verified** | Neither side has the defect our `compare-in-64-bits` mutant reproduces. |
| Arithmetic in 128-bit intermediates, reduced, then narrowed | Same strategy, documented in the header | same, **verified** | Neither wraps. |
| Overflow returns `RationalError::Overflow` in a `RationalResult` | Overflow after reduction throws `std::logic_error` | **different error model** | A port must reconcile this with the register's error-model ruling. KARMA's coupling paths must not let an exception cross a rank boundary, so a thrown overflow needs wrapping at the epoch boundary. |
| `rational_from_dyadic` / `rational_from_dyadic_bounded` | none | absent | Legacy `double` interoperation is a KARMA-only problem: SOMA never had a `double` time to migrate from. Keep both on either outcome. |
| `ExactInterval{begin, end}` | time slab / `Window` | same in substance | Slab and interval agree in meaning. |

## Field semantics

| This model | SOMA `FieldSpec` | Disposition | Consequence |
| --- | --- | --- | --- |
| `id` | `name` | same | Rename. |
| `version` (`SemanticVersion`) | none per field; the descriptor carries `kGraphVersion = 2` for the whole graph | absent | Per-field compatibility negotiation has no counterpart. A port either adds per-field versioning or gives up per-field negotiation. |
| `units` as seven SI exponents | `soma::Units` with the same seven `int8_t` exponents (length, mass, time, current, temperature, amount, luminosity) | **same** | This row was expected to be the hard one and is not: the dimension vectors agree field for field. Dimensional checking ports directly. |
| `units.scale` and `units.offset` as exact rationals | none | absent | SOMA compares dimensions but carries no affine conversion, so it cannot express millimetres-to-metres or Celsius-to-Kelvin as data. The transform lane must supply it. |
| `shape` plus `components` | `components` only, documented as "1 scalar, 3 vector, 9 tensor"; `same_shape()` compares only the count | narrower | A 3-vector and a 3-component scalar bundle are indistinguishable. Either add shape or accept the weaker check. |
| `location` plus `continuity` plus `basis_degree` | `disc` (`Discretization`) plus `order` | same in substance | Both express where a value lives and how it is represented; the spellings differ and the mapping is not one-to-one. The FE-coefficient case is the one to check row by row. |
| `role` (prognostic, diagnostic, input, parameter) | `intent` (`Provides`, `Requires`) | narrower | Intent is closer to KARMA's port direction than to role. Diagnostic and parameter have no counterpart. |
| `temporal` (instantaneous committed or trial, interval average, interval integral) | `TimeLevel{anchor, Rational offset, int iterate}` on every DAG arc | **different axis** | SOMA says *when* a value lives, including which fixed-point iterate (`iterate = -1` is the previous iterate). KARMA says how the value is *interpreted* over an interval. Neither subsumes the other and a port must carry both. SOMA's own handoff records that sub-stepped space-time delivery does not reach the component, which is this axis. |
| `ownership` (owned, ghosted, replicated) | none in the graph; rank sets live on the component declaration | absent | Add, or ghost semantics stay implicit in the transport layer. |
| `access` and `update` | `intent` implies access; accumulation lives on the channel as `AccumulateSpec` | **placement differs** | SOMA makes accumulation a property of the transfer, KARMA a property of the field. The port must pick one home; carrying both invites disagreement. |
| none | `conservation` (`Conservation`) on the field | **placement differs** | The mirror image of the row above: SOMA declares conservation on the field, KARMA on `TransferSemantics`. A remap cannot conserve what the field does not claim, so this is a real decision and not a rename. |
| `metadata` preserved verbatim | none | absent | Add, or unknown vendor metadata is silently dropped by a ported path. |

## Port, component, transform, transfer

| This model | SOMA | Disposition | Consequence |
| --- | --- | --- | --- |
| `PortSemantics{id, direction, field, requirement}` | `Channel` endpoints plus `intent` | same in substance | Rename plus the intent/role reconciliation above. |
| `FieldRequirement` version interval and required capability bits | none | absent | This is where "reject unknown required bits" lands per port. Without it, negotiation is descriptor-global. |
| `ComponentSemantics.supports_fractional_steps` | `ComponentDecl` plus `Window{CouplingPeriod, FromComponent, Explicit}` | same in substance, **verify per value** | `Window::Kind` expresses who owns the step, which is the same question; confirm the values line up before mapping. |
| `ComponentSemantics.supports_checkpoint_restore` | derived checkpoint sets computed from the schedule | narrower | SOMA derives checkpoint requirements rather than having components declare them. Both are defensible; they are not the same fact, and the derived form cannot express a component that simply cannot roll back. |
| `requires_communicator` | communicator is set externally through the C ABI | same in substance | Equivalent by convention. |
| `TransformSemantics` with `explicitly_requested` | `RemapSpec`, `RelaxSpec`, `ExtrapolateSpec`, `SpacetimeSpec` on the channel | narrower | SOMA has richer transfer specifications and no counterpart to the rule that only exact zero-offset unit scaling may be inserted automatically. Add the rule or implicit transforms creep back in. |
| `TransferSemantics.conservation` | `RemapSpec.flags` plus field-level `Conservation` | **placement differs** | See the conservation row above. |
| `ScheduleNodeSemantics` | schedule tree node | same in substance | Rename. |
| Canonical serialization: containers sorted by identifier | descriptor writes containers in index order with no sort | **verified difference** | SOMA's descriptor bytes, and therefore any hash over them, depend on declaration order. Two graphs identical in meaning but built in a different order do not compare equal. This is the defect our `ports-unsorted` mutant reproduces; it is a real behavior in SOMA today, and a port must either sort or accept order-sensitive identity. |
| Length-prefixed strings, fixed-width little-endian integers | `pod(uint32 size)` followed by bytes for every string | same, **verified** | Neither side has the unprefixed-concatenation defect our `strings-unprefixed` mutant reproduces. |
| `semantic_hash` (FNV-1a over canonical bytes) | magic `0x534F4D41` and `kGraphVersion` in the descriptor; no canonical hash found | absent | A distributed lookup key must agree bit-for-bit across ranks. Adding one over an order-sensitive serialization would be a mistake. |

## KARMA-local records a port does not carry

| This model | Disposition | Note |
| --- | --- | --- |
| `field_semantics_from_spec`, `port_semantics_from_spec`, `component_semantics_from_spec` | local | Adapters from the existing specification records, which the oracle pins. They keep the oracle unmodified and disappear when the oracle does. |
| `exact_time` 32-bit denominator bound | local | An artifact of migrating from `double` positions, which SOMA never had. |
| `karma.units.unrecognized` metadata key | local | Preserves an unrecognized free-text unit literal so the loss stays visible. |
| `FieldEntryKey` canonical 274-byte encoding | local identity constraint | Persisted transfer maps and safe-point checkpoints are addressed by it. Any adopted identity model must carry a MOAB global identity and a 32-word algebraic DoF key without narrowing, and must preserve the sort order the compatibility oracle pins. SOMA's `base/ids.hpp` uses small integer handles, which is narrower; the adapter cost is a migration of existing files, not a rename. |

## Summary

Time and units port almost for free: the rational clock agrees down to the
128-bit comparison strategy, and the SI dimension vectors are the same seven
`int8_t` exponents. Two rows are real design decisions rather than renames,
because each side put the same fact in a different place: conservation
(SOMA on the field, KARMA on the transfer) and accumulation (SOMA on the
channel, KARMA on the field).

One row is a genuine mutual gap. SOMA's `TimeLevel` carries anchor, rational
offset, and fixed-point iterate - *when* a value lives - while KARMA's
`temporal` carries how a value is interpreted over an interval. A ported model
needs both, and SOMA's own recorded gap in sub-stepped delivery sits on this
axis.

What is absent on the SOMA side is everything that makes a field's meaning
negotiable per field rather than per graph: per-field versioning, required
capability bits, affine unit conversion, role beyond provides/requires,
ownership, and verbatim metadata preservation. Those must be added there under a
port.

Finally, one verified defect argues against porting SOMA's descriptor as the
identity carrier without change: it serializes containers in declaration order,
so descriptor identity is order-sensitive. Two graphs with identical meaning
compare unequal if their components were declared in a different order.
