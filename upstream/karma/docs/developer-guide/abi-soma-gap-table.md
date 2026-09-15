# C ABI correspondence

This guide compares KARMA's candidate provider ABI with SOMA's participant
ABI. It describes integration work that a future bridge would need; it is not
itself an ABI specification.

## Direction and lifecycle

KARMA lets a host select a named implementation and drive a C function table.
SOMA gives a participant `begin_step`/`end_step` calls and a graph-loading
entry point. Existing applications may require both directions, so a bridge
must keep the ownership of the time loop explicit.

KARMA's lifecycle includes version and capability negotiation, component
creation, field-table binding, initialization, advancement over an exact
rational interval, commit, checkpoint, restore, and named diagnostics. SOMA
uses a participant handle, graph descriptors, and a communicator supplied
through its C interface. The records are conceptually related but are not
binary-compatible.

## Data and error contracts

KARMA's field table is C11 and dependency-free. It carries fixed-width sizes,
component counts, scalar type, ownership, and buffer lifetime rules. Status
codes and diagnostics are deterministic and can be reconciled before a later
MPI collective. SOMA's ABI has a different record layout and currently passes
some values, including time, in forms that an adapter must convert to KARMA's
exact rational representation.

The bridge must reject unknown required capability bits, validate every
`struct_size` before reading a record, and prevent exceptions from crossing a
language or rank boundary. A C caller must be able to distinguish invalid
arguments, unsupported versions, unavailable implementations, failed
collective operations, and non-finite field data.

## Graph and configuration

KARMA's current graph is a C++ object for two in-process components. SOMA can
load a graph descriptor at its C boundary. KARMA therefore needs a canonical
configuration record before a C caller can construct arbitrary topology. The
record must sort identifier-keyed containers, use length-prefixed strings and
fixed-width little-endian values, and carry an explicit schema version.

## Field identity and transfers

The KARMA transfer-map and checkpoint formats address entries by a canonical
`FieldEntryKey` containing MOAB identity and the complete algebraic DoF key.
SOMA's compact integer handles cannot replace that key without migration.
Transfer records must also preserve temporal interpretation, ownership,
accumulation, extrapolation, relaxation, and conservation semantics. Ordinary
normalization and L2 projection are not conservative remapping.

## Export and compatibility surface

KARMA exports a small explicit symbol set from `libkarma_abi`; the public
headers include only C standard headers. The ABI remains a candidate until
native, Fortran, and Python components exercise the same call prefix. The
compatibility checker must run before component creation and return all
incompatible properties in deterministic order.

## Current boundary

The implemented runtime is limited to in-process MPI components sharing one
communicator. Dynamic loading, Fortran/Python adapters, arbitrary graph
topology, and conservative remapping are not enabled. This document records
the gaps so an eventual bridge can be implemented without weakening the
existing identity, transaction, and numerical contracts.
