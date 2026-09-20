#pragma once

/// Every MPI tag the library uses, in one place.
///
/// They are collected here because they are not independent: a tag is only
/// meaningful relative to a communicator, and several of these are used on the
/// *base* communicator the caller supplies. Two mechanisms picking the same
/// value there can match each other's messages, which appears as a wrong
/// result or a hang and not as an error.
///
/// That is not hypothetical. `kEpoch` was first chosen as 0x5C13 by taking the
/// crystal router's 0x5C12 and adding one -- which landed exactly on the tag
/// `MPI_Comm_create_group` was already being given on the same communicator.
/// The two do not currently overlap in time (one runs at setup, the other
/// during a step), so nothing failed, but the next mechanism to be added would
/// have had no way to see the clash. Hence this file: a new tag goes here, next
/// to the others, and the collision is visible.
///
/// All values are well under `MPI_TAG_UB`, whose standard minimum is 32767.
/// Anything above that is rejected outright by every implementation -- an
/// earlier version of `kEpoch` packed 'EPOC' into an int (1162891075) and every
/// parallel test failed with "Invalid tag".

namespace fuse::tags {

/// The crystal router's point-to-point exchange, on its own duplicated
/// communicator.
inline constexpr int kCrystalRouter = 0x5C12;

/// `MPI_Comm_create_group` when forming a pair's union communicator, on the
/// caller's base communicator.
inline constexpr int kUnionCommCreate = 0x5C13;

/// The rank-agreement reduction, on the caller's base communicator. Distinct
/// from `kUnionCommCreate` because both use that communicator.
inline constexpr int kEpoch = 0x5C14;

/// An iteration region's per-sweep residual reduction (REVIEW.md R-10), on
/// the caller's base communicator. Distinct from `kEpoch`: both run a
/// reduce-then-broadcast over the same schedule shape, but a region's sweep
/// and a rank-agreement point can be in flight on the same communicator at
/// once (a region's body may itself contain an Epoch), so sharing a tag
/// would let one mechanism's message match the other's receive.
inline constexpr int kIterationRegion = 0x5C15;

/// The largest tag this library uses. Checked once at setup against the
/// implementation's `MPI_TAG_UB`, so an implementation with an unusually low
/// bound is a refusal before the run rather than a failure inside a reduction.
inline constexpr int kMaxUsed = kIterationRegion;

}  // namespace fuse::tags
