#pragma once

/// Rank-to-device binding (docs/dev/device-resident-arcs.md Phase 3): which
/// Kokkos device id a process should bind to, decided BEFORE
/// Kokkos::initialize()/Kokkos::ScopeGuard construction -- Kokkos has no
/// API to rebind a device after that point.
///
/// Header-only and framework-free on purpose: `resolve_device_id()` below
/// is pure arithmetic, so a test can assert the binding rule with a mocked
/// device count as well as the real one. higgs (this track's host) has
/// exactly one GPU, so nothing running there can exercise
/// `local_rank % device_count > 1` against a real device backend; the
/// mock is how that branch gets covered at all.

#include <cstddef>
#include <optional>

namespace fuse::abi {

/// `explicit_override`, if set, wins outright -- the caller reads it from
/// an explicit `--kokkos-device-id=N` argument or the `KOKKOS_DEVICE_ID`
/// environment variable before calling this function; this function never
/// touches either itself, so it stays testable with a plain int. Absent an
/// override, the answer is this process's local rank within its node,
/// modulo how many devices this build's Kokkos backend reports visible.
/// `device_count <= 0` (no device backend, or a probe that found nothing)
/// returns device id 0 -- a defined answer rather than a division by
/// zero -- but the real call sites never pass a non-null device id to
/// Kokkos in that case at all (see `resolve_device_id_for_this_process`
/// at the two call sites in src/abi/participant.cpp and
/// test/support/fuse_test_main.cpp), so 0 from this branch is never
/// actually handed to `InitializationSettings::set_device_id`.
[[nodiscard]] inline int resolve_device_id(
    int local_rank, int device_count,
    std::optional<int> explicit_override) noexcept {
  if (explicit_override.has_value()) return *explicit_override;
  if (device_count <= 0) return 0;
  if (local_rank < 0) local_rank = 0;
  return local_rank % device_count;
}

}  // namespace fuse::abi
