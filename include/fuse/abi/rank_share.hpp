#pragma once

/// `ProviderAdapter::register_fields()` (src/abi/provider_adapter.cpp)
/// needs THIS rank's own share of a region when the region's own component
/// spans MORE THAN ONE rank, not merely "hosted or not" -- see this
/// header's own doc comment on `even_rank_share`.

#include <cstddef>
#include <utility>

namespace fuse::abi_detail {

/// The half-open slice `[lo, hi)` of index range `[0, total)` that
/// `position` owns among `group_size` equally-weighted positions, split
/// as evenly as possible (the first `total % group_size` positions get
/// one extra) -- the SAME rule `fuse::native::rank_slice`
/// (src/backend/native_generators.cpp, include/fuse/backend/
/// native_backend.hpp) applies to a real multi-rank mesh decomposition,
/// reproduced here rather than called directly.
///
/// WHY DUPLICATED, NOT REUSED. `native_backend.hpp` pulls in Kokkos
/// (`native_mesh.hpp`'s own `MeshArrays<Kokkos::HostSpace>`) -- a
/// dependency `src/abi/provider_adapter.cpp` has never needed, and
/// a shared, Kokkos-free primitive factored out of `rank_slice` is not in
/// place. The
/// formula is reproduced EXACTLY (`rank_slice`'s own two branches,
/// `position < extra` taking one extra element), not approximated; `0`
/// and `1` `group_size` both degenerate to the whole range, matching
/// `rank_slice`'s own `num_ranks <= 1` short-circuit. This could be
/// factored into a shared, dependency-free location so the two formulas
/// can never drift apart; it is not done here.
///
/// `position` and `group_size` are caller-validated (0 <= position <
/// group_size is assumed, not checked): this function has no failure
/// mode of its own and reports nothing -- a caller handing it an
/// out-of-range position has already made an error this function cannot
/// detect from the two integers alone.
[[nodiscard]] inline std::pair<std::size_t, std::size_t> even_rank_share(
    std::size_t total, int position, int group_size) noexcept {
  if (group_size <= 1) return {0, total};
  const std::size_t base = total / static_cast<std::size_t>(group_size);
  const std::size_t extra = total % static_cast<std::size_t>(group_size);
  const std::size_t p = static_cast<std::size_t>(position);
  const std::size_t lo = (p < extra) ? p * (base + 1) : extra * (base + 1) + (p - extra) * base;
  const std::size_t size = base + ((p < extra) ? 1 : 0);
  return {lo, lo + size};
}

}  // namespace fuse::abi_detail
