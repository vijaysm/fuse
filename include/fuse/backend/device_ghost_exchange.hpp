#pragma once

/// runtime-exchange track (layering decision from the lead, recorded in
/// docs/dev/device-resident-arcs.md): a device-resident drive of
/// `fuse::linalg::GhostExchanger::update()`'s own exchange, for a caller
/// whose `owned`/`ghosts` arrays live in a Kokkos execution space's
/// memory rather than always on the host. `fuse::linalg` itself stays
/// Kokkos-free (comm.hpp's own header comment); this class is the lowest
/// layer that is not (tools/check-layering.py: `backend` sits above
/// `linalg`), reached through `fuse::linalg::PackedExchanger`
/// (comm.hpp) -- the pack/unpack maps and raw `MPI_Alltoallv`
/// `GhostExchanger::update()` already derives once and reuses, exposed
/// so a caller that can pack and unpack on ITS OWN memory space can
/// drive the identical exchange without `update()`'s own host-side
/// pack/unpack loop.
///
/// NOT WIRED INTO AN EXISTING PRODUCTION CALLER (stated here, and in
/// report-runtime-exchange.md, as a deviation): `fuse::linalg::DistVector`
/// is the one place `GhostExchanger::update()` is called today
/// (src/linalg/dist_vector.cpp), and it is plain-host `std::vector`
/// storage by design (component-internal Newton-Krylov linear algebra,
/// which docs/dev/device-resident-arcs.md already lists as staying host);
/// `fuse::backend::NativeMeshBackend::remap_halo_pattern` builds a
/// `GhostPattern` but never itself calls `update()` on it (that is
/// `examples/brusselator/species_core.hpp`'s own `GhostExchangerFactory`,
/// example code outside this library). This class is new infrastructure
/// for the next device-resident consumer, built and tested on its own
/// terms (test/parallel/device_ghost_exchange_test.cpp) rather than
/// retrofitted onto a caller that does not yet need it.

#include "fuse/base/config.hpp"
#include "fuse/base/error.hpp"
#include "fuse/base/result.hpp"
#include "fuse/linalg/comm.hpp"

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace fuse::backend {

/// `update<Space>(owned, owned_n, ghosts, ghost_n)` takes the SAME
/// contract `fuse::linalg::GhostExchanger::update()` does, except
/// `owned`/`ghosts` are `Space`-resident pointers (host or device)
/// rather than always host. The path taken, decided fresh each call:
///
///   1. `exchanger->ghost_size() == 0` (one rank, or a pattern with no
///      ghosts at all): nothing to move, in any space.
///   2. The base exchanger's `PackedExchanger` facet is reachable
///      (an `MpiExchanger`, never the serial exchanger -- see 1 above,
///      which already covers the only case a serial exchanger reaches)
///      AND `Space` is a device space AND
///      `fuse::linalg::mpi_is_gpu_aware()`: pack and unpack with Kokkos
///      kernels ON `Space`, then `exchange_packed()` with DEVICE
///      pointers directly -- no host round trip of the ghost values.
///   3. Otherwise (host space, or not GPU-aware, or
///      `FUSE_TRANSPORT_HOST_STAGING=1` forcing the fallback): pack and
///      unpack with Kokkos kernels on `Space`, mirroring the packed
///      buffers to and from the host around the one `exchange_packed()`
///      (or, lacking a `PackedExchanger` facet at all, `update()` itself)
///      call that actually touches MPI.
///
/// `last_update_took_device_direct_path()` reports which of paths 2/3
/// the MOST RECENT `update()` call took -- the same role
/// `RunStats::device_transport_moves` plays for
/// `fuse::transport::MpiTransport` (docs/dev/device-resident-arcs.md
/// Phase 3), for a test to assert against.
class DeviceGhostExchanger {
 public:
  explicit DeviceGhostExchanger(std::shared_ptr<const linalg::GhostExchanger> exchanger)
      : exchanger_(std::move(exchanger)) {
#if FUSE_HAVE_MPI
    packed_ = std::dynamic_pointer_cast<const linalg::PackedExchanger>(exchanger_);
#endif
  }

  template <class Space>
  [[nodiscard]] Result<void> update(const double* owned, std::size_t owned_n, double* ghosts,
                                    std::size_t ghost_n) const {
    // DELIBERATELY NO "ghost_n == 0, return early" check here: ghost_n is
    // a PER-RANK count (this rank's own local pattern size), and
    // `exchange_packed()` underneath is a COLLECTIVE `MPI_Alltoallv` --
    // one rank quietly skipping it because ITS OWN ghost_n happens to be
    // 0 while a PEER's is not deadlocks that peer forever (measured:
    // `parallel.nodal_remap_runner` at 2+ ranks hung exactly this way
    // before this fix). A zero-count Alltoallv entry is a well-defined,
    // cheap no-op MPI already handles correctly, the same reason
    // `MpiExchanger::update()` itself (comm_mpi.cpp) never special-cases
    // ghost_n == 0 either -- every rank of the communicator always
    // enters the SAME collective, whatever its own local counts are.
    last_device_direct_ = false;

#if FUSE_HAVE_MPI
    if (packed_) {
      const auto& pack_idx = packed_->pack_index();
      const auto& unpack_idx = packed_->unpack_index();
      const std::size_t send_total = pack_idx.size();
      const std::size_t recv_total = unpack_idx.size();

      Kokkos::View<const double*, Space, Kokkos::MemoryTraits<Kokkos::Unmanaged>> owned_view(
          owned, owned_n);
      Kokkos::View<double*, Space, Kokkos::MemoryTraits<Kokkos::Unmanaged>> ghosts_view(ghosts,
                                                                                        ghost_n);
      Kokkos::View<double*, Space> send_buf(
          Kokkos::view_alloc(Kokkos::WithoutInitializing, "ghost_exchange_send"), send_total);
      Kokkos::View<double*, Space> recv_buf(
          Kokkos::view_alloc(Kokkos::WithoutInitializing, "ghost_exchange_recv"), recv_total);

      constexpr bool kSpaceIsHost =
          Kokkos::SpaceAccessibility<Space, Kokkos::HostSpace>::accessible;
      if constexpr (kSpaceIsHost) {
        // Host space: the index maps are already host `std::vector`s --
        // no device cache needed, a Host-space kernel reads them
        // directly (the same "no Kokkos View required for correctness
        // on the host leg" posture execute_on<Host>'s own branches take
        // elsewhere in this track).
        const std::int64_t* pack_p = pack_idx.data();
        const std::int64_t* unpack_p = unpack_idx.data();
        Kokkos::parallel_for(
            "ghost_exchange_pack_host", Kokkos::RangePolicy<Space>(0, send_total),
            KOKKOS_LAMBDA(std::size_t k) {
              send_buf(k) = owned_view(static_cast<std::size_t>(pack_p[k]));
            });
        Kokkos::fence("ghost_exchange_pack_host_fence");
      } else {
        ensure_device_maps(pack_idx, unpack_idx);
        auto d_pack = d_pack_index_;
        Kokkos::parallel_for(
            "ghost_exchange_pack_device", Kokkos::RangePolicy<Space>(0, send_total),
            KOKKOS_LAMBDA(std::size_t k) {
              send_buf(k) = owned_view(static_cast<std::size_t>(d_pack(k)));
            });
        Kokkos::fence("ghost_exchange_pack_device_fence");
      }

      const bool gpu_aware = linalg::mpi_is_gpu_aware();
      if (!kSpaceIsHost && gpu_aware) {
        const auto ex = packed_->exchange_packed(send_buf.data(), recv_buf.data());
        if (!ex.ok()) return ex.error();
        last_device_direct_ = true;
      } else {
        auto send_host = Kokkos::create_mirror_view(send_buf);
        auto recv_host = Kokkos::create_mirror_view(recv_buf);
        Kokkos::deep_copy(send_host, send_buf);
        const auto ex = packed_->exchange_packed(send_host.data(), recv_host.data());
        if (!ex.ok()) return ex.error();
        Kokkos::deep_copy(recv_buf, recv_host);
      }

      if constexpr (kSpaceIsHost) {
        const std::int64_t* unpack_p = unpack_idx.data();
        Kokkos::parallel_for(
            "ghost_exchange_unpack_host", Kokkos::RangePolicy<Space>(0, recv_total),
            KOKKOS_LAMBDA(std::size_t k) {
              ghosts_view(static_cast<std::size_t>(unpack_p[k])) = recv_buf(k);
            });
        Kokkos::fence("ghost_exchange_unpack_host_fence");
      } else {
        auto d_unpack = d_unpack_index_;
        Kokkos::parallel_for(
            "ghost_exchange_unpack_device", Kokkos::RangePolicy<Space>(0, recv_total),
            KOKKOS_LAMBDA(std::size_t k) {
              ghosts_view(static_cast<std::size_t>(d_unpack(k))) = recv_buf(k);
            });
        Kokkos::fence("ghost_exchange_unpack_device_fence");
      }
      return Result<void>::success();
    }
#endif
    // No `PackedExchanger` facet: a serial, no-MPI exchanger, which
    // `ghost_size() == 0` above already catches unconditionally
    // (comm_serial.cpp's `SerialExchanger::update()` FUSE_ASSERTs
    // `ghost_n == 0`) -- this branch is therefore unreachable in
    // practice. Kept as a defensive host-staged fallback through the
    // generic `GhostExchanger` interface rather than an assert, so an
    // exchanger this class does not yet know about (one neither serial
    // nor an `MpiExchanger`) still gets a correct, if host-round-
    // tripping, answer instead of undefined behaviour.
    std::vector<double> owned_host(owned_n), ghost_host(ghost_n, 0.0);
    Kokkos::View<const double*, Space, Kokkos::MemoryTraits<Kokkos::Unmanaged>> owned_view(
        owned, owned_n);
    Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        owned_host_view(owned_host.data(), owned_n);
    Kokkos::deep_copy(owned_host_view, owned_view);
    const auto ex = exchanger_->update(owned_host.data(), owned_n, ghost_host.data(), ghost_n);
    if (!ex.ok()) return ex.error();
    Kokkos::View<double*, Space, Kokkos::MemoryTraits<Kokkos::Unmanaged>> ghosts_view(ghosts,
                                                                                      ghost_n);
    Kokkos::View<const double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        ghost_host_view(ghost_host.data(), ghost_n);
    Kokkos::deep_copy(ghosts_view, ghost_host_view);
    return Result<void>::success();
  }

  [[nodiscard]] bool last_update_took_device_direct_path() const noexcept {
    return last_device_direct_;
  }

  /// The ADJOINT of `update()`: scatters `contrib` (`Space`-resident,
  /// `ghost_size()` entries, in the SAME ghost order `update()`'s own
  /// `ghosts` argument uses) into `accum` (`Space`-resident, `accum_n`
  /// entries), ACCUMULATING -- `accum[k] += ...`, not `=` -- at
  /// whichever local index each contribution's owner resolved to.
  /// `accum` is NOT zeroed by this call; the caller zeroes it first
  /// (`nodal_remap_distributed.cpp`'s own H^T scatter needs this because
  /// several contributions, from different rows and possibly different
  /// ranks, legitimately land on the same destination entry -- the exact
  /// reduction `apply_distributed_HT`'s own `sum_by_gid` performs, done
  /// here with `Kokkos::atomic_add` instead of a pre-sorted local sum;
  /// see docs/dev/device-resident-arcs.md's own note on the
  /// determinism trade this makes against `apply_distributed_HT`'s
  /// bit-identical-across-rank-counts guarantee).
  ///
  /// Requires the `PackedExchanger` facet (an `MpiExchanger`); unlike
  /// `update()`, there is no generic fallback through the base
  /// `GhostExchanger` interface (it has no adjoint of its own), so this
  /// is called only when `exchanger->ghost_size() > 0`, which (comm_
  /// serial.cpp's own `SerialExchanger::update()` contract) can only be
  /// true for a real `MpiExchanger`.
  template <class Space>
  [[nodiscard]] Result<void> update_adjoint(const double* contrib, std::size_t contrib_n,
                                            double* accum, std::size_t accum_n) const {
    // Same reasoning as update()'s own comment just above: no per-rank
    // "ghost_n == 0" early return, since exchange_packed_adjoint() is
    // the identical COLLECTIVE this rank's peers may still need to
    // enter even when THIS rank's own counts are all zero.
    last_device_direct_ = false;
#if FUSE_HAVE_MPI
    if (packed_) {
      const auto& pack_idx = packed_->pack_index();      // ANSWER order, length send_total
      const auto& unpack_idx = packed_->unpack_index();  // REQUEST order, length recv_total == ghost_size()
      const std::size_t send_total = pack_idx.size();
      const std::size_t recv_total = unpack_idx.size();

      Kokkos::View<const double*, Space, Kokkos::MemoryTraits<Kokkos::Unmanaged>> contrib_view(
          contrib, contrib_n);
      Kokkos::View<double*, Space, Kokkos::MemoryTraits<Kokkos::Unmanaged>> accum_view(accum,
                                                                                       accum_n);
      Kokkos::View<double*, Space> req_buf(
          Kokkos::view_alloc(Kokkos::WithoutInitializing, "ghost_exchange_adjoint_req"),
          recv_total);
      Kokkos::View<double*, Space> ans_buf(
          Kokkos::view_alloc(Kokkos::WithoutInitializing, "ghost_exchange_adjoint_ans"),
          send_total);

      constexpr bool kSpaceIsHost =
          Kokkos::SpaceAccessibility<Space, Kokkos::HostSpace>::accessible;
      if constexpr (kSpaceIsHost) {
        const std::int64_t* unpack_p = unpack_idx.data();
        Kokkos::parallel_for(
            "ghost_exchange_adjoint_gather_host", Kokkos::RangePolicy<Space>(0, recv_total),
            KOKKOS_LAMBDA(std::size_t k) {
              req_buf(k) = contrib_view(static_cast<std::size_t>(unpack_p[k]));
            });
        Kokkos::fence("ghost_exchange_adjoint_gather_host_fence");
      } else {
        ensure_device_maps(pack_idx, unpack_idx);
        auto d_unpack = d_unpack_index_;
        Kokkos::parallel_for(
            "ghost_exchange_adjoint_gather_device", Kokkos::RangePolicy<Space>(0, recv_total),
            KOKKOS_LAMBDA(std::size_t k) {
              req_buf(k) = contrib_view(static_cast<std::size_t>(d_unpack(k)));
            });
        Kokkos::fence("ghost_exchange_adjoint_gather_device_fence");
      }

      const bool gpu_aware = linalg::mpi_is_gpu_aware();
      if (!kSpaceIsHost && gpu_aware) {
        const auto ex = packed_->exchange_packed_adjoint(req_buf.data(), ans_buf.data());
        if (!ex.ok()) return ex.error();
        last_device_direct_ = true;
      } else {
        auto req_host = Kokkos::create_mirror_view(req_buf);
        auto ans_host = Kokkos::create_mirror_view(ans_buf);
        Kokkos::deep_copy(req_host, req_buf);
        const auto ex = packed_->exchange_packed_adjoint(req_host.data(), ans_host.data());
        if (!ex.ok()) return ex.error();
        Kokkos::deep_copy(ans_buf, ans_host);
      }

      if constexpr (kSpaceIsHost) {
        const std::int64_t* pack_p = pack_idx.data();
        Kokkos::parallel_for(
            "ghost_exchange_adjoint_scatter_host", Kokkos::RangePolicy<Space>(0, send_total),
            KOKKOS_LAMBDA(std::size_t k) {
              Kokkos::atomic_add(&accum_view(static_cast<std::size_t>(pack_p[k])), ans_buf(k));
            });
        Kokkos::fence("ghost_exchange_adjoint_scatter_host_fence");
      } else {
        auto d_pack = d_pack_index_;
        Kokkos::parallel_for(
            "ghost_exchange_adjoint_scatter_device", Kokkos::RangePolicy<Space>(0, send_total),
            KOKKOS_LAMBDA(std::size_t k) {
              Kokkos::atomic_add(&accum_view(static_cast<std::size_t>(d_pack(k))), ans_buf(k));
            });
        Kokkos::fence("ghost_exchange_adjoint_scatter_device_fence");
      }
      return Result<void>::success();
    }
#endif
    (void)accum_n;
    return Failure{"E-RT-NODAL-DIST-NOMPI",
                   "DeviceGhostExchanger::update_adjoint: ghost_size() > 0 but the base "
                   "exchanger has no PackedExchanger facet -- unreachable for a real "
                   "MpiExchanger, so this indicates an exchanger type this class does not "
                   "yet support"};
  }

 private:
  std::shared_ptr<const linalg::GhostExchanger> exchanger_;
#if FUSE_HAVE_MPI
  std::shared_ptr<const linalg::PackedExchanger> packed_;
  // Device copies of pack_index()/unpack_index(), built lazily on first
  // use and cached for this object's lifetime: the pattern is fixed once
  // `fuse::linalg::make_mpi_exchanger()` builds it, so re-deriving these
  // every call would re-pay a cost setup already amortised once.
  mutable Kokkos::View<std::int64_t*, Kokkos::DefaultExecutionSpace::memory_space> d_pack_index_;
  mutable Kokkos::View<std::int64_t*, Kokkos::DefaultExecutionSpace::memory_space>
      d_unpack_index_;
  mutable bool maps_cached_ = false;

  void ensure_device_maps(const std::vector<std::int64_t>& pack_idx,
                          const std::vector<std::int64_t>& unpack_idx) const {
    if (maps_cached_) return;
    using DeviceMemSpace = Kokkos::DefaultExecutionSpace::memory_space;
    d_pack_index_ = Kokkos::View<std::int64_t*, DeviceMemSpace>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "ghost_exchange_pack_index"),
        pack_idx.size());
    d_unpack_index_ = Kokkos::View<std::int64_t*, DeviceMemSpace>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "ghost_exchange_unpack_index"),
        unpack_idx.size());
    auto h_pack = Kokkos::create_mirror_view(d_pack_index_);
    auto h_unpack = Kokkos::create_mirror_view(d_unpack_index_);
    for (std::size_t k = 0; k < pack_idx.size(); ++k) h_pack(k) = pack_idx[k];
    for (std::size_t k = 0; k < unpack_idx.size(); ++k) h_unpack(k) = unpack_idx[k];
    Kokkos::deep_copy(d_pack_index_, h_pack);
    Kokkos::deep_copy(d_unpack_index_, h_unpack);
    maps_cached_ = true;
  }
#endif
  mutable bool last_device_direct_ = false;
};

}  // namespace fuse::backend
