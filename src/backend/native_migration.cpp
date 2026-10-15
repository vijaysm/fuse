// The only translation unit in fuse::native that names an MPI type: every
// function declared in native_migration.hpp without one (docs/dev/layering.md).

#include "fuse/backend/native_migration.hpp"

#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <unordered_map>

#include "fuse/base/config.hpp"
#include "fuse/base/error.hpp"
#include "fuse/backend/native/mesh_pack_device.hpp"
#include "fuse/linalg/comm.hpp"  // fuse::linalg::mpi_is_gpu_aware, the shared probe

#if FUSE_HAVE_MPI
#include <mpi.h>
#endif

namespace fuse::native {

namespace {

#if FUSE_HAVE_MPI
MPI_Comm resolve(CommHandle comm) {
  FUSE_ASSERT(comm != kNullComm,
             "native detail collective called with kNullComm; every caller must "
             "short-circuit locally before reaching here");
  return MPI_Comm_f2c(comm);
}
#else
void refuse_no_mpi() {
  FUSE_ASSERT(false,
             "fuse::native migration collective called with a non-null CommHandle in a "
             "build with FUSE_HAVE_MPI=0; migrate() should have refused before reaching "
             "here");
}
#endif

int comm_size(CommHandle comm) {
  if (comm == kNullComm) return 1;
#if FUSE_HAVE_MPI
  int size = 1;
  MPI_Comm_size(resolve(comm), &size);
  return size;
#else
  refuse_no_mpi();
  return 1;
#endif
}

int comm_rank(CommHandle comm) {
  if (comm == kNullComm) return 0;
#if FUSE_HAVE_MPI
  int rank = 0;
  MPI_Comm_rank(resolve(comm), &rank);
  return rank;
#else
  refuse_no_mpi();
  return 0;
#endif
}

int allreduce_min_int(int local, CommHandle comm) {
  if (comm == kNullComm) return local;
#if FUSE_HAVE_MPI
  int global = local;
  MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, resolve(comm));
  return global;
#else
  refuse_no_mpi();
  return local;
#endif
}

// ---- byte-buffer read helper (unpack only; pack now goes through
// `fuse::native::device::pack_to`, mesh_pack_device.hpp) -------------------

template <class T>
T read_pod(const std::vector<std::byte>& buf, std::size_t& pos) {
  static_assert(std::is_trivially_copyable_v<T>);
  T v{};
  std::memcpy(&v, buf.data() + pos, sizeof(T));
  pos += sizeof(T);
  return v;
}

/// Copies a device-resident packed buffer to a host `std::vector<std::byte>`
/// -- the ONE host round trip `pack_for_migration` still pays, for the wire
/// bytes themselves (never the mesh's own geometry), until
/// `exchange_migration_buffers` below becomes GPU-aware (this track's own
/// follow-up item).
std::vector<std::byte> to_host_bytes(const Kokkos::View<std::byte*, NativeMeshView::DeviceMemSpace>& v) {
  auto mirror = Kokkos::create_mirror_view(v);
  Kokkos::deep_copy(mirror, v);
  std::vector<std::byte> out(mirror.extent(0));
  if (!out.empty()) std::memcpy(out.data(), mirror.data(), mirror.extent(0));
  return out;
}

}  // namespace

// Migration track, 2026-10-15: a thin wrapper over the shared probe
// (`fuse::linalg::mpi_is_gpu_aware()`, src/linalg/mpi_gpu_aware.cpp) --
// kept as its own function, rather than deleted in favour of every call
// site naming `fuse::linalg::mpi_is_gpu_aware()` directly, for source
// compatibility with this header's existing callers
// (`exchange_migration_buffers` below, and every caller outside this
// file that already spells it `fuse::native::mpi_is_gpu_aware()`). See
// comm.hpp's own declaration for the probe's full behaviour (the
// `MPIX_Query_cuda_support()` run-time check, the `FUSE_MPI_GPU_AWARE`
// configure-time fallback, and the `FUSE_TRANSPORT_HOST_STAGING=1`
// override) -- this file used to duplicate that logic; see
// src/linalg/mpi_gpu_aware.cpp's header comment for why it no longer does.
bool mpi_is_gpu_aware() { return fuse::linalg::mpi_is_gpu_aware(); }

Result<std::vector<std::vector<std::byte>>> pack_for_migration(const NativeMeshView& mesh,
                                                                const std::vector<int>& dest,
                                                                int num_ranks) {
  const std::size_t nowned = mesh.owned_count(mesh.dimension());
  if (dest.size() != nowned) {
    return Failure{"E-MESH-MIGRATE-INCOMPLETE",
                   fmt::format("migrate needs exactly one destination per owned cell: this "
                              "rank owns {} cells but was given {} destinations", nowned,
                              dest.size())};
  }
  for (int d : dest) {
    if (d < 0 || d >= num_ranks) {
      return Failure{"E-MESH-MIGRATE-RANGE",
                     fmt::format("destination {} is outside the communicator [0, {})", d,
                                 num_ranks)};
    }
  }

  // Device-resident pack (Migration track, 2026-10-15): `dest`, one entry
  // per OWNED cell, is this function's only host-side input besides the
  // view itself -- copied to a small device array below (bookkeeping, not
  // mesh geometry) so `device::pack_to` can read `mesh.device_arrays()`
  // directly. No `NativeMeshView` host-serving accessor (`cell_vertices`,
  // `vertex_x`, `cell_faces`) is called anywhere in this function, so no
  // host mirror is materialized by a migration pack.
  using Space = NativeMeshView::DeviceMemSpace;
  Kokkos::View<int*, Kokkos::HostSpace> dest_h(
      Kokkos::view_alloc("pack_for_migration::dest_h", Kokkos::WithoutInitializing), nowned);
  for (std::size_t i = 0; i < nowned; ++i) dest_h(i) = dest[i];
  Kokkos::View<int*, Space> dest_d(
      Kokkos::view_alloc("pack_for_migration::dest_d", Kokkos::WithoutInitializing), nowned);
  Kokkos::deep_copy(dest_d, dest_h);

  auto packed = device::pack_to<false, Space>(mesh.device_arrays(), dest_d, num_ranks);

  std::vector<std::vector<std::byte>> result(packed.size());
  for (std::size_t r = 0; r < packed.size(); ++r) result[r] = to_host_bytes(packed[r]);
  return result;
}

std::vector<std::vector<std::byte>> exchange_migration_buffers(
    const std::vector<std::vector<std::byte>>& send, CommHandle comm) {
  if (comm == kNullComm) {
    FUSE_ASSERT(send.size() == 1,
               "exchange_migration_buffers: kNullComm means exactly one rank, so `send` "
               "must have exactly one entry (got {})", send.size());
    return {send[0]};
  }
#if FUSE_HAVE_MPI
  MPI_Comm c = resolve(comm);
  const int num_ranks = comm_size(comm);
  FUSE_ASSERT(static_cast<int>(send.size()) == num_ranks,
             "exchange_migration_buffers: send.size() ({}) must equal the communicator size "
             "({})", send.size(), num_ranks);

  std::vector<int> send_sizes(static_cast<std::size_t>(num_ranks));
  for (int r = 0; r < num_ranks; ++r) send_sizes[static_cast<std::size_t>(r)] =
                                          static_cast<int>(send[static_cast<std::size_t>(r)].size());
  std::vector<int> recv_sizes(static_cast<std::size_t>(num_ranks));
  MPI_Alltoall(send_sizes.data(), 1, MPI_INT, recv_sizes.data(), 1, MPI_INT, c);

  std::vector<int> send_displs(static_cast<std::size_t>(num_ranks));
  std::vector<int> recv_displs(static_cast<std::size_t>(num_ranks));
  int send_total = 0, recv_total = 0;
  for (int r = 0; r < num_ranks; ++r) {
    send_displs[static_cast<std::size_t>(r)] = send_total;
    recv_displs[static_cast<std::size_t>(r)] = recv_total;
    send_total += send_sizes[static_cast<std::size_t>(r)];
    recv_total += recv_sizes[static_cast<std::size_t>(r)];
  }

  std::vector<std::byte> flat_send(static_cast<std::size_t>(send_total));
  for (int r = 0; r < num_ranks; ++r) {
    const auto& s = send[static_cast<std::size_t>(r)];
    if (!s.empty()) {
      std::memcpy(flat_send.data() + send_displs[static_cast<std::size_t>(r)], s.data(),
                 s.size());
    }
  }
  std::vector<std::byte> flat_recv(static_cast<std::size_t>(recv_total));
  MPI_Alltoallv(flat_send.data(), send_sizes.data(), send_displs.data(), MPI_BYTE,
               flat_recv.data(), recv_sizes.data(), recv_displs.data(), MPI_BYTE, c);

  std::vector<std::vector<std::byte>> result(static_cast<std::size_t>(num_ranks));
  for (int r = 0; r < num_ranks; ++r) {
    const auto n = static_cast<std::size_t>(recv_sizes[static_cast<std::size_t>(r)]);
    result[static_cast<std::size_t>(r)].assign(
        flat_recv.begin() + recv_displs[static_cast<std::size_t>(r)],
        flat_recv.begin() + recv_displs[static_cast<std::size_t>(r)] + static_cast<long>(n));
  }
  return result;
#else
  refuse_no_mpi();
  return {};
#endif
}

// Migration track, 2026-10-15: the GPU-aware replacement for the
// `std::vector` overload above (native_migration.hpp's own "GPU-AWARE
// MPI" paragraph has this primitive's status: proven standalone here,
// not yet called by `migrate`/`build_native_remap_halo`). `Space` is a
// device memory space with `fuse::linalg::mpi_is_gpu_aware()` true (and
// `FUSE_TRANSPORT_HOST_STAGING` not forcing the fallback) hands
// `MPI_Alltoallv` the device buffers' own raw pointers via one flat
// device-resident send/receive buffer (Alltoallv needs ONE base pointer
// plus displacements, while `send[r]` are independently allocated
// Views); every other case stages through host `std::vector<std::byte>`
// buffers, the identical two-phase shape the `std::vector` overload
// above already uses, then copies the result back to `Space`.
template <class Space>
std::vector<Kokkos::View<std::byte*, Space>> exchange_migration_buffers(
    const std::vector<Kokkos::View<std::byte*, Space>>& send, CommHandle comm) {
  if (comm == kNullComm) {
    FUSE_ASSERT(send.size() == 1,
               "exchange_migration_buffers<Space>: kNullComm means exactly one rank, so "
               "`send` must have exactly one entry (got {})", send.size());
    return {send[0]};
  }
#if FUSE_HAVE_MPI
  MPI_Comm c = resolve(comm);
  const int num_ranks = comm_size(comm);
  FUSE_ASSERT(static_cast<int>(send.size()) == num_ranks,
             "exchange_migration_buffers<Space>: send.size() ({}) must equal the "
             "communicator size ({})", send.size(), num_ranks);

  std::vector<int> send_sizes(static_cast<std::size_t>(num_ranks));
  for (int r = 0; r < num_ranks; ++r) {
    send_sizes[static_cast<std::size_t>(r)] = static_cast<int>(send[static_cast<std::size_t>(r)].extent(0));
  }
  std::vector<int> recv_sizes(static_cast<std::size_t>(num_ranks));
  MPI_Alltoall(send_sizes.data(), 1, MPI_INT, recv_sizes.data(), 1, MPI_INT, c);

  std::vector<int> send_displs(static_cast<std::size_t>(num_ranks));
  std::vector<int> recv_displs(static_cast<std::size_t>(num_ranks));
  int send_total = 0, recv_total = 0;
  for (int r = 0; r < num_ranks; ++r) {
    send_displs[static_cast<std::size_t>(r)] = send_total;
    recv_displs[static_cast<std::size_t>(r)] = recv_total;
    send_total += send_sizes[static_cast<std::size_t>(r)];
    recv_total += recv_sizes[static_cast<std::size_t>(r)];
  }

  // Device-resident path only when `Space` genuinely is a device memory
  // space AND the run-time probe (not merely "this build links CUDA-aware
  // MPI") says so -- the identical condition `FUSE_TRANSPORT_HOST_STAGING`
  // overrides inside the probe itself (comm.hpp).
  constexpr bool kSpaceIsHost = std::is_same_v<Space, Kokkos::HostSpace>;
  const bool use_device_ptrs = !kSpaceIsHost && fuse::linalg::mpi_is_gpu_aware();

  std::vector<Kokkos::View<std::byte*, Space>> result(static_cast<std::size_t>(num_ranks));
  if (use_device_ptrs) {
    Kokkos::View<std::byte*, Space> flat_send(
        Kokkos::view_alloc("exchange_migration_buffers::flat_send_d", Kokkos::WithoutInitializing),
        send_total);
    for (int r = 0; r < num_ranks; ++r) {
      const auto& s = send[static_cast<std::size_t>(r)];
      if (s.extent(0) == 0) continue;
      const auto dst = Kokkos::subview(
          flat_send, Kokkos::make_pair(send_displs[static_cast<std::size_t>(r)],
                                       send_displs[static_cast<std::size_t>(r)] +
                                           send_sizes[static_cast<std::size_t>(r)]));
      Kokkos::deep_copy(dst, s);
    }
    Kokkos::View<std::byte*, Space> flat_recv(
        Kokkos::view_alloc("exchange_migration_buffers::flat_recv_d", Kokkos::WithoutInitializing),
        recv_total);
    MPI_Alltoallv(flat_send.data(), send_sizes.data(), send_displs.data(), MPI_BYTE,
                 flat_recv.data(), recv_sizes.data(), recv_displs.data(), MPI_BYTE, c);
    for (int r = 0; r < num_ranks; ++r) {
      const auto n = recv_sizes[static_cast<std::size_t>(r)];
      Kokkos::View<std::byte*, Space> buf(
          Kokkos::view_alloc("exchange_migration_buffers::recv_buf_d", Kokkos::WithoutInitializing),
          n);
      if (n > 0) {
        const auto src = Kokkos::subview(
            flat_recv, Kokkos::make_pair(recv_displs[static_cast<std::size_t>(r)],
                                         recv_displs[static_cast<std::size_t>(r)] + n));
        Kokkos::deep_copy(buf, src);
      }
      result[static_cast<std::size_t>(r)] = buf;
    }
  } else {
    std::vector<std::byte> flat_send_h(static_cast<std::size_t>(send_total));
    for (int r = 0; r < num_ranks; ++r) {
      const auto& s = send[static_cast<std::size_t>(r)];
      if (s.extent(0) == 0) continue;
      auto mirror = Kokkos::create_mirror_view(s);
      Kokkos::deep_copy(mirror, s);
      std::memcpy(flat_send_h.data() + send_displs[static_cast<std::size_t>(r)], mirror.data(),
                 mirror.extent(0));
    }
    std::vector<std::byte> flat_recv_h(static_cast<std::size_t>(recv_total));
    MPI_Alltoallv(flat_send_h.data(), send_sizes.data(), send_displs.data(), MPI_BYTE,
                 flat_recv_h.data(), recv_sizes.data(), recv_displs.data(), MPI_BYTE, c);
    for (int r = 0; r < num_ranks; ++r) {
      const auto n = static_cast<std::size_t>(recv_sizes[static_cast<std::size_t>(r)]);
      Kokkos::View<std::byte*, Space> buf(
          Kokkos::view_alloc("exchange_migration_buffers::recv_buf_d2", Kokkos::WithoutInitializing),
          n);
      if (n > 0) {
        auto mirror = Kokkos::create_mirror_view(buf);
        std::memcpy(mirror.data(), flat_recv_h.data() + recv_displs[static_cast<std::size_t>(r)], n);
        Kokkos::deep_copy(buf, mirror);
      }
      result[static_cast<std::size_t>(r)] = buf;
    }
  }
  return result;
#else
  refuse_no_mpi();
  return {};
#endif
}

template std::vector<Kokkos::View<std::byte*, Kokkos::HostSpace>>
exchange_migration_buffers<Kokkos::HostSpace>(
    const std::vector<Kokkos::View<std::byte*, Kokkos::HostSpace>>&, CommHandle);
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
template std::vector<Kokkos::View<std::byte*, NativeMeshView::DeviceMemSpace>>
exchange_migration_buffers<NativeMeshView::DeviceMemSpace>(
    const std::vector<Kokkos::View<std::byte*, NativeMeshView::DeviceMemSpace>>&, CommHandle);
#endif

Result<MeshArrays<Kokkos::HostSpace>> unpack_migrated_mesh(
    const std::vector<std::vector<std::byte>>& received, int dim, int my_rank) {
  NativeMeshBuilder builder(dim);
  std::unordered_map<std::int64_t, int> vertex_by_gid;  // gid -> local vertex index

  for (const auto& buf : received) {
    std::size_t pos = 0;
    while (pos < buf.size()) {
      const auto cell_gid = read_pod<std::int64_t>(buf, pos);
      const auto type = static_cast<CellType>(read_pod<std::uint8_t>(buf, pos));
      const auto n = static_cast<std::size_t>(read_pod<std::int32_t>(buf, pos));

      std::vector<int> closure(n);
      for (std::size_t k = 0; k < n; ++k) {
        const auto gid = read_pod<std::int64_t>(buf, pos);
        const double x = read_pod<double>(buf, pos);
        const double y = read_pod<double>(buf, pos);
        const double z = read_pod<double>(buf, pos);
        auto it = vertex_by_gid.find(gid);
        if (it == vertex_by_gid.end()) {
          const int lv = builder.add_vertex(gid, x, y, z, my_rank, true);
          it = vertex_by_gid.emplace(gid, lv).first;
        }
        closure[k] = it->second;
      }

      if (type == CellType::Polyhedron) {
        const auto nfaces = static_cast<std::size_t>(read_pod<std::int32_t>(buf, pos));
        std::vector<std::vector<int>> faces(nfaces);
        for (std::size_t f = 0; f < nfaces; ++f) {
          const auto nfv = static_cast<std::size_t>(read_pod<std::int32_t>(buf, pos));
          faces[f].resize(nfv);
          for (std::size_t k = 0; k < nfv; ++k) {
            const auto ci = static_cast<std::size_t>(read_pod<std::int32_t>(buf, pos));
            faces[f][k] = closure[ci];
          }
        }
        builder.add_polyhedron_cell(faces, cell_gid, my_rank, true);
      } else {
        builder.add_cell(type, closure, cell_gid, my_rank, true);
      }
    }
  }

  return builder.build();
}

void resolve_vertex_ownership(MeshArrays<Kokkos::HostSpace>& arrays, CommHandle comm) {
  const std::size_t nv = arrays.vertex_count();
  if (comm == kNullComm) {
    for (std::size_t v = 0; v < nv; ++v) {
      arrays.vertex_owner(v) = 0;
      arrays.vertex_is_owned(v) = true;
    }
    return;
  }
#if FUSE_HAVE_MPI
  MPI_Comm c = resolve(comm);
  const int num_ranks = comm_size(comm);
  const int my_rank = comm_rank(comm);

  std::vector<std::int64_t> local_gids(nv);
  for (std::size_t v = 0; v < nv; ++v) local_gids[v] = arrays.vertex_gid(v);

  const int local_count = static_cast<int>(nv);
  std::vector<int> counts(static_cast<std::size_t>(num_ranks));
  MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, c);
  std::vector<int> displs(static_cast<std::size_t>(num_ranks));
  int total = 0;
  for (int r = 0; r < num_ranks; ++r) {
    displs[static_cast<std::size_t>(r)] = total;
    total += counts[static_cast<std::size_t>(r)];
  }
  std::vector<std::int64_t> gathered(static_cast<std::size_t>(total));
  MPI_Allgatherv(local_gids.data(), local_count, MPI_INT64_T, gathered.data(), counts.data(),
                displs.data(), MPI_INT64_T, c);

  // Global minimum rank holding each gid, from every rank's contiguous
  // segment of `gathered` (`displs`/`counts` mark segment boundaries).
  std::unordered_map<std::int64_t, int> min_rank_of;
  min_rank_of.reserve(static_cast<std::size_t>(total));
  for (int r = 0; r < num_ranks; ++r) {
    const int start = displs[static_cast<std::size_t>(r)];
    const int count = counts[static_cast<std::size_t>(r)];
    for (int k = 0; k < count; ++k) {
      const std::int64_t gid = gathered[static_cast<std::size_t>(start + k)];
      auto it = min_rank_of.find(gid);
      if (it == min_rank_of.end() || r < it->second) min_rank_of[gid] = r;
    }
  }

  for (std::size_t v = 0; v < nv; ++v) {
    const int owner = min_rank_of.at(arrays.vertex_gid(v));
    arrays.vertex_owner(v) = owner;
    arrays.vertex_is_owned(v) = (owner == my_rank);
  }
#else
  refuse_no_mpi();
#endif
}

Result<std::unique_ptr<NativeMeshView>> migrate(const NativeMeshView& mesh,
                                                const std::vector<int>& dest, CommHandle comm) {
  const int num_ranks = comm_size(comm);
  const int my_rank = comm_rank(comm);

  // LOCAL validity, computed before any collective (this rank's own
  // Result), then reduced with every other rank -- see the header
  // comment: a rank must never return this rejection on its own and skip
  // the exchange while its neighbours proceed into it.
  auto local = pack_for_migration(mesh, dest, num_ranks);
  const int locally_ok = local.ok() ? 1 : 0;
  const int globally_ok = allreduce_min_int(locally_ok, comm);
  if (globally_ok == 0) {
    if (!local.ok()) return local.error();
    return Failure{"E-MESH-MIGRATE-PEER-REFUSED",
                   fmt::format("rank {} is proceeding, but at least one other rank in this "
                              "communicator refused its own destination assignment; every "
                              "rank refuses together so none hangs waiting on the exchange",
                              my_rank)};
  }

  auto received = exchange_migration_buffers(local.value(), comm);
  auto arrays_r = unpack_migrated_mesh(received, mesh.dimension(), my_rank);

  // NM-6: `unpack_migrated_mesh` rebuilds THIS rank's own mesh from
  // whatever it alone received, so it can fail on data no peer shares
  // (e.g. a malformed incoming record) -- reconciled here, BEFORE
  // `resolve_vertex_ownership`'s own Allgather/Allgatherv (comm.hpp's own
  // two-phase pattern), the same way the pre-exchange verdict above
  // already is.
  const int unpack_locally_ok = arrays_r.ok() ? 1 : 0;
  const int unpack_globally_ok = allreduce_min_int(unpack_locally_ok, comm);
  if (unpack_globally_ok == 0) {
    if (!arrays_r.ok()) return arrays_r.error();
    return Failure{"E-MESH-MIGRATE-PEER-REFUSED",
                   fmt::format("rank {} unpacked its migrated cells successfully, but at "
                              "least one other rank in this communicator failed rebuilding "
                              "its own mesh from what it received; every rank refuses "
                              "together so none hangs waiting on vertex-ownership resolution",
                              my_rank)};
  }

  auto arrays = arrays_r.value();
  resolve_vertex_ownership(arrays, comm);
  return NativeMeshView::create_from_host(arrays, my_rank);
}

}  // namespace fuse::native
