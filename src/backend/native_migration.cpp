// The only translation unit in fuse::native that names an MPI type: every
// function declared in native_migration.hpp without one (docs/dev/layering.md).

#include "fuse/backend/native_migration.hpp"

#include <cstring>
#include <type_traits>
#include <unordered_map>

#include "fuse/base/config.hpp"
#include "fuse/base/error.hpp"

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

// ---- byte-buffer append/read helpers -------------------------------------

template <class T>
void append_pod(std::vector<std::byte>& buf, const T& v) {
  static_assert(std::is_trivially_copyable_v<T>);
  const auto* p = reinterpret_cast<const std::byte*>(&v);
  buf.insert(buf.end(), p, p + sizeof(T));
}

template <class T>
T read_pod(const std::vector<std::byte>& buf, std::size_t& pos) {
  static_assert(std::is_trivially_copyable_v<T>);
  T v{};
  std::memcpy(&v, buf.data() + pos, sizeof(T));
  pos += sizeof(T);
  return v;
}

void pack_one_cell(const NativeMeshView& mesh, std::size_t c, std::vector<std::byte>& out) {
  const auto type = mesh.cell_type_of(c);
  const auto& verts = mesh.cell_vertices(c);  // mesh-local vertex indices
  append_pod<std::int64_t>(out, static_cast<std::int64_t>(mesh.global_id(mesh.dimension(), c)));
  append_pod<std::uint8_t>(out, static_cast<std::uint8_t>(type));
  append_pod<std::int32_t>(out, static_cast<std::int32_t>(verts.size()));
  // mesh-local vertex index -> position within THIS cell's closure (0..n-1),
  // needed below to translate a polyhedron's face lists.
  std::unordered_map<int, int> local_to_closure;
  local_to_closure.reserve(verts.size());
  for (std::size_t k = 0; k < verts.size(); ++k) {
    const int lv = verts[k];
    local_to_closure[lv] = static_cast<int>(k);
    append_pod<std::int64_t>(out, static_cast<std::int64_t>(mesh.vertex_gid_of(
                                      static_cast<std::size_t>(lv))));
    append_pod<double>(out, mesh.vertex_x(static_cast<std::size_t>(lv), 0));
    append_pod<double>(out, mesh.vertex_x(static_cast<std::size_t>(lv), 1));
    append_pod<double>(out, mesh.vertex_x(static_cast<std::size_t>(lv), 2));
  }
  if (type == CellType::Polyhedron) {
    const auto& faces = mesh.cell_faces(c);
    append_pod<std::int32_t>(out, static_cast<std::int32_t>(faces.size()));
    for (const auto& face : faces) {
      append_pod<std::int32_t>(out, static_cast<std::int32_t>(face.size()));
      for (int lv : face) {
        append_pod<std::int32_t>(out, static_cast<std::int32_t>(local_to_closure.at(lv)));
      }
    }
  }
}

}  // namespace

bool mpi_is_gpu_aware() { return FUSE_MPI_GPU_AWARE != 0; }

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

  std::vector<std::vector<std::byte>> result(static_cast<std::size_t>(num_ranks));
  std::size_t oi = 0;
  for (std::size_t c = 0; c < mesh.cell_count(); ++c) {
    if (!mesh.is_owned(mesh.dimension(), c)) continue;
    const int d = dest[oi++];
    pack_one_cell(mesh, c, result[static_cast<std::size_t>(d)]);
  }
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
  if (!arrays_r.ok()) return arrays_r.error();
  auto arrays = arrays_r.value();
  resolve_vertex_ownership(arrays, comm);
  return NativeMeshView::create_from_host(arrays, my_rank);
}

}  // namespace fuse::native
