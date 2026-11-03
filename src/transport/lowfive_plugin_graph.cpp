// The graph/bindings/
// rendezvous half of the `fuse-lowfive-vol` plugin -- see lowfive_plugin_
// internal.hpp's own header comment for why this is a SEPARATE
// translation unit from lowfive_plugin_vol.cpp rather than one file: this
// one needs `fuse/model/yaml_io.hpp` (transitively, `fuse/base/
// result.hpp`'s `fmt` 12) and must NEVER see a `<lowfive/...>` header
// (whose own include directory carries a bundled, incompatible `fmt`
// 8.1.1) -- enforced by src/CMakeLists.txt linking this file's own
// object library against `fuse::model`/`fuse::base` only, never `fuse::
// lowfive`.
#include "fuse/base/config.hpp"

#if FUSE_HAVE_LOWFIVE

#include "lowfive_plugin_internal.hpp"

#include "fuse/model/yaml_io.hpp"
#include "fuse/transport/lowfive.hpp"
#include "fuse/transport/lowfive_graph.hpp"

#if FUSE_HAVE_MPI
#include <mpi.h>
#endif

#include <fmt/format.h>

#include <cstdint>
#include <cstdlib>
#include <string>

namespace fuse::lowfive_plugin_internal {

namespace {

#if FUSE_HAVE_MPI
// The distributed half: find the ONE LowFive-bound channel's
// writer/reader RankSets this process is a member of, via
// fuse::plan_lowfive_rendezvous() (include/fuse/transport/lowfive.hpp,
// its tier-0-tested rank-set arithmetic). Forming the
// actual intercommunicator (MPI_Comm_split/MPI_Intercomm_create) is
// deliberately NOT this function's job -- lowfive_plugin_vol.cpp does
// that, from the plain RendezvousPlan this returns, the same split this
// header's own comment explains.
//
// SCOPED TO MPMD LAUNCH ONLY: MPI_COMM_WORLD's own rank IS the graph's
// base-communicator rank in that mode. In ACCEPT mode the base
// communicator connect() builds by merging two intercommunicators is not
// visible to this path.
//
// `fuse_coupler_abi()`'s `base_communicator()` entry
// (`fuse_base_communicator_info`, `include/fuse/abi/entries.def`) does
// return the merged communicator setup() formed, for every launch mode,
// not MPMD alone. This function does not use it: it reads
// `MPI_COMM_WORLD` directly because it runs inside `fuse-lowfive-vol`'s
// OWN plugin-side env-route path (the `HDF5_VOL_CONNECTOR` route), a
// different consumer from the Brusselator example's programmatic attach
// path (`examples/brusselator/participants/lowfive_attach.cpp`, which
// DOES use the accessor). Extending the env-route path itself to accept
// mode -- reading `base_communicator()` here instead of assuming
// `MPI_COMM_WORLD` -- is not done here.
bool ConfigureDistributed(const fuse::CouplingGraph& g, Config* out) {
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (!mpi_initialized) return false;

  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // The first LowFive-bound channel found decides this process's own
  // writer/reader role -- the reference graph has exactly one
  // channel per direction, each a separate field, so "first found"
  // never has to choose between two disagreeing channels in practice;
  // a graph with more than one would need this loop to confirm every
  // LowFive-bound channel agrees on the SAME writer/reader pair this
  // process belongs to, which is not yet checked here.
  for (std::uint32_t i = 0; i < g.channel_count(); ++i) {
    const fuse::Channel& ch = g.channel_at(fuse::ChannelId{i});
    const fuse::FieldSpec& src = g.field(ch.source);
    if (!src.hdf5_path.has_value()) continue;
    const fuse::ComponentId writer_id = g.owner_of(ch.source);
    const fuse::ComponentId reader_id = g.owner_of(ch.target);
    if (!writer_id.valid() || !reader_id.valid()) continue;
    const fuse::RankSet writer_ranks = g.component(writer_id).ranks;
    const fuse::RankSet reader_ranks = g.component(reader_id).ranks;
    if (!writer_ranks.contains(world_rank) && !reader_ranks.contains(world_rank)) {
      continue;  // this channel's two sides do not include this rank
    }
    auto plan = fuse::plan_lowfive_rendezvous(writer_ranks, reader_ranks, world_rank);
    if (!plan.ok()) {
      fmt::print(stderr, "fuse-lowfive-vol: plan_lowfive_rendezvous failed: [{}] {}\n",
                plan.error().code, plan.error().message);
      return false;
    }
    out->distributed = true;
    out->rendezvous.is_writer = plan.value().is_writer;
    out->rendezvous.split_color = plan.value().split_color;
    out->rendezvous.split_key = plan.value().split_key;
    out->rendezvous.local_leader = plan.value().local_leader;
    out->rendezvous.remote_leader = plan.value().remote_leader;
    out->rendezvous.tag = plan.value().tag;
    return true;
  }
  return false;  // no LowFive-bound channel involves this rank
}
#endif  // FUSE_HAVE_MPI

}  // namespace

bool LoadConfig(Config* out) {
  *out = Config{};
  const char* bindings_path = std::getenv("FUSE_LOWFIVE_BINDINGS");
  if (!bindings_path) {
    fmt::print(stderr,
              "fuse-lowfive-vol: FUSE_LOWFIVE_BINDINGS not set -- no field "
              "bindings applied, every HDF5 call will reach a real file\n");
    return false;
  }
  auto g = fuse::load_yaml_file(bindings_path);
  if (!g.ok()) {
    fmt::print(stderr, "fuse-lowfive-vol: failed to load graph '{}': [{}] {}\n",
              bindings_path, g.error().code, g.error().message);
    return false;
  }
  auto table = fuse::build_lowfive_bindings(g.value());
  if (!table.ok()) {
    fmt::print(stderr, "fuse-lowfive-vol: build_lowfive_bindings('{}') failed: [{}] {}\n",
              bindings_path, table.error().code, table.error().message);
    return false;
  }
  for (const fuse::LowFiveFieldBinding& b : table.value().bindings()) {
    // LowFiveFieldBindingTable's own port syntax ("<file>.h5/<dataset>",
    // fuse/transport/lowfive.hpp) is already "<filename pattern>/<HDF5
    // path pattern>" -- exactly the two arguments set_memory() itself
    // takes, so this splits on the first '/' rather than re-deriving
    // the pattern.
    const std::string& port = b.hdf5_path;
    const std::size_t slash = port.find('/');
    if (slash == std::string::npos) continue;
    out->bindings.push_back({port.substr(0, slash), port.substr(slash)});
  }

#if FUSE_HAVE_MPI
  ConfigureDistributed(g.value(), out);
#endif
  return true;
}

}  // namespace fuse::lowfive_plugin_internal

#endif  // FUSE_HAVE_LOWFIVE
