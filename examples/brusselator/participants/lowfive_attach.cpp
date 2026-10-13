// S6 phase 3 (Conformance, "LowFive" track), "LowFive attach" track: see
// lowfive_attach.h's own header comment for WHY this exists (the env-
// route null dereference measured on the host, KNOWN_ISSUES.md D38/D39)
// and lowfive_attach_decision.hpp's own header comment for the pure
// decision this file wraps with the real graph load, MPI rendezvous, and
// LowFive/HDF5 calls.
//
// ONE TRANSLATION UNIT, naming both a FUSE header that pulls in fmt 12
// (fuse/model/yaml_io.hpp, via fuse/base/result.hpp) AND LowFive's own
// headers (whose include directory historically carried a bundled,
// incompatible fmt 8.1.1) -- UNLIKE src/transport/lowfive_plugin_vol.cpp/
// lowfive_plugin_graph.cpp's own two-translation-unit split. That split
// was necessary when it was written (KNOWN_ISSUES.md D14's own Unit 3b
// account); it is not necessary now. The actual root cause was never "two
// fmt versions in one translation unit" -- it was `fuse_require(fmt ...)`
// having no minimum version, so with `/opt/lowfive` first on
// `CMAKE_PREFIX_PATH`, `find_package(fmt)` resolved LowFive's own bundled
// 8.1.1 project-wide. `fuse_require(fmt VERSION 9 ...)`
// (cmake/FuseDependencies.cmake, commit f59b446) fixed that at the
// source, measured end to end (KNOWN_ISSUES.md D14's "root cause" account,
// item 5: a fresh configure in EITHER CMAKE_PREFIX_PATH ordering builds
// every target clean, INCLUDING lowfive_transport.cpp itself, which also
// names both header families in one file and always has). This file
// relies on that same floor, not on a second instance of the two-TU
// split -- `cmake/FuseDependencies.cmake`'s own comment calls the split
// "redundant... correctness rests on the VERSION 9 floor alone," which is
// the lead's own ruling on the existing plugin, not a license to skip
// re-deriving it: this file DOES reuse the plugin's own graph-side
// building blocks (`fuse::build_lowfive_bindings()`, `fuse::
// plan_lowfive_rendezvous()`), just not lowfive_plugin_internal.hpp's own
// plain-data Config indirection, which existed ONLY to cross the
// (now-unnecessary) TU boundary and has nothing else to offer a caller
// that does not need that boundary.
//
// WHY ONE INTERCOMMUNICATOR SERVES A BIDIRECTIONAL EXCHANGE. The
// Brusselator's own two HDF5-bound channels (u_out: U writes, V reads;
// v_out: V writes, U reads) swap which RankSet is "writer" and which is
// "reader" between the two channels, but both channels partition
// MPI_COMM_WORLD into the SAME two groups, {U-ranks} and {V-ranks).
// `fuse::plan_lowfive_rendezvous()`'s own `split_color` (0 for the
// canonical channel's writer side, 1 for its reader side) is therefore a
// GROUP label, not a per-channel role: computed once, from whichever
// qualifying channel this file finds first (lowfive_attach_decision.
// hpp's own `decide_attach_mode()` additionally confirms every OTHER
// qualifying channel agrees on the same two groups, closing the gap
// src/transport/lowfive_plugin_graph.cpp's own ConfigureDistributed()
// comment names as unchecked there), and the resulting ONE
// intercommunicator is handed to LowFive::DistMetadataVOL's single-
// intercomm constructor exactly as the env-route plugin's own
// ConfigureOnce() does -- MPI intercommunicator point-to-point is
// bidirectional by construction, and DistMetadataVOL's own per-dataset
// routing (`set_memory()`, applied per binding below) is what tells it
// which claimed file a given H5Fcreate/H5Fopen belongs to, not which
// direction data happens to move in on a given call. The communicator
// `plan.split_color`/`split_key` describe is NOT formed by a fresh
// MPI_Comm_split() call in this file -- it is the caller's own, already
// -formed physics-only communicator (`physics_comm_f`, u_main.c's own
// u_comm / v_main.f90's own v_comm), verified against the plan's own
// expected size/position rather than re-split from MPI_COMM_WORLD. See
// bruss_lowfive_attach()'s own body below for why: that fresh split was
// this file's own first version, found wrong while implementing (lowfive_
// attach.h's own header comment has the full account).
//
// DEADLOCK-AVOIDANCE DECISION (owner's own ruling, this track): the
// serve_after_write()/receive_before_read() hooks below are placed at
// the natural points -- serve right after THIS process's own H5Fclose,
// receive right before THIS process's own H5Fopen -- exactly mirroring
// Demo B's one-directional producer/consumer call order (examples/
// demo_b/producer_main.cpp, consumer_main.cpp). Whether LowFive's own
// `DistMetadataVOL::serve_all()`/`broadcast_files()` can mutually block
// if both ranks' own schedules reach "serve mine" before "receive
// peer's" at the same window is UNVERIFIED here: no MPI in this sandbox,
// and LowFive's own source is not available to read either. No
// artificial reordering is introduced to work around a risk that cannot
// be measured in this sandbox either way -- host_pending names the exact
// row and what a hang versus a clean pass/fail looks like.
#include "lowfive_attach.h"

#include "fuse/base/config.hpp"

#if FUSE_HAVE_LOWFIVE

#include "lowfive_attach_decision.hpp"

#include "fuse/model/yaml_io.hpp"
#include "fuse/transport/lowfive_graph.hpp"

#include <lowfive/H5VOLProperty.hpp>
#include <lowfive/vol-metadata.hpp>
#if defined(FUSE_LOWFIVE_DIST_LIB_FOUND) && FUSE_HAVE_MPI
#include <lowfive/vol-dist-metadata.hpp>
#endif

#include <hdf5.h>

#if FUSE_HAVE_MPI
#include <mpi.h>
#endif

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Local name, mirroring src/transport/lowfive_transport.cpp's own
// FUSE_LOWFIVE_HAVE_DIST macro exactly (same three preconditions), under
// a name scoped to this file so the two definitions can never collide
// across translation units.
#if FUSE_HAVE_LOWFIVE && FUSE_HAVE_MPI && defined(FUSE_LOWFIVE_DIST_LIB_FOUND)
#define FUSE_EXAMPLES_LOWFIVE_ATTACH_HAVE_DIST 1
#else
#define FUSE_EXAMPLES_LOWFIVE_ATTACH_HAVE_DIST 0
#endif

namespace {

void CopyErrorMessage(const std::string& msg, char* errbuf, long long errbuf_len) {
  if (!errbuf || errbuf_len <= 0) return;
  const auto cap = static_cast<std::size_t>(errbuf_len - 1);
  const std::size_t n = msg.size() < cap ? msg.size() : cap;
  std::memcpy(errbuf, msg.data(), n);
  errbuf[n] = '\0';
}

/// Process-wide attach state. `bruss_lowfive_attach()` populates this at
/// most once (a second call is refused, never re-registers); serve/
/// receive and the write/read shim functions all read it afterward.
struct AttachState {
  bool attached = false;
  hid_t fapl = H5I_INVALID_HID;
  bool distributed = false;
#if FUSE_EXAMPLES_LOWFIVE_ATTACH_HAVE_DIST
  LowFive::DistMetadataVOL* dist_vol = nullptr;  // never owned/deleted; see VolSingleton's own note
#endif
};

AttachState& State() {
  static AttachState s;  // leaked, deliberately -- same reasoning as VolSingleton's own `g`
  return s;
}

/// Builds the FAPL carrying `vol`, verifies it actually landed (the same
/// read-back-and-compare VolSingleton/DistVolSingleton's own construction
/// already does in src/transport/lowfive_transport.cpp), and leaks the
/// `H5VOLProperty` deliberately: its destructor calls
/// `H5VLunregister_connector()`, which must never run while this
/// process's VOL is still in use (lowfive/H5VOLProperty.hpp's own
/// destructor; VolSingleton's own header comment #2 has the full
/// account of the "whichever H5VOLProperty is destroyed first
/// unregisters it out from under the other" hazard this avoids).
template <typename Vol>
bool BuildVerifiedFapl(Vol& vol, hid_t* fapl_out, std::string* error_out) {
  auto* prop = new LowFive::H5VOLProperty(vol);  // intentionally leaked
  const hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
  if (fapl < 0) {
    *error_out = "H5Pcreate(H5P_FILE_ACCESS) failed";
    return false;
  }
  prop->apply(fapl);
  hid_t on_list = H5I_INVALID_HID;
  const bool got = H5Pget_vol_id(fapl, &on_list) >= 0;
  const bool matches = got && on_list == prop->vol_id;
  if (on_list >= 0) H5VLclose(on_list);
  if (!matches) {
    H5Pclose(fapl);
    *error_out = "H5Pset_vol did not put the attached LowFive connector on "
                "the file-access property list";
    return false;
  }
  *fapl_out = fapl;
  return true;
}

void ApplyBindings(LowFive::MetadataVOL& vol, const fuse::LowFiveFieldBindingTable& table,
                   bool* ok, std::string* error_out) {
  *ok = true;
  for (const fuse::LowFiveFieldBinding& b : table.bindings()) {
    auto split = fuse::split_hdf5_path(b.hdf5_path);
    if (!split.ok()) {
      // build_lowfive_bindings() already validated every binding's own
      // hdf5_path through split_hdf5_path() (LowFiveFieldBindingTable::
      // add()'s own contract); a failure here would mean that contract
      // broke, not a new input -- defensive, not a reachable case.
      *ok = false;
      *error_out = "split_hdf5_path('" + b.hdf5_path + "'): [" + split.error().code +
                  "] " + split.error().message;
      return;
    }
    vol.set_memory(split.value().first, split.value().second);
  }
}

}  // namespace

extern "C" {

int bruss_lowfive_attach(const char* graph_path, long long physics_comm_f,
                         long long* fapl_id_out,
                         char* errbuf, long long errbuf_len) {
  if (fapl_id_out) *fapl_id_out = -1;
  AttachState& s = State();
  if (s.attached) {
    CopyErrorMessage(
        "bruss_lowfive_attach() called a second time in this process; the "
        "VOL registration is process-wide and exactly once "
        "(src/transport/lowfive_transport.cpp's own VolSingleton has the "
        "full account of why re-registering is not safe)",
        errbuf, errbuf_len);
    return 1;
  }
  if (!graph_path || !*graph_path) {
    CopyErrorMessage("bruss_lowfive_attach(): graph_path is empty", errbuf, errbuf_len);
    return 1;
  }

#if !FUSE_HAVE_MPI
  (void)graph_path;
  CopyErrorMessage(
      "bruss_lowfive_attach(): this build has no MPI (-DFUSE_ENABLE_MPI=OFF); "
      "the Brusselator participants always need it",
      errbuf, errbuf_len);
  return 1;
#else
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (!mpi_initialized) {
    CopyErrorMessage("bruss_lowfive_attach(): MPI is not initialised; call MPI_Init() first",
                     errbuf, errbuf_len);
    return 1;
  }
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  auto g = fuse::load_yaml_file(graph_path);
  if (!g.ok()) {
    CopyErrorMessage("load_yaml_file('" + std::string(graph_path) + "'): [" + g.error().code +
                     "] " + g.error().message,
                     errbuf, errbuf_len);
    return 1;
  }
  const fuse::CouplingGraph& graph = g.value();

  auto table = fuse::build_lowfive_bindings(graph);
  if (!table.ok()) {
    CopyErrorMessage("build_lowfive_bindings: [" + table.error().code + "] " +
                     table.error().message,
                     errbuf, errbuf_len);
    return 1;
  }
  if (table.value().empty()) {
    CopyErrorMessage(
        "build_lowfive_bindings: the loaded graph has no field with "
        "hdf5_path set; --transport lowfive has nothing to attach to",
        errbuf, errbuf_len);
    return 1;
  }

  // Mirror src/transport/lowfive_plugin_graph.cpp's own
  // ConfigureDistributed() loop, generalized to collect EVERY qualifying
  // channel (not just the first) so decide_attach_mode() can check they
  // all agree -- see lowfive_attach_decision.hpp's own header comment.
  std::vector<fuse_examples::lowfive_attach::ChannelRoles> channels;
  for (std::uint32_t i = 0; i < graph.channel_count(); ++i) {
    const fuse::Channel& ch = graph.channel_at(fuse::ChannelId{i});
    const fuse::FieldSpec& src = graph.field(ch.source);
    if (!src.hdf5_path.has_value()) continue;
    const fuse::ComponentId writer_id = graph.owner_of(ch.source);
    const fuse::ComponentId reader_id = graph.owner_of(ch.target);
    if (!writer_id.valid() || !reader_id.valid()) continue;
    const fuse::ComponentDecl& writer_c = graph.component(writer_id);
    const fuse::ComponentDecl& reader_c = graph.component(reader_id);
    if (!writer_c.ranks.contains(world_rank) && !reader_c.ranks.contains(world_rank)) {
      continue;  // this channel's two sides do not include this rank
    }
    fuse_examples::lowfive_attach::ChannelRoles roles;
    roles.field_name = writer_c.name + "." + src.name;
    roles.writer_ranks = writer_c.ranks;
    roles.reader_ranks = reader_c.ranks;
    roles.writer_launch = writer_c.launch.mode;
    roles.reader_launch = reader_c.launch.mode;
    channels.push_back(std::move(roles));
  }

  auto decision = fuse_examples::lowfive_attach::decide_attach_mode(channels, world_rank);
  if (!decision.ok()) {
    CopyErrorMessage("[" + decision.error().code + "] " + decision.error().message, errbuf,
                     errbuf_len);
    return 1;
  }

  // ABORT AT EXIT, registered BEFORE any native-VOL HDF5 call this
  // process might make -- the SAME fix VolSingleton/DistVolSingleton's
  // own constructors register (src/transport/lowfive_transport.cpp) and
  // docs/user/lowfive.md's "The HDF5_VOL_CONNECTOR abort" section
  // records in full.
  std::atexit(+[] { H5close(); });

  hid_t fapl = H5I_INVALID_HID;
  bool distributed_now = false;
  std::string build_error;

  if (decision.value().mode == fuse_examples::lowfive_attach::AttachMode::kDistributed) {
#if FUSE_EXAMPLES_LOWFIVE_ATTACH_HAVE_DIST
    const fuse::LowFiveRendezvousPlan& plan = decision.value().rendezvous;
    // `local` is THIS process's own, ALREADY-FORMED physics-only
    // communicator (u_main.c's own u_comm / v_main.f90's own v_comm) --
    // NOT a fresh MPI_Comm_split(MPI_COMM_WORLD, ...) of this function's
    // own. See lowfive_attach.h's own header comment ("FOUND WHILE
    // IMPLEMENTING") for why: a split of MPI_COMM_WORLD here would be a
    // NEW collective the coupler process never joins (coupler_main.c
    // never calls this function), hanging U and V. MPI_Intercomm_create
    // needs no such split: it is collective only over the union of the
    // two SIDES' own groups, which u_comm/v_comm already are.
    if (physics_comm_f == 0) {
      CopyErrorMessage(
          "bruss_lowfive_attach(): physics_comm_f is 0 (MPI_COMM_NULL); "
          "pass this process's own already-formed physics-only "
          "communicator (u_comm/v_comm), not an unset handle",
          errbuf, errbuf_len);
      return 1;
    }
    const MPI_Comm local = MPI_Comm_f2c(static_cast<MPI_Fint>(physics_comm_f));
    if (local == MPI_COMM_NULL) {
      CopyErrorMessage(
          "bruss_lowfive_attach(): physics_comm_f converts to MPI_COMM_NULL",
          errbuf, errbuf_len);
      return 1;
    }
    // Verify `local` actually has the shape the rendezvous plan assumed
    // (this rank's own side of the channel, same size and same relative
    // position `plan_lowfive_rendezvous()` computed from the graph's own
    // RankSet) -- "rather than trust it", the same discipline `fuse::
    // LowFiveTransport::build_plan_from_peers()`'s own header comment
    // states for an equivalent numbering assumption.
    int local_size = 0, local_rank = 0;
    MPI_Comm_size(local, &local_size);
    MPI_Comm_rank(local, &local_rank);
    const fuse::RankSet& own_side =
        plan.is_writer ? channels.front().writer_ranks : channels.front().reader_ranks;
    if (static_cast<std::size_t>(local_size) != own_side.size() ||
        own_side.index_of(world_rank) != local_rank) {
      CopyErrorMessage(
          "bruss_lowfive_attach(): physics_comm_f's own size/rank does not "
          "match this rank's own side of the graph's RankSet; passed the "
          "wrong communicator",
          errbuf, errbuf_len);
      return 1;
    }
    MPI_Comm intercomm = MPI_COMM_NULL;
    MPI_Intercomm_create(local, plan.local_leader, MPI_COMM_WORLD, plan.remote_leader, plan.tag,
                         &intercomm);
    int is_inter = 0;
    MPI_Comm_test_inter(intercomm, &is_inter);
    if (!is_inter) {
      CopyErrorMessage(
          "MPI_Intercomm_create() did not produce a real intercommunicator; "
          "split_color/local_leader/remote_leader/tag do not match between "
          "this process's own rendezvous plan and its peer's",
          errbuf, errbuf_len);
      return 1;
    }
    LowFive::DistMetadataVOL& vol = LowFive::DistMetadataVOL::create_DistMetadataVOL(local, intercomm);
    bool ok = false;
    ApplyBindings(vol, table.value(), &ok, &build_error);
    if (!ok) {
      CopyErrorMessage(build_error, errbuf, errbuf_len);
      return 1;
    }
    if (!BuildVerifiedFapl(vol, &fapl, &build_error)) {
      CopyErrorMessage(build_error, errbuf, errbuf_len);
      return 1;
    }
    s.dist_vol = &vol;
    distributed_now = true;
#else
    CopyErrorMessage(
        "bruss_lowfive_attach(): this process needs LowFive::DistMetadataVOL "
        "(a cross-process LowFive-bound channel was found) but this build "
        "has no liblowfive-dist (FUSE_LOWFIVE_DIST_LIB_FOUND not set); "
        "rebuild against a LowFive install that ships the dist archive "
        "(cmake/FindLowFive.cmake's own LOWFIVE_HAVE_DIST probe)",
        errbuf, errbuf_len);
    return 1;
#endif
  } else {
    LowFive::MetadataVOL& vol = LowFive::MetadataVOL::create_MetadataVOL();
    bool ok = false;
    ApplyBindings(vol, table.value(), &ok, &build_error);
    if (!ok) {
      CopyErrorMessage(build_error, errbuf, errbuf_len);
      return 1;
    }
    // Without `set_keep(true)`, LowFive discards a file's metadata at
    // H5Fclose and the read leg's H5Fopen finds nothing -- the identical
    // reasoning VolSingleton::VolSingleton's own comment gives.
    vol.set_keep(true);
    if (!BuildVerifiedFapl(vol, &fapl, &build_error)) {
      CopyErrorMessage(build_error, errbuf, errbuf_len);
      return 1;
    }
    distributed_now = false;
  }

  s.fapl = fapl;
  s.distributed = distributed_now;
  s.attached = true;
  if (fapl_id_out) *fapl_id_out = static_cast<long long>(fapl);
  return 0;
#endif  // FUSE_HAVE_MPI
}

int bruss_lowfive_serve_after_write(char* errbuf, long long errbuf_len) {
  AttachState& s = State();
  if (!s.attached) {
    CopyErrorMessage(
        "bruss_lowfive_serve_after_write(): bruss_lowfive_attach() was never "
        "called successfully in this process",
        errbuf, errbuf_len);
    return 1;
  }
  if (!s.distributed) return 0;  // single-process: nothing to serve
#if FUSE_EXAMPLES_LOWFIVE_ATTACH_HAVE_DIST
  if (!s.dist_vol) {
    CopyErrorMessage("internal: distributed attach state with no DistMetadataVOL pointer",
                     errbuf, errbuf_len);
    return 1;
  }
  s.dist_vol->serve_all();
  return 0;
#else
  CopyErrorMessage("internal: distributed attach state with no DistMetadataVOL support built",
                   errbuf, errbuf_len);
  return 1;
#endif
}

int bruss_lowfive_receive_before_read(char* errbuf, long long errbuf_len) {
  AttachState& s = State();
  if (!s.attached) {
    CopyErrorMessage(
        "bruss_lowfive_receive_before_read(): bruss_lowfive_attach() was "
        "never called successfully in this process",
        errbuf, errbuf_len);
    return 1;
  }
  if (!s.distributed) return 0;  // single-process: nothing to receive
#if FUSE_EXAMPLES_LOWFIVE_ATTACH_HAVE_DIST
  if (!s.dist_vol) {
    CopyErrorMessage("internal: distributed attach state with no DistMetadataVOL pointer",
                     errbuf, errbuf_len);
    return 1;
  }
  s.dist_vol->broadcast_files();
  return 0;
#else
  CopyErrorMessage("internal: distributed attach state with no DistMetadataVOL support built",
                   errbuf, errbuf_len);
  return 1;
#endif
}

}  // extern "C"

#else  // !FUSE_HAVE_LOWFIVE

#include <cstring>
#include <string>

namespace {
void CopyErrorMessageNoLowFive(char* errbuf, long long errbuf_len) {
  static const char kMsg[] = "not built with LowFive (FUSE_ENABLE_LOWFIVE was OFF)";
  if (!errbuf || errbuf_len <= 0) return;
  const auto cap = static_cast<std::size_t>(errbuf_len - 1);
  const std::size_t n = sizeof(kMsg) - 1 < cap ? sizeof(kMsg) - 1 : cap;
  std::memcpy(errbuf, kMsg, n);
  errbuf[n] = '\0';
}
}  // namespace

extern "C" {

int bruss_lowfive_attach(const char*, long long, long long* fapl_id_out, char* errbuf,
                         long long errbuf_len) {
  if (fapl_id_out) *fapl_id_out = -1;
  CopyErrorMessageNoLowFive(errbuf, errbuf_len);
  return 1;
}

int bruss_lowfive_serve_after_write(char* errbuf, long long errbuf_len) {
  CopyErrorMessageNoLowFive(errbuf, errbuf_len);
  return 1;
}

int bruss_lowfive_receive_before_read(char* errbuf, long long errbuf_len) {
  CopyErrorMessageNoLowFive(errbuf, errbuf_len);
  return 1;
}

}  // extern "C"

#endif  // FUSE_HAVE_LOWFIVE
