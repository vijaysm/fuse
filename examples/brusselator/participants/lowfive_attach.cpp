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
// DEADLOCK-AVOIDANCE DECISION -- ROUND 2, SECOND ATTEMPT, supersedes
// both round 1's own ruling AND this round's own FIRST attempt (both
// quoted in full in KNOWN_ISSUES.md D39 for the record -- including the
// first attempt's own claim, right here, that role-ordered serving was
// "the one order that cannot leave both sides waiting on each other".
// That claim was WRONG and is not repeated below).
//
// Round 1 placed serve_after_write()/receive_before_read() at the
// "natural" points -- serve right after H5Fclose, receive right before
// H5Fopen -- and the host measured a 120s hang: `serve_on_close`
// (include/lowfive/vol-dist-metadata.hpp:33) defaults true, so `file_
// close()` (src/vol-dist-metadata.cpp:320-364) already calls `serve_
// all()` INSIDE H5Fclose, before round 1's own hook call (placed AFTER
// H5Fclose) ever ran -- both sides raced into `serve_all()` together,
// neither having received first.
//
// This round's FIRST attempt set `serve_on_close = false` and moved the
// first-reader's own flush into the PRE-open hook (`receive_before_
// read()`, called right before H5Fopen) -- and was reviewed against
// LowFive's own source and rejected before any host run: a process
// blocked inside `serve_all()` for file K is released only when the PEER
// completes its own OPEN-AND-CLOSE of file K (`DistMetadataVOL::
// file_close()`'s own `RemoteFile` branch calling `finish(0)`, src/
// vol-dist-metadata.cpp line ~342) -- not by the peer merely reaching a
// pre-open hook. Flushing there ran the first-reader's own blocking
// serve BEFORE it had done anything that could release the first-
// server's own pending one, and vice versa: circular wait, one write/
// read cycle later than round 1's, same two LowFive-level stacks.
// lowfive_attach_decision.hpp's own "ROUND 2 REVIEW" comment has the
// full citations and the corrected argument.
//
// THE FIX, this version: `serve_on_close = false` is kept (below); the
// flush moves again, this time to a NEW event fired AFTER this
// process's own H5Fclose of the peer's file (`kImportClosed`, a new
// hook called from shim.cpp's own bruss_lowfive_read(), right after
// EVERY H5Fclose on a path where H5Fopen already succeeded -- see
// shim.cpp's own comment for why every such path, not only the success
// path) -- the event LowFive's own source actually ties the release to.
// `receive_before_read()` (the pre-open hook) no longer flushes
// anything and no longer calls `broadcast_files()` either (lowfive_
// attach_decision.hpp's own comment has the citation for why that call
// was never justified for this row's own one-rank-per-component
// layout). This necessarily serializes the two transfers (never both
// sides serving in the same window); the helper-thread follow-up that
// would let them overlap again stays NOT started this round, for the
// same reason recorded before (Index::serve()'s own barrier/ibarrier
// path and Query::get_filenames()'s own broadcast share one `local`
// communicator, and this build's HDF5 is Threadsafety:no with a plain
// MPI_Init) -- KNOWN_ISSUES.md D39 records it.
//
// Checked, this time, against an explicit-state simulation of the
// blocking semantics above (lowfive_attach_model_check.hpp), not only
// against a word-of-mouth argument -- see that header's own top comment
// for exactly what it does and does not establish.
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
  // Round 2 ("role-ordered serve"): the pure state machine this
  // process's own write/read events drive -- meaningful only when
  // `distributed` is true (single-process mode never defers anything,
  // since LowFive::MetadataVOL has no serve_on_close/serve_all() of
  // this kind at all).
  fuse_examples::lowfive_attach::ServeState serve_state;
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
    // ROUND 4, DEFECT B fix: `split.value().second` (straight from
    // split_hdf5_path(), e.g. "grid/u_out") has no leading slash, but
    // LowFive::Object::fullname() -- what MetadataVOL::dataset_create()'s
    // own is_memory check actually compares against -- always produces
    // one ("/grid/u_out"). Without normalizing, is_memory is false for
    // every Brusselator dataset this attach registers, and the dataset
    // is never attached as in-memory at all: see lowfive_attach_
    // decision.hpp's own "DEFECT B" comment for the full citations.
    // Applied here, in the ONE place both the distributed and the
    // single-process branches of bruss_lowfive_attach() already share,
    // rather than at either call site -- both need the same fix.
    auto normalized =
        fuse_examples::lowfive_attach::NormalizeDatasetPattern(split.value().second);
    if (!normalized.ok()) {
      *ok = false;
      *error_out = "NormalizeDatasetPattern('" + split.value().second + "'): [" +
                  normalized.error().code + "] " + normalized.error().message;
      return;
    }
    vol.set_memory(split.value().first, normalized.value());
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
    roles.writer_name = writer_c.name;
    roles.reader_name = reader_c.name;
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
    // ROUND 2 ("role-ordered serve"), the measured fix: `serve_on_close`
    // is a PUBLIC bool member, default true (lowfive/vol-dist-
    // metadata.hpp:33 on this host's own /Users/mahadevan/Code/hdf5/
    // LowFive, rev 7f3a58b), and `DistMetadataVOL::file_close()` calls
    // `serve_all()` for every local in-memory file whenever it is true
    // (vol-dist-metadata.cpp:355-358) -- BEFORE this process's own code
    // ever runs a line, which made round 1's own bruss_lowfive_serve_
    // after_write() (placed AFTER H5Fclose) a no-op and is the measured
    // cause of the host's own 120s hang (lowfive_attach_decision.hpp's
    // own "Round 2" comment has the full stack). Setting it false here
    // makes serving an explicit decision this file's own serve_state_
    // step() calls make, never H5Fclose's.
    vol.serve_on_close = false;
    bool ok = false;
    ApplyBindings(vol, table.value(), &ok, &build_error);
    if (!ok) {
      CopyErrorMessage(build_error, errbuf, errbuf_len);
      return 1;
    }
    // ROUND 4, DEFECT A fix: nothing before this line ever called
    // LowFive's own DistMetadataVOL::set_intercomm(), so a CONSUMER's
    // first remote H5Fopen on this process threw file_open(): no
    // intercomm found -- the exact exception the host measured on both
    // sides. lowfive_attach_decision.hpp's own "DEFECT A"/"DEFECT B"
    // comment has the full citations for both this fix and the
    // normalization already applied inside ApplyBindings() above.
    // Intercomm index 0 is explicit here, not hidden inside
    // BuildIntercommRegistrations() itself -- see that function's own
    // doc for why this attach's own architecture can never need a
    // second index (decide_attach_mode() already refuses, by name, any
    // graph requiring one), and why a design that ever did would need a
    // different function signature, not a different value passed here.
    auto registrations =
        fuse_examples::lowfive_attach::BuildIntercommRegistrations(table.value(), /*intercomm_index=*/0);
    if (!registrations.ok()) {
      CopyErrorMessage("[" + registrations.error().code + "] " + registrations.error().message,
                       errbuf, errbuf_len);
      return 1;
    }
    for (const auto& reg : registrations.value()) {
      vol.set_intercomm(reg.file, reg.dataset_pattern, reg.intercomm_index);
    }
    if (!BuildVerifiedFapl(vol, &fapl, &build_error)) {
      CopyErrorMessage(build_error, errbuf, errbuf_len);
      return 1;
    }
    // Role-ordered serve (round 2): which of this channel's own two
    // COMPONENTS serves immediately at write and which defers -- from
    // their own RankSets alone, identical on every rank of either
    // component (the collective-safety requirement serve_all()'s own
    // local.barrier() places on this decision). `this_name` is simply
    // whichever side (writer/reader) `plan.is_writer` already says this
    // rank is on -- the SAME canonical channel decide_attach_mode()
    // already validated every other qualifying channel against.
    const std::string& this_name =
        plan.is_writer ? channels.front().writer_name : channels.front().reader_name;
    auto roles = fuse_examples::lowfive_attach::decide_serve_roles(
        channels.front().writer_name, channels.front().writer_ranks,
        channels.front().reader_name, channels.front().reader_ranks);
    if (!roles.ok()) {
      CopyErrorMessage("[" + roles.error().code + "] " + roles.error().message, errbuf,
                       errbuf_len);
      return 1;
    }
    s.serve_state.role = (this_name == roles.value().first_server_name)
                             ? fuse_examples::lowfive_attach::ServeRole::kFirstServer
                             : fuse_examples::lowfive_attach::ServeRole::kFirstReader;
    s.serve_state.deferred_pending = false;
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
  // Unchanged by the round-2 review: the WRITE side's own event/action
  // pair (kExportWrittenAndClosed -> kServeNow/kDeferServe) was never
  // the part that was wrong -- only the READ side's own flush placement
  // was. See lowfive_attach_decision.hpp's own "ROUND 2 REVIEW" comment.
  auto action = fuse_examples::lowfive_attach::serve_state_step(
      &s.serve_state, fuse_examples::lowfive_attach::ServeEvent::kExportWrittenAndClosed);
  if (!action.ok()) {
    CopyErrorMessage("[" + action.error().code + "] " + action.error().message, errbuf,
                     errbuf_len);
    return 1;
  }
  switch (action.value()) {
    case fuse_examples::lowfive_attach::ServeAction::kServeNow:
      s.dist_vol->serve_all();
      return 0;
    case fuse_examples::lowfive_attach::ServeAction::kDeferServe:
      return 0;  // nothing to do now; flushed at this side's own next kImportClosed
    default:
      CopyErrorMessage(
          "internal: serve_state_step() returned an action not valid for an export event",
          errbuf, errbuf_len);
      return 1;
  }
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
  // ROUND 2 REVIEW: this hook no longer flushes anything, and no longer
  // calls broadcast_files() either -- lowfive_attach_decision.hpp's own
  // "ROUND 2 REVIEW" comment has the full account, with LowFive-source
  // citations, of why: the real receive is H5Fopen itself (about to run,
  // right after this call returns), and broadcast_files() propagates an
  // already-known file registry to a multi-rank component's OWN sibling
  // ranks over `local`, never to the peer component over the intercomm
  // -- not needed for this row's own one-rank-per-component layout, and
  // not reachable from any code path inside DistMetadataVOL::file_open()/
  // file_close() themselves (grep confirms zero internal callers in the
  // LowFive checkout this cites). `serve_state_step()` is still called,
  // purely so the event sequence stays validated end to end (a future
  // mutation that makes kImportAboutToOpen return anything other than
  // kReceiveOnly is caught by this switch's own `default` case, and by
  // lowfive_attach_model_check.hpp's own search).
  auto action = fuse_examples::lowfive_attach::serve_state_step(
      &s.serve_state, fuse_examples::lowfive_attach::ServeEvent::kImportAboutToOpen);
  if (!action.ok()) {
    CopyErrorMessage("[" + action.error().code + "] " + action.error().message, errbuf,
                     errbuf_len);
    return 1;
  }
  switch (action.value()) {
    case fuse_examples::lowfive_attach::ServeAction::kReceiveOnly:
      return 0;  // nothing LowFive-level to do; H5Fopen performs the real receive
    default:
      CopyErrorMessage(
          "internal: serve_state_step() returned an action not valid for kImportAboutToOpen "
          "(expected kReceiveOnly always; see lowfive_attach_decision.hpp's own ROUND 2 "
          "REVIEW comment)",
          errbuf, errbuf_len);
      return 1;
  }
#else
  CopyErrorMessage("internal: distributed attach state with no DistMetadataVOL support built",
                   errbuf, errbuf_len);
  return 1;
#endif
}

int bruss_lowfive_after_read_close(char* errbuf, long long errbuf_len) {
  AttachState& s = State();
  if (!s.attached) {
    CopyErrorMessage(
        "bruss_lowfive_after_read_close(): bruss_lowfive_attach() was "
        "never called successfully in this process",
        errbuf, errbuf_len);
    return 1;
  }
  if (!s.distributed) return 0;  // single-process: nothing to flush
#if FUSE_EXAMPLES_LOWFIVE_ATTACH_HAVE_DIST
  if (!s.dist_vol) {
    CopyErrorMessage("internal: distributed attach state with no DistMetadataVOL pointer",
                     errbuf, errbuf_len);
    return 1;
  }
  // NEW this round, replacing the flush the FIRST round-2 attempt placed
  // in the pre-open hook above: this process has now completed its OWN
  // open-and-close of the peer's file, which -- read from LowFive's own
  // source (lowfive_attach_decision.hpp's own "ROUND 2 REVIEW" comment)
  // -- is exactly the event that would release a PEER's own pending
  // serve_all(); only now is it safe for a first-reader to enter its own.
  auto action = fuse_examples::lowfive_attach::serve_state_step(
      &s.serve_state, fuse_examples::lowfive_attach::ServeEvent::kImportClosed);
  if (!action.ok()) {
    CopyErrorMessage("[" + action.error().code + "] " + action.error().message, errbuf,
                     errbuf_len);
    return 1;
  }
  switch (action.value()) {
    case fuse_examples::lowfive_attach::ServeAction::kFlushDeferred:
      s.dist_vol->serve_all();
      return 0;
    case fuse_examples::lowfive_attach::ServeAction::kNoopAfterImport:
      return 0;
    default:
      CopyErrorMessage(
          "internal: serve_state_step() returned an action not valid for kImportClosed",
          errbuf, errbuf_len);
      return 1;
  }
#else
  CopyErrorMessage("internal: distributed attach state with no DistMetadataVOL support built",
                   errbuf, errbuf_len);
  return 1;
#endif
}

int bruss_lowfive_finalize(char* errbuf, long long errbuf_len) {
  AttachState& s = State();
  // Safe to call unconditionally, including when bruss_lowfive_attach()
  // was never called at all (a plain --transport fuse run) -- NOT an
  // error, unlike serve_after_write/receive_before_read's own
  // "never attached" refusal, because this function's own caller
  // (u_main.c/v_main.f90) calls it once at the very end of main()
  // regardless of --transport, as a hygiene check, not a per-delivery
  // operation gated on an attach having already happened.
  if (!s.attached || !s.distributed) return 0;
#if FUSE_EXAMPLES_LOWFIVE_ATTACH_HAVE_DIST
  auto action = fuse_examples::lowfive_attach::serve_state_step(
      &s.serve_state, fuse_examples::lowfive_attach::ServeEvent::kFinalize);
  if (!action.ok()) {
    CopyErrorMessage("[" + action.error().code + "] " + action.error().message, errbuf,
                     errbuf_len);
    return 1;
  }
  return 0;
#else
  (void)errbuf;
  (void)errbuf_len;
  return 0;
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

int bruss_lowfive_after_read_close(char* errbuf, long long errbuf_len) {
  CopyErrorMessageNoLowFive(errbuf, errbuf_len);
  return 1;
}

int bruss_lowfive_finalize(char*, long long) {
  // Unlike attach/serve/receive, always a harmless no-op here: this
  // build has no LowFive at all, so there is never anything deferred to
  // flush, and the caller (u_main.c/v_main.f90) calls this one
  // unconditionally at the end of main() regardless of --transport.
  return 0;
}

}  // extern "C"

#endif  // FUSE_HAVE_LOWFIVE
