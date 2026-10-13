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
#include <lowfive/log.hpp>
#include <lowfive/vol-metadata.hpp>
#if defined(FUSE_LOWFIVE_DIST_LIB_FOUND) && FUSE_HAVE_MPI
#include <lowfive/vol-dist-metadata.hpp>
#endif

#include <hdf5.h>

#if FUSE_HAVE_MPI
#include <mpi.h>
#endif

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
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
  // ROUND 5 (diagnostic trace track): a base-class pointer valid in
  // EITHER branch of bruss_lowfive_attach() (LowFive::DistMetadataVOL
  // derives from LowFive::MetadataVOL, and MetadataVOL::files is what
  // bruss_lowfive_attach_file_count() reads -- lowfive_attach.h's own
  // doc comment on that function has the full citation) -- never
  // owned/deleted, same reasoning as VolSingleton's own note on
  // dist_vol below.
  LowFive::MetadataVOL* vol = nullptr;
#if FUSE_EXAMPLES_LOWFIVE_ATTACH_HAVE_DIST
  LowFive::DistMetadataVOL* dist_vol = nullptr;  // never owned/deleted; see VolSingleton's own note
  // Round 2 ("role-ordered serve"): the pure state machine this
  // process's own write/read events drive -- meaningful only when
  // `distributed` is true (single-process mode never defers anything,
  // since LowFive::MetadataVOL has no serve_on_close/serve_all() of
  // this kind at all).
  fuse_examples::lowfive_attach::ServeState serve_state;
  // ROUND 6: this component's own exported file name(s) -- lowfive_
  // attach_decision.hpp's own FindOwnExportedFiles(), computed once at
  // attach time from the SAME binding table/this_name the attach
  // already derives, never recomputed per write/read call. Dropped,
  // one at a time, right after each is actually served (this file's
  // own two call sites below) -- never the peer's own file, which this
  // rank never owns the lifetime of.
  std::vector<std::string> own_exported_files;
  // ROUND 8: the per-process LOCAL order assertion's own tracker
  // (lowfive_attach_decision.hpp's own CheckAndAdvanceOrder()) --
  // checked, and advanced, before EVERY serve_state_step() call below,
  // never after (task 3b's own "before the next blocking call").
  fuse_examples::lowfive_attach::OrderTracker order_tracker;
  // ROUND 8, task 6: a monotonically increasing per-process event
  // index (every real hook call below increments it once, in trace
  // order) -- distinct from order_tracker's own period_index/slot
  // (which resets every period); this one never resets, so two trace
  // lines can be compared for "did anything happen in between" even
  // across periods.
  long long event_index = 0;
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
                   bool* ok, std::string* error_out,
                   std::vector<std::pair<std::string, std::string>>* applied_out = nullptr) {
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
    // ROUND 5 (diagnostic trace track): the attach-time trace (item 1
    // of the brief) wants the EXACT (file, dataset) pairs passed to
    // set_memory() above -- collecting them here, in the one place
    // that already computes the normalized value, is the single
    // source of truth; re-deriving the same list a second time at the
    // trace call site would risk drifting out of sync with what this
    // function actually applied.
    if (applied_out) applied_out->emplace_back(split.value().first, normalized.value());
  }
}

// ROUND 5 (diagnostic trace track): item 1's own cached-bool
// requirement ("Keep the trace code out of the hot path when
// disabled") -- getenv() runs exactly once per process, the FIRST
// time ANY of this file's/shim.cpp's own trace call sites asks.
// fuse_examples::lowfive_attach::ParseTraceEnabledEnv() is the pure
// half (lowfive_attach_decision.hpp); this wraps it with the one
// process-wide getenv() call neither that function nor a unit test
// can perform.
bool TraceEnabledCached() {
  static const bool enabled =
      fuse_examples::lowfive_attach::ParseTraceEnabledEnv(std::getenv("FUSE_LOWFIVE_ATTACH_TRACE"));
  return enabled;
}

// ROUND 6: `LowFive::MetadataVOL::drop(std::string)` (src/vol-
// metadata.cpp, cited in lowfive_attach_decision.hpp's own "ROOT CAUSE"
// comment) dereferences `files.find(name)` WITHOUT checking `end()` --
// undefined behaviour on an absent name. This wrapper guards that;
// `files` is a PUBLIC member (confirmed round 5, reused here). Absent
// is treated as a harmless no-op, not a refusal: this round's own
// protocol only ever calls this right after a file this SAME rank just
// served, so the name should always be present -- but a double-drop
// (a future bug, or a mutation of the call-site guard this round's own
// mutation rows below test for) finding the file already gone is
// exactly what we wanted anyway, not a new problem to report.
template <typename Vol>
void GuardedDropOwnExportedFile(Vol& vol, const std::string& file) {
  if (vol.files.find(file) != vol.files.end()) vol.drop(file);
}

// ROUND 5 (diagnostic trace track), item 5 of the brief: `FUSE_LOWFIVE_
// LOG=<level>` turns on LowFive's own spdlog-based tracing
// (`LowFive::create_logger(level)`, include/lowfive/log.hpp) -- "With
// 'trace' LowFive prints 'enter serve_all, files.size = N' and Index
// logs, which settle H2/H4 directly" (vol-dist-metadata.cpp:42,
// src/vol-dist/index.cpp:13 on this host's own LowFive checkout, both
// read above). Read src/log.cpp first: `create_logger()` calls
// `spd::stderr_color_mt("lowfive")`, which THROWS `spdlog::spdlog_ex`
// (derived from std::exception) if a logger named "lowfive" is already
// registered in spdlog's own global registry -- `get_logger()`
// (log-private.hpp, NOT an installed/public header, so this file
// cannot call it) only ever builds a local, unregistered null-sink
// fallback when "lowfive" is not found, so it does NOT register
// anything itself; the only way "lowfive" could already be registered
// when this runs is a SECOND call to create_logger() in the same
// process. bruss_lowfive_attach() already refuses a second call in
// this process (AttachState's own doc comment), so in THIS codebase's
// current usage that second call cannot happen -- caught here anyway,
// by catching any std::exception rather than naming spdlog::spdlog_ex
// explicitly (which would need <spdlog/spdlog.h> on this translation
// unit's own include path; this file has no other reason to add that
// dependency, so it does not), because the lead's own brief asked this
// to be handled, not assumed unreachable. On the caught-duplicate path
// this function does NOT call set_level() on the already-registered
// logger (no handle to it without spdlog::get(), which needs the same
// header): that logger keeps whichever level ITS OWN create_logger()
// call set, which may not match THIS process's own FUSE_LOWFIVE_LOG
// request -- a known, documented limitation of the minimal-change
// approach, not a silent bug.
void EnableLowFiveLoggerOnce() {
  const char* level = std::getenv("FUSE_LOWFIVE_LOG");
  if (!level || level[0] == '\0') return;
  try {
    LowFive::create_logger(std::string(level));
  } catch (const std::exception&) {
    // Already registered by an earlier call in this process (see this
    // function's own doc comment above) -- nothing more this file can
    // do without <spdlog/spdlog.h>.
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
  // ROUND 5 (diagnostic trace track), item 5: before any LowFive VOL is
  // constructed, so its own ctor-time trace lines (if FUSE_LOWFIVE_LOG
  // is set) are captured too.
  EnableLowFiveLoggerOnce();

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

  // ROUND 8, task 3a: the STATIC order check, at attach time, before
  // decide_attach_mode() or anything else that could block -- refuses
  // by name (E-LOWFIVE-ATTACH-ORDER-UNVALIDATED) unless this graph's
  // own channel order/component names/rank counts EXACTLY match the
  // one measured, validated configuration. Collective-safe: `channels`
  // is already the SAME, rank-independent data decide_attach_mode()'s
  // own INCONSISTENT check below relies on (every rank of both
  // components loads the identical graph file and runs the identical
  // loop above), so every rank reaches the identical verdict.
  {
    const auto signature = fuse_examples::lowfive_attach::ComputeGraphOrderSignature(channels);
    auto order_ok = fuse_examples::lowfive_attach::CheckGraphOrderValidated(signature);
    if (TraceEnabledCached()) {
      const std::string verdict =
          order_ok.ok() ? std::string("accepted") : ("refused: " + order_ok.error().code);
      std::fprintf(stderr, "TRACE(bruss_lowfive_attach): order-check-verdict=%s\n",
                   verdict.c_str());
    }
    if (!order_ok.ok()) {
      CopyErrorMessage("[" + order_ok.error().code + "] " + order_ok.error().message, errbuf,
                       errbuf_len);
      return 1;
    }
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
    //
    // ROUND 6: `serve_on_close=false` ALONE is not enough -- lowfive_
    // attach_decision.hpp's own "ROOT CAUSE" comment has the full
    // account, measured on the host: without `keep=true` too, `H5Fclose
    // ()`'s own tail call into `MetadataVOL::file_close()` erases and
    // deletes the file BEFORE this process's own explicit serve_after_
    // write() hook ever runs, so that hook's own serve_all() finds
    // files.size()==0 and returns without serving (the round-5 hang).
    // `decide_vol_settings()` is the ONE place that decides BOTH
    // settings together, so the invariant it keeps
    // (VolSettingsSatisfyInvariant()) is checked once, not re-derived
    // at each of this function's own two branches.
    const auto vol_settings = fuse_examples::lowfive_attach::decide_vol_settings(/*distributed=*/true);
    vol.serve_on_close = vol_settings.serve_on_close;
    vol.set_keep(vol_settings.keep);
    bool ok = false;
    std::vector<std::pair<std::string, std::string>> applied_memory_patterns;
    ApplyBindings(vol, table.value(), &ok, &build_error, &applied_memory_patterns);
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
    s.vol = &vol;
    distributed_now = true;
    // ROUND 6: the file(s) this rank's own two serve call sites below
    // must drop, one at a time, right after each is actually served --
    // computed once, here, from the SAME table/this_name already in
    // scope (lowfive_attach_decision.hpp's own FindOwnExportedFiles()
    // doc comment has the full account of why this, never a hardcoded
    // literal).
    auto own_files = fuse_examples::lowfive_attach::FindOwnExportedFiles(table.value(), this_name);
    if (!own_files.ok()) {
      CopyErrorMessage("[" + own_files.error().code + "] " + own_files.error().message, errbuf,
                       errbuf_len);
      return 1;
    }
    s.own_exported_files = own_files.value();
    // ROUND 5 (diagnostic trace track), item 1 of the brief: one line
    // with everything the lead asked for -- world rank, component,
    // role, distributed flag, the registered (file, dataset_pattern,
    // intercomm_index) list, the (file, dataset) memory patterns, and
    // the serve_on_close value actually set. `intercomms.size()`/
    // `files.size()` are not in the brief's own enumerated list but are
    // directly diagnostic for H4 (wrong/empty intercomm selection) and
    // cost nothing extra to read (both are public members, confirmed
    // from the installed headers) -- included for that reason. ROUND 6,
    // item 6 of the brief: `keep=` added alongside the already-printed
    // `serve_on_close=`, so the next host run's own attach-time line
    // shows the full pair this round's own fix applies together.
    if (TraceEnabledCached()) {
      std::fprintf(stderr,
                   "TRACE(bruss_lowfive_attach): world_rank=%d component=%s role=%s "
                   "distributed=1 registrations=%s memory_patterns=%s serve_on_close=%d keep=%d "
                   "intercomms.size=%zu files.size=%zu\n",
                   world_rank, this_name.c_str(),
                   s.serve_state.role == fuse_examples::lowfive_attach::ServeRole::kFirstServer
                       ? "first-server"
                       : "first-reader",
                   fuse_examples::lowfive_attach::FormatRegistrationsForTrace(registrations.value())
                       .c_str(),
                   fuse_examples::lowfive_attach::FormatMemoryPatternsForTrace(applied_memory_patterns)
                       .c_str(),
                   vol.serve_on_close ? 1 : 0, vol.keep ? 1 : 0, vol.intercomms.size(),
                   vol.files.size());
    }
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
    std::vector<std::pair<std::string, std::string>> applied_memory_patterns;
    ApplyBindings(vol, table.value(), &ok, &build_error, &applied_memory_patterns);
    if (!ok) {
      CopyErrorMessage(build_error, errbuf, errbuf_len);
      return 1;
    }
    // Without `set_keep(true)`, LowFive discards a file's metadata at
    // H5Fclose and the read leg's H5Fopen finds nothing -- the identical
    // reasoning VolSingleton::VolSingleton's own comment gives.
    //
    // ROUND 6: unchanged by this round, per the lead's own instruction
    // -- this branch already needed, and already had, exactly what
    // decide_vol_settings(false).keep now names explicitly for the
    // distributed branch (both branches' own `keep` requirement traces
    // to the SAME MetadataVOL::file_close() behaviour this header's own
    // "ROOT CAUSE" comment cites). `serve_on_close` does not apply here
    // at all: `LowFive::MetadataVOL` (this branch's own VOL type, unlike
    // the distributed branch's DistMetadataVOL) has no such member, and
    // nothing in this branch ever calls serve_all()/drop() either --
    // this VOL never serves anyone, so files simply accumulate in
    // `files` for this process's own lifetime (the single-process path
    // round 4's own citation already measured end to end; this round
    // does not revisit that measurement).
    vol.set_keep(true);
    if (!BuildVerifiedFapl(vol, &fapl, &build_error)) {
      CopyErrorMessage(build_error, errbuf, errbuf_len);
      return 1;
    }
    distributed_now = false;
    s.vol = &vol;
    // ROUND 5 (diagnostic trace track): the single-process branch has
    // no role/serve choreography at all (LowFive::MetadataVOL has no
    // serve_on_close/serve_all()), so this line is deliberately
    // shorter than the distributed branch's own -- there is nothing
    // else to report here that the distributed trace's own fields
    // would mean anything for.
    if (TraceEnabledCached()) {
      std::fprintf(stderr,
                   "TRACE(bruss_lowfive_attach): world_rank=%d distributed=0 "
                   "memory_patterns=%s files.size=%zu\n",
                   world_rank,
                   fuse_examples::lowfive_attach::FormatMemoryPatternsForTrace(applied_memory_patterns)
                       .c_str(),
                   vol.files.size());
    }
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
  // ROUND 8, task 3b: the LOCAL per-event order assertion, checked and
  // advanced BEFORE serve_state_step() -- a deviation is refused here,
  // before the kServeNow branch below's own blocking serve_all() call
  // ever runs, never after.
  {
    auto order_check = fuse_examples::lowfive_attach::CheckAndAdvanceOrder(
        s.serve_state.role, s.order_tracker,
        fuse_examples::lowfive_attach::ServeEvent::kExportWrittenAndClosed);
    if (!order_check.ok()) {
      CopyErrorMessage("[" + order_check.error().code + "] " + order_check.error().message,
                       errbuf, errbuf_len);
      return 1;
    }
    s.order_tracker = order_check.value();
  }
  // ROUND 8, task 6: the policy decision and its own reason, BEFORE
  // calling serve_state_step() (which mutates serve_state.deferred_
  // pending but not completed_any_read for THIS event) -- decide_
  // serve_action()'s own condition, read directly, not re-derived.
  const bool cold_start_first_reader =
      s.serve_state.role == fuse_examples::lowfive_attach::ServeRole::kFirstReader &&
      !s.serve_state.completed_any_read;
  const long long this_event_index = s.event_index++;
  if (TraceEnabledCached()) {
    std::fprintf(stderr,
                 "TRACE(bruss_lowfive_serve_after_write): event_index=%lld period=%d policy=%s\n",
                 this_event_index, s.order_tracker.period_index,
                 cold_start_first_reader ? "cold-start-first-reader" : "steady-state");
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
  // ROUND 5 (diagnostic trace track), item 4 of the brief: one line
  // before and after each serve_all() call, with the action taken and
  // monotonic elapsed seconds -- "a serve_all that returns in
  // microseconds is the H2 signature; one that blocks until the peer
  // opens is the healthy signature." The kDeferServe branch also
  // traces (no serve_all() call at all), so H3 ("the hook returned
  // another action") shows up as a trace line with NO matching "exit
  // elapsed=" line, as distinct from H2's own near-zero elapsed line.
  switch (action.value()) {
    case fuse_examples::lowfive_attach::ServeAction::kServeNow: {
      if (TraceEnabledCached())
        std::fprintf(stderr, "TRACE(bruss_lowfive_serve_after_write): action=kServeNow enter\n");
      const auto t0 = std::chrono::steady_clock::now();
      s.dist_vol->serve_all();
      // ROUND 6: this component's own exported file(s) are only now
      // safe to drop -- SERVED, not merely written -- so the next
      // period's own H5Fcreate(TRUNC) of the same name does not meet a
      // stale copy (lowfive_attach_decision.hpp's own "ROOT CAUSE"
      // comment has the full citation on file_create()'s own emplace()-
      // on-duplicate-key behaviour).
      for (const auto& f : s.own_exported_files) GuardedDropOwnExportedFile(*s.dist_vol, f);
      if (TraceEnabledCached()) {
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr,
                     "TRACE(bruss_lowfive_serve_after_write): action=kServeNow exit elapsed=%.6fs "
                     "files.size_after_drop=%zu\n",
                     elapsed, s.dist_vol->files.size());
      }
      return 0;
    }
    case fuse_examples::lowfive_attach::ServeAction::kDeferServe:
      if (TraceEnabledCached())
        std::fprintf(stderr,
                     "TRACE(bruss_lowfive_serve_after_write): action=kDeferServe (no serve_all "
                     "call; flushed at this side's own next kImportClosed)\n");
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
  //
  // ROUND 8, task 3b: checked (NOT advanced -- CheckAndAdvanceOrder()'s
  // own doc comment on why kImportAboutToOpen only checks) BEFORE this
  // process's own H5Fopen, which runs right after this call returns --
  // "before the next blocking call" for the READ side specifically.
  {
    auto order_check = fuse_examples::lowfive_attach::CheckAndAdvanceOrder(
        s.serve_state.role, s.order_tracker,
        fuse_examples::lowfive_attach::ServeEvent::kImportAboutToOpen);
    if (!order_check.ok()) {
      CopyErrorMessage("[" + order_check.error().code + "] " + order_check.error().message,
                       errbuf, errbuf_len);
      return 1;
    }
    s.order_tracker = order_check.value();
  }
  const long long this_event_index = s.event_index++;
  if (TraceEnabledCached()) {
    std::fprintf(stderr,
                 "TRACE(bruss_lowfive_receive_before_read): event_index=%lld period=%d "
                 "policy=receive-only\n",
                 this_event_index, s.order_tracker.period_index);
  }
  auto action = fuse_examples::lowfive_attach::serve_state_step(
      &s.serve_state, fuse_examples::lowfive_attach::ServeEvent::kImportAboutToOpen);
  if (!action.ok()) {
    CopyErrorMessage("[" + action.error().code + "] " + action.error().message, errbuf,
                     errbuf_len);
    return 1;
  }
  // ROUND 5 (diagnostic trace track), item 4: this hook never calls
  // serve_all() at all (ROUND 2 REVIEW's own account above), so its
  // own trace line is a single "enter/is about to open" marker with no
  // matching elapsed-time line -- a real H5Fopen-blocks-here signature
  // in the surrounding write/read trace (shim.cpp) is what settles
  // whether the peer is actually serving at this point, not this hook.
  switch (action.value()) {
    case fuse_examples::lowfive_attach::ServeAction::kReceiveOnly:
      if (TraceEnabledCached())
        std::fprintf(stderr,
                     "TRACE(bruss_lowfive_receive_before_read): action=kReceiveOnly (nothing "
                     "LowFive-level to do; H5Fopen performs the real receive)\n");
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
  //
  // ROUND 8, task 3b: checked and advanced BEFORE serve_state_step(),
  // same discipline as the write side above -- a deviation is refused
  // before the kFlushDeferred branch's own blocking serve_all() call.
  {
    auto order_check = fuse_examples::lowfive_attach::CheckAndAdvanceOrder(
        s.serve_state.role, s.order_tracker,
        fuse_examples::lowfive_attach::ServeEvent::kImportClosed);
    if (!order_check.ok()) {
      CopyErrorMessage("[" + order_check.error().code + "] " + order_check.error().message,
                       errbuf, errbuf_len);
      return 1;
    }
    s.order_tracker = order_check.value();
  }
  const long long this_event_index = s.event_index++;
  if (TraceEnabledCached()) {
    std::fprintf(stderr,
                 "TRACE(bruss_lowfive_after_read_close): event_index=%lld period=%d "
                 "policy=%s\n",
                 this_event_index, s.order_tracker.period_index,
                 s.serve_state.deferred_pending ? "flush-deferred" : "noop-after-import");
  }
  auto action = fuse_examples::lowfive_attach::serve_state_step(
      &s.serve_state, fuse_examples::lowfive_attach::ServeEvent::kImportClosed);
  if (!action.ok()) {
    CopyErrorMessage("[" + action.error().code + "] " + action.error().message, errbuf,
                     errbuf_len);
    return 1;
  }
  // ROUND 5 (diagnostic trace track), item 4: same before/after/
  // elapsed-time shape as serve_after_write()'s own kServeNow case
  // above -- this is the FIRST-READER's own (deferred) serve_all()
  // call, the other of the two places this process ever blocks in it.
  switch (action.value()) {
    case fuse_examples::lowfive_attach::ServeAction::kFlushDeferred: {
      if (TraceEnabledCached())
        std::fprintf(stderr, "TRACE(bruss_lowfive_after_read_close): action=kFlushDeferred enter\n");
      const auto t0 = std::chrono::steady_clock::now();
      s.dist_vol->serve_all();
      // ROUND 6: same reasoning as serve_after_write()'s own kServeNow
      // case above -- this is the FIRST-READER's own deferred serve,
      // now actually served, so its own exported file is safe to drop.
      for (const auto& f : s.own_exported_files) GuardedDropOwnExportedFile(*s.dist_vol, f);
      if (TraceEnabledCached()) {
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr,
                     "TRACE(bruss_lowfive_after_read_close): action=kFlushDeferred exit elapsed=%.6fs "
                     "files.size_after_drop=%zu\n",
                     elapsed, s.dist_vol->files.size());
      }
      return 0;
    }
    case fuse_examples::lowfive_attach::ServeAction::kNoopAfterImport:
      if (TraceEnabledCached())
        std::fprintf(stderr,
                     "TRACE(bruss_lowfive_after_read_close): action=kNoopAfterImport (no serve_all "
                     "call)\n");
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

// ROUND 5 (diagnostic trace track): lowfive_attach.h's own doc comment
// on each of these three has the full account of what it reports and
// why it is shaped this way; TraceEnabledCached()/ParseTraceEnabledEnv()
// are the single getenv()/pure-parse pair every trace call site (this
// file and shim.cpp) goes through.
int bruss_lowfive_trace_enabled(void) { return TraceEnabledCached() ? 1 : 0; }

long long bruss_lowfive_attach_file_count(void) {
  AttachState& s = State();
  if (!s.attached || !s.vol) return -1;
  return static_cast<long long>(s.vol->files.size());
}

// ROUND 6: see lowfive_attach.h's own doc comment on this function for
// why it exists (replacing the round-5 trace's unreliable H5Lexists()
// check) and why it is FILE-level, not dataset-level.
int bruss_lowfive_attach_has_real_file(const char* file) {
  AttachState& s = State();
  if (!s.attached || !s.vol || !file) return -1;
  return s.vol->has_real_file(file) ? 1 : 0;
}

void bruss_lowfive_trace_skip(const char* which, long long window_samples, long long count,
                              long long components) {
  if (!TraceEnabledCached()) return;
  std::fprintf(stderr,
               "TRACE(%s): SKIPPED (window_samples=%lld count=%lld components=%lld -- the "
               "window_samples > 0 && data guard was false)\n",
               which ? which : "?", window_samples, count, components);
}

}  // extern "C"

#else  // !FUSE_HAVE_LOWFIVE

#include <cstdio>
#include <cstdlib>
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

// ROUND 5 (diagnostic trace track): the three functions below do not
// depend on HDF5/LowFive at all (plain getenv()/fprintf()), so they
// work the same in this build configuration -- u_main.c's/v_main.f90's
// own skip-guard trace (item 3 of the brief) fires regardless of
// -DFUSE_ENABLE_LOWFIVE, same as bruss_lowfive_available() above.
// This duplicates (rather than shares) the env-parsing logic the
// FUSE_HAVE_LOWFIVE branch's own TraceEnabledCached() uses, because
// that branch's own pure half (ParseTraceEnabledEnv(),
// lowfive_attach_decision.hpp) is only included under `#if
// FUSE_HAVE_LOWFIVE` above (it pulls in fuse/base/result.hpp and
// friends this translation unit has no other reason to need here) --
// restructuring that include guard just to share one string
// comparison was judged not worth the risk for this round.
namespace {
bool TraceEnabledCachedNoLowFive() {
  static const bool enabled = [] {
    const char* raw = std::getenv("FUSE_LOWFIVE_ATTACH_TRACE");
    return raw != nullptr && raw[0] == '1' && raw[1] == '\0';
  }();
  return enabled;
}
}  // namespace

int bruss_lowfive_trace_enabled(void) { return TraceEnabledCachedNoLowFive() ? 1 : 0; }

long long bruss_lowfive_attach_file_count(void) {
  // No LowFive VOL exists in this build at all -- never "attached", so
  // always the same "not available" answer bruss_lowfive_attach()
  // itself gives (fapl_id_out = -1).
  return -1;
}

// ROUND 6: same reasoning as bruss_lowfive_attach_file_count() above.
int bruss_lowfive_attach_has_real_file(const char*) { return -1; }

void bruss_lowfive_trace_skip(const char* which, long long window_samples, long long count,
                              long long components) {
  if (!TraceEnabledCachedNoLowFive()) return;
  std::fprintf(stderr,
               "TRACE(%s): SKIPPED (window_samples=%lld count=%lld components=%lld -- the "
               "window_samples > 0 && data guard was false)\n",
               which ? which : "?", window_samples, count, components);
}

}  // extern "C"

#endif  // FUSE_HAVE_LOWFIVE
