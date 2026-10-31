// See lowfive_attach.h's own header comment for WHY this exists (the env-
// route null dereference measured on the host) and
// lowfive_attach_decision.hpp's own header comment for the pure
// decision this file wraps with the real graph load, MPI rendezvous, and
// LowFive/HDF5 calls.
//
// ONE TRANSLATION UNIT, naming both a FUSE header that pulls in fmt 12
// (fuse/model/yaml_io.hpp, via fuse/base/result.hpp) AND LowFive's own
// headers (whose include directory historically carried a bundled,
// incompatible fmt 8.1.1) -- UNLIKE src/transport/lowfive_plugin_vol.cpp/
// lowfive_plugin_graph.cpp's own two-translation-unit split. That split
// is not necessary. The actual root cause was never "two fmt versions in
// one translation unit" -- it was `fuse_require(fmt ...)` having no
// minimum version, so with `/opt/lowfive` first on `CMAKE_PREFIX_PATH`,
// `find_package(fmt)` resolved LowFive's own bundled 8.1.1 project-wide.
// `fuse_require(fmt VERSION 9 ...)` (cmake/FuseDependencies.cmake) fixes
// that at the source: a fresh configure in EITHER CMAKE_PREFIX_PATH
// ordering builds every target clean, INCLUDING lowfive_transport.cpp
// itself, which also names both header families in one file. This file
// relies on that same floor, not on a second instance of the two-TU
// split -- `cmake/FuseDependencies.cmake`'s own comment calls the split
// "redundant... correctness rests on the VERSION 9 floor alone". This file
// DOES reuse the plugin's own graph-side building blocks
// (`fuse::build_lowfive_bindings()`, `fuse::plan_lowfive_rendezvous()`),
// just not lowfive_plugin_internal.hpp's own plain-data Config
// indirection, which exists ONLY to cross the (unnecessary) TU boundary
// and has nothing else to offer a caller that does not need that
// boundary.
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
// `plan.split_color`/`split_key` describe IS formed by a fresh
// MPI_Comm_split() in this file, by `bruss_lowfive_join_local_split()` --
// but of the BRIDGE (the base communicator fuse_setup() formed for this
// process's own Connect- or Mpmd-launched component, now readable through
// the base_communicator() ABI entry), never of
// MPI_COMM_WORLD directly, and never the caller's own already-formed
// physics-only communicator (u_main.c's own u_comm / v_main.f90's own
// v_comm) either -- both of those are wrong: splitting MPI_COMM_WORLD
// directly hangs when the bundled coupler/placeholder rank does not take
// part, and reusing u_comm/v_comm assumes, rather than derives, that
// their local rank numbering already matches plan.split_key's own
// expectation -- true for U (MPI_Intercomm_merge's own high/low rule
// keeps U's ranks unchanged) but NOT for V, whose merged position is
// offset by U's local world size (see
// WrongThisRankFromLocalWorldInsteadOfBridgeMisclassifiesV in
// test/verification/brusselator_lowfive_attach_test.cpp for the tier-0
// proof). The fresh split costs one extra collective per process, paid
// once at attach time, not per exchange. See bruss_lowfive_attach()'s own
// body below for how the resulting `local_comm_f` is then verified
// against the plan's own expected size/position exactly as before.
//
// DEADLOCK AVOIDANCE. `serve_on_close` (include/lowfive/
// vol-dist-metadata.hpp:33) defaults true, so `file_close()`
// (src/vol-dist-metadata.cpp:320-364) already calls `serve_all()` INSIDE
// H5Fclose -- a serve hook placed AFTER H5Fclose runs too late: both
// sides race into `serve_all()` together, neither having received first
// (a 120 s hang). So `serve_on_close = false` is set (below), and the
// flush of a first-reader's deferred serve happens at the event
// LowFive's own source ties the release to: AFTER this process's own
// H5Fclose of the peer's file (`kImportClosed`, a hook called from
// shim.cpp's own bruss_lowfive_read(), right after EVERY H5Fclose on a
// path where H5Fopen already succeeded -- see shim.cpp's own comment for
// why every such path, not only the success path). A process blocked
// inside `serve_all()` for file K is released only when the PEER
// completes its own OPEN-AND-CLOSE of file K (`DistMetadataVOL::
// file_close()`'s own `RemoteFile` branch calling `finish(0)`, src/
// vol-dist-metadata.cpp line ~342) -- not by the peer merely reaching a
// pre-open hook. Flushing in the pre-open hook (`receive_before_read()`,
// called right before H5Fopen) would run the first-reader's own blocking
// serve BEFORE it had done anything that could release the first-
// server's own pending one, and vice versa: circular wait. So
// `receive_before_read()` flushes nothing and does not call
// `broadcast_files()` either (lowfive_attach_decision.hpp's own "RECEIVE
// AND RELEASE" comment has the full citations, including why that call is
// not justified for this row's own one-rank-per-component layout). This
// necessarily serializes the two transfers (never both sides serving in
// the same window); the helper-thread follow-up that would let them
// overlap again is not done, because Index::serve()'s own barrier/
// ibarrier path and Query::get_filenames()'s own broadcast share one
// `local` communicator, and this build's HDF5 is Threadsafety:no with a
// plain MPI_Init.
//
// Checked against an explicit-state simulation of the blocking semantics
// above (lowfive_attach_model_check.hpp), not only against a
// word-of-mouth argument -- see that header's own top comment
// for exactly what it does and does not establish.
#include "lowfive_attach.h"

#include "fuse/base/config.hpp"

// These four need only fuse::model/
// fuse::transport and <mpi.h> -- NEVER a LowFive/HDF5 header -- so they
// are available regardless of FUSE_HAVE_LOWFIVE, unlike everything
// below. bruss_lowfive_join_local_split() (defined below, outside the
// FUSE_HAVE_LOWFIVE guard) is the one caller that needs them in a build
// without LowFive at all -- brusselator_coupler_main links this file
// without ever otherwise touching LowFive, so its own translation unit
// must still compile clean there.
#include "lowfive_attach_decision.hpp"

#include "fuse/model/yaml_io.hpp"
#include "fuse/transport/lowfive_graph.hpp"

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

#if FUSE_HAVE_LOWFIVE

#include <lowfive/H5VOLProperty.hpp>
#include <lowfive/log.hpp>
#include <lowfive/vol-metadata.hpp>
#if defined(FUSE_LOWFIVE_DIST_LIB_FOUND) && FUSE_HAVE_MPI
#include <lowfive/vol-dist-metadata.hpp>
#endif

#include <hdf5.h>

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
  // A base-class pointer valid in
  // EITHER branch of bruss_lowfive_attach() (LowFive::DistMetadataVOL
  // derives from LowFive::MetadataVOL, and MetadataVOL::files is what
  // bruss_lowfive_attach_file_count() reads -- lowfive_attach.h's own
  // doc comment on that function has the full citation) -- never
  // owned/deleted, same reasoning as VolSingleton's own note on
  // dist_vol below.
  LowFive::MetadataVOL* vol = nullptr;
#if FUSE_EXAMPLES_LOWFIVE_ATTACH_HAVE_DIST
  LowFive::DistMetadataVOL* dist_vol = nullptr;  // never owned/deleted; see VolSingleton's own note
  // The "role-ordered serve" pure state machine this
  // process's own write/read events drive -- meaningful only when
  // `distributed` is true (single-process mode never defers anything,
  // since LowFive::MetadataVOL has no serve_on_close/serve_all() of
  // this kind at all).
  fuse_examples::lowfive_attach::ServeState serve_state;
  // This component's own exported file name(s) -- lowfive_
  // attach_decision.hpp's own FindOwnExportedFiles(), computed once at
  // attach time from the SAME binding table/this_name the attach
  // already derives, never recomputed per write/read call. Dropped,
  // one at a time, right after each is actually served (this file's
  // own two call sites below) -- never the peer's own file, which this
  // rank never owns the lifetime of.
  std::vector<std::string> own_exported_files;
  // The per-process LOCAL order assertion's own tracker
  // (lowfive_attach_decision.hpp's own CheckAndAdvanceOrder()) --
  // checked, and advanced, before EVERY serve_state_step() call below,
  // never after (always before the next blocking call).
  fuse_examples::lowfive_attach::OrderTracker order_tracker;
  // A monotonically increasing per-process event
  // index (every real hook call below increments it once, in trace
  // order) -- distinct from order_tracker's own exchange_index/slot
  // (which resets every exchange; see lowfive_attach_decision.hpp's own
  // "EXCHANGE vs. PERIOD" comment); this one never resets, so two trace lines can be
  // compared for "did anything happen in between" even across
  // exchanges.
  long long event_index = 0;
  // Every communicator created is freed, on every rank, in the order
  // that cannot hang -- `local_comm` is the fresh split bruss_lowfive_join_local_
  // split() formed from the bridge (stashed there, before this rank ever
  // calls bruss_lowfive_attach() at all); `intercomm` is the one
  // MPI_Intercomm_create() forms inside bruss_lowfive_attach(). Both
  // MPI_COMM_NULL until actually created; bruss_lowfive_finalize() frees
  // whichever is non-null, AFTER H5close() and BEFORE the caller's own
  // MPI_Finalize (that function's own doc comment has the full account
  // of why that order, not MPI_Finalize's own implicit cleanup, is the
  // one that cannot hang).
  MPI_Comm local_comm = MPI_COMM_NULL;
  MPI_Comm intercomm = MPI_COMM_NULL;
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
    // `split.value().second` (straight from
    // split_hdf5_path(), e.g. "grid/u_out") has no leading slash, but
    // LowFive::Object::fullname() -- what MetadataVOL::dataset_create()'s
    // own is_memory check actually compares against -- always produces
    // one ("/grid/u_out"). Without normalizing, is_memory is false for
    // every Brusselator dataset this attach registers, and the dataset
    // is never attached as in-memory at all: see lowfive_attach_
    // decision.hpp's own "Slash normalisation" comment for the full citations.
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
    // The attach-time trace wants the EXACT (file, dataset) pairs passed to
    // set_memory() above -- collecting them here, in the one place
    // that already computes the normalized value, is the single
    // source of truth; re-deriving the same list a second time at the
    // trace call site would risk drifting out of sync with what this
    // function actually applied.
    if (applied_out) applied_out->emplace_back(split.value().first, normalized.value());
  }
}

// Cached-bool trace switch (the trace code stays out of the hot path when
// disabled) -- getenv() runs exactly once per process, the FIRST
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

// `LowFive::MetadataVOL::drop(std::string)` (src/vol-
// metadata.cpp, cited in lowfive_attach_decision.hpp's own "ROOT CAUSE"
// comment) dereferences `files.find(name)` WITHOUT checking `end()` --
// undefined behaviour on an absent name. This wrapper guards that;
// `files` is a PUBLIC member. Absent
// is treated as a harmless no-op, not a refusal: this protocol only ever
// calls this right after a file this SAME rank just
// served, so the name should always be present -- but a double-drop
// (a future bug, or a mutation of the call-site guard) finding the file
// already gone is exactly what we wanted anyway, not a new problem to
// report.
template <typename Vol>
void GuardedDropOwnExportedFile(Vol& vol, const std::string& file) {
  if (vol.files.find(file) != vol.files.end()) vol.drop(file);
}

// `FUSE_LOWFIVE_
// LOG=<level>` turns on LowFive's own spdlog-based tracing
// (`LowFive::create_logger(level)`, include/lowfive/log.hpp) -- with
// 'trace' LowFive prints 'enter serve_all, files.size = N' and Index
// logs (vol-dist-metadata.cpp:42, src/vol-dist/index.cpp:13 in the
// LowFive checkout). Read src/log.cpp first: `create_logger()` calls
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
// dependency, so it does not). On the caught-duplicate path
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

// The collective EVERY rank of the bridge
// must join, coupler/placeholder included -- lowfive_attach.h's own doc
// comment on this function has the full account of why, and of why it
// is safe and cheap to call unconditionally (an ordinary, non-LowFive
// graph reaches "nothing to join" identically on every rank, with no MPI
// call at all). Needs fuse::model (graph load, build_lowfive_bindings())
// and <mpi.h> only -- never a LowFive/HDF5 header, so this function
// compiles and is safe to link into a binary (brusselator_coupler_main)
// that never otherwise touches LowFive at all.
int bruss_lowfive_join_local_split(const char* graph_path, long long bridge_comm_f,
                                   long long* local_comm_f_out,
                                   char* errbuf, long long errbuf_len) {
  if (local_comm_f_out) *local_comm_f_out = 0;
  if (!graph_path || !*graph_path) {
    CopyErrorMessage("bruss_lowfive_join_local_split(): graph_path is empty", errbuf, errbuf_len);
    return 1;
  }
#if !FUSE_HAVE_MPI
  (void)bridge_comm_f;
  CopyErrorMessage(
      "bruss_lowfive_join_local_split(): this build has no MPI "
      "(-DFUSE_ENABLE_MPI=OFF); the Brusselator participants always need it",
      errbuf, errbuf_len);
  return 1;
#else
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (!mpi_initialized) {
    CopyErrorMessage(
        "bruss_lowfive_join_local_split(): MPI is not initialised; call MPI_Init() first",
        errbuf, errbuf_len);
    return 1;
  }
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
    // Collective agreement the easy way (connect.hpp's own module
    // comment): every rank loads the SAME graph file and reaches this
    // IDENTICAL verdict with no MPI call at all -- an ordinary,
    // non-LowFive run, or a build without -DFUSE_ENABLE_LOWFIVE=ON
    // (whose graph could never carry an hdf5_path field any caller of
    // this function would pass either). Nothing to join; not an error.
    return 0;
  }

  if (bridge_comm_f == 0) {
    CopyErrorMessage(
        "bruss_lowfive_join_local_split(): bridge_comm_f is 0 (MPI_COMM_NULL); "
        "pass base_communicator()'s own value for this context (entries.def), "
        "not an unset handle",
        errbuf, errbuf_len);
    return 1;
  }
  const MPI_Comm bridge = MPI_Comm_f2c(static_cast<MPI_Fint>(bridge_comm_f));
  if (bridge == MPI_COMM_NULL) {
    CopyErrorMessage(
        "bruss_lowfive_join_local_split(): bridge_comm_f converts to MPI_COMM_NULL",
        errbuf, errbuf_len);
    return 1;
  }
  int this_rank = 0;
  MPI_Comm_rank(bridge, &this_rank);

  const auto channels = fuse_examples::lowfive_attach::CollectChannelsForRank(graph, this_rank);
  const auto choice = fuse_examples::lowfive_attach::DecideLocalSplitChoice(channels, this_rank);

  // COLLECTIVE over the WHOLE of `bridge`, every rank, regardless of
  // `choice.participates` -- MPI_Comm_split() is collective over its own
  // INPUT communicator; MPI_UNDEFINED is the standard's own documented
  // mechanism for "I take part in this collective but belong to no
  // output group" (a non-participating rank gets back MPI_COMM_NULL,
  // which is NOT an error here -- see this function's own header
  // comment in lowfive_attach.h).
  MPI_Comm local = MPI_COMM_NULL;
  const int color = choice.participates ? choice.color : MPI_UNDEFINED;
  const int key = choice.participates ? choice.key : this_rank;
  MPI_Comm_split(bridge, color, key, &local);

  if (choice.participates && local == MPI_COMM_NULL) {
    CopyErrorMessage(
        "bruss_lowfive_join_local_split(): MPI_Comm_split() returned "
        "MPI_COMM_NULL for a rank DecideLocalSplitChoice() said should "
        "participate -- a color/key mismatch with the plan this function "
        "itself derived",
        errbuf, errbuf_len);
    return 1;
  }
  if (local_comm_f_out) {
    *local_comm_f_out = (local == MPI_COMM_NULL) ? 0 : static_cast<long long>(MPI_Comm_c2f(local));
  }
  if (TraceEnabledCached()) {
    std::fprintf(stderr,
                 "TRACE(bruss_lowfive_join_local_split): bridge_rank=%d participates=%d "
                 "color=%d key=%d\n",
                 this_rank, choice.participates ? 1 : 0, color, key);
  }
  // Every communicator created is freed -- `local` is stashed in
  // AttachState here,
  // for bruss_lowfive_attach() to hand to MPI_Intercomm_create() and for
  // bruss_lowfive_finalize() to free, in that order, AFTER this rank's
  // own H5close() and BEFORE the caller's own MPI_Finalize (that
  // function's own doc comment has the full account). A
  // non-participating rank's own `local` is MPI_COMM_NULL -- nothing to
  // stash, nothing to free.
#if FUSE_EXAMPLES_LOWFIVE_ATTACH_HAVE_DIST
  if (local != MPI_COMM_NULL) {
    State().local_comm = local;
  }
#endif
  return 0;
#endif  // FUSE_HAVE_MPI
}

int bruss_lowfive_attach(const char* graph_path, long long bridge_comm_f,
                         long long local_comm_f, long long* fapl_id_out,
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
  if (bridge_comm_f == 0) {
    CopyErrorMessage(
        "bruss_lowfive_attach(): bridge_comm_f is 0 (MPI_COMM_NULL); pass "
        "base_communicator()'s own value for this context, not an unset handle",
        errbuf, errbuf_len);
    return 1;
  }
  const MPI_Comm bridge = MPI_Comm_f2c(static_cast<MPI_Fint>(bridge_comm_f));
  if (bridge == MPI_COMM_NULL) {
    CopyErrorMessage("bruss_lowfive_attach(): bridge_comm_f converts to MPI_COMM_NULL", errbuf,
                     errbuf_len);
    return 1;
  }
  // This rank's own
  // position in the BRIDGE, never MPI_COMM_WORLD -- the two coincide for
  // Mpmd (connect_mpmd() merely dups MPI_COMM_WORLD) and for U in an
  // accept launch (MPI_Intercomm_merge's own high=0 side keeps its local
  // numbers), but NOT for V in an accept launch, whose own position is
  // OFFSET by U's local world size. Every RankSet-membership check below
  // (CollectChannelsForRank, decide_attach_mode, plan_lowfive_rendezvous)
  // is declared in the GRAPH's own bridge-relative numbering, so this is
  // the one rank number that makes all three agree, regardless of launch
  // mode.
  int this_rank = 0;
  MPI_Comm_rank(bridge, &this_rank);
  // Before any LowFive VOL is constructed, so its own ctor-time trace lines (if FUSE_LOWFIVE_LOG
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

  // The per-rank channel-collection loop is
  // CollectChannelsForRank() (lowfive_attach_decision.hpp), shared with
  // bruss_lowfive_join_local_split() above so the two calls -- which
  // MUST agree on this rank's own role, since join_local_split()'s own
  // split already baked a color into `local_comm_f` -- can never
  // silently diverge.
  const auto channels = fuse_examples::lowfive_attach::CollectChannelsForRank(graph, this_rank);

  // The STATIC order check, at attach time, before
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

  // `connect_mode_component_count` is this rank's
  // own, independent read of the SAME graph every rank already loaded --
  // a count, not a RankSet membership test, so it needs no per-rank
  // filtering of its own (every rank sees the identical graph-wide
  // total). `fuse_base_communicator_info::formed` is NOT re-derived here
  // -- the caller (u_main.c/v_main.f90) already checked it before ever
  // calling this function (never calling bruss_lowfive_attach() at all
  // when formed==0, the same gating that keeps the coupler/placeholder
  // out of this function entirely) -- so `bridge_formed=true` always,
  // from THIS call site; decide_attach_mode()'s own `false` branch is
  // reachable only from a direct, pure-function test, by design (a refusal is
  // verified only if a test makes it fire).
  int connect_mode_component_count = 0;
  for (std::uint32_t i = 0; i < graph.component_count(); ++i) {
    if (graph.component(fuse::ComponentId{i}).launch.mode == fuse::LaunchMode::Connect) {
      ++connect_mode_component_count;
    }
  }
  fuse_examples::lowfive_attach::AttachBridgeContext bridge_ctx;
  bridge_ctx.bridge_formed = true;
  bridge_ctx.connect_mode_component_count = connect_mode_component_count;

  auto decision = fuse_examples::lowfive_attach::decide_attach_mode(channels, this_rank, bridge_ctx);
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
    // `local` is the FRESH split bruss_lowfive_join_local_split()
    // already formed from the bridge (NEVER u_comm/v_comm reused -- see
    // lowfive_attach.h's own header comment on
    // bruss_lowfive_join_local_split() for why), keyed by RankSet
    // membership exactly as plan_lowfive_rendezvous() itself expects. The
    // intercommunicator is formed over the BRIDGE, never MPI_COMM_WORLD: in an
    // Mpmd launch the two are one and the same (connect_mpmd() merely
    // dups MPI_COMM_WORLD), but in an accept (Connect) launch they are
    // NOT -- MPI_COMM_WORLD spans only one of the two separately
    // launched executables, while the bridge spans both, which is
    // exactly what MPI_Intercomm_create()'s own `peer_comm` argument
    // must do to resolve `plan.remote_leader`'s own rank number
    // correctly.
    if (local_comm_f == 0) {
      CopyErrorMessage(
          "bruss_lowfive_attach(): local_comm_f is 0 (MPI_COMM_NULL); call "
          "bruss_lowfive_join_local_split() first and pass its own nonzero "
          "result, never an unset handle",
          errbuf, errbuf_len);
      return 1;
    }
    const MPI_Comm local = MPI_Comm_f2c(static_cast<MPI_Fint>(local_comm_f));
    if (local == MPI_COMM_NULL) {
      CopyErrorMessage(
          "bruss_lowfive_attach(): local_comm_f converts to MPI_COMM_NULL",
          errbuf, errbuf_len);
      return 1;
    }
    // Verify `local` actually has the shape the rendezvous plan assumed
    // (this rank's own side of the channel, same size and same relative
    // position `plan_lowfive_rendezvous()` computed from the graph's own
    // RankSet) -- "rather than trust it", the same discipline `fuse::
    // LowFiveTransport::build_plan_from_peers()`'s own header comment
    // states for an equivalent numbering assumption. Defence in depth
    // now, not the only check: bruss_lowfive_join_local_split()'s own
    // MPI_Comm_split() already used this exact color/key pair, so a
    // mismatch here would mean the TWO calls' own CollectChannelsForRank
    // results disagreed (a caller passing a stale local_comm_f from a
    // DIFFERENT graph, for instance), not merely an unverified handle.
    int local_size = 0, local_rank = 0;
    MPI_Comm_size(local, &local_size);
    MPI_Comm_rank(local, &local_rank);
    const fuse::RankSet& own_side =
        plan.is_writer ? channels.front().writer_ranks : channels.front().reader_ranks;
    if (static_cast<std::size_t>(local_size) != own_side.size() ||
        own_side.index_of(this_rank) != local_rank) {
      CopyErrorMessage(
          "bruss_lowfive_attach(): local_comm_f's own size/rank does not "
          "match this rank's own side of the graph's RankSet -- passed a "
          "stale or mismatched handle",
          errbuf, errbuf_len);
      return 1;
    }
    MPI_Comm intercomm = MPI_COMM_NULL;
    MPI_Intercomm_create(local, plan.local_leader, bridge, plan.remote_leader, plan.tag,
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
    // Stashed here for bruss_lowfive_
    // finalize() to free, AFTER H5close() and BEFORE MPI_Finalize (that
    // function's own doc comment has the full account of the order).
    s.intercomm = intercomm;
    LowFive::DistMetadataVOL& vol = LowFive::DistMetadataVOL::create_DistMetadataVOL(local, intercomm);
    // `serve_on_close`
    // is a PUBLIC bool member, default true (lowfive/vol-dist-
    // metadata.hpp:33 in /Users/mahadevan/Code/hdf5/LowFive, rev
    // 7f3a58b), and `DistMetadataVOL::file_close()` calls
    // `serve_all()` for every local in-memory file whenever it is true
    // (vol-dist-metadata.cpp:355-358) -- BEFORE this process's own code
    // ever runs a line, which makes a bruss_lowfive_serve_after_write()
    // placed AFTER H5Fclose a no-op and is the cause of a 120 s hang
    // (lowfive_attach_decision.hpp's own "WHY THE SERVE IS ROLE-ORDERED"
    // comment has the full stack). Setting it false here makes serving an
    // explicit decision this file's own serve_state_step() calls make,
    // never H5Fclose's.
    //
    // `serve_on_close=false` ALONE is not enough -- lowfive_
    // attach_decision.hpp's own "ROOT CAUSE" comment has the full
    // account: without `keep=true` too, `H5Fclose()`'s own tail call
    // into `MetadataVOL::file_close()` erases and deletes the file BEFORE
    // this process's own explicit serve_after_write() hook ever runs, so
    // that hook's own serve_all() finds files.size()==0 and returns
    // without serving (a hang).
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
    // Without this call, a CONSUMER's first remote H5Fopen on this process
    // throws file_open(): no intercomm found.
    // lowfive_attach_decision.hpp's own "set_intercomm()" and "Slash
    // normalisation" comments have the full citations for both this and
    // the normalization already applied inside ApplyBindings() above.
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
    // Role-ordered serve: which of this channel's own two
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
    // The file(s) this rank's own two serve call sites below
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
    // Attach-time trace: one line with bridge rank, component, role,
    // distributed flag, the registered (file, dataset_pattern,
    // intercomm_index) list, the (file, dataset) memory patterns, and the
    // serve_on_close and keep values actually set. `intercomms.size()`/
    // `files.size()` are directly diagnostic for a wrong/empty intercomm
    // selection and cost nothing extra to read (both are public members,
    // confirmed from the installed headers). `bridge_rank=` names this
    // rank's own position in the BRIDGE, not necessarily MPI_COMM_WORLD
    // (see this function's own `this_rank` comment above);
    // `intercomm_formed=1` confirms MPI_Intercomm_create() itself
    // completed before anything else in this branch could run (the first
    // MPI-level fact a host run should separate from everything
    // downstream of it).
    if (TraceEnabledCached()) {
      std::fprintf(stderr,
                   "TRACE(bruss_lowfive_attach): bridge_rank=%d component=%s role=%s "
                   "distributed=1 intercomm_formed=1 registrations=%s memory_patterns=%s "
                   "serve_on_close=%d keep=%d intercomms.size=%zu files.size=%zu\n",
                   this_rank, this_name.c_str(),
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
    // This branch needs exactly what decide_vol_settings(false).keep
    // names explicitly for the distributed branch (both branches' own
    // `keep` requirement traces to the SAME MetadataVOL::file_close()
    // behaviour this header's own "ROOT CAUSE" comment cites).
    // `serve_on_close` does not apply here at all: `LowFive::MetadataVOL`
    // (this branch's own VOL type, unlike the distributed branch's
    // DistMetadataVOL) has no such member, and nothing in this branch ever
    // calls serve_all()/drop() either -- this VOL never serves anyone, so
    // files simply accumulate in `files` for this process's own lifetime
    // (the single-process path is measured end to end).
    vol.set_keep(true);
    if (!BuildVerifiedFapl(vol, &fapl, &build_error)) {
      CopyErrorMessage(build_error, errbuf, errbuf_len);
      return 1;
    }
    distributed_now = false;
    s.vol = &vol;
    // The single-process branch has
    // no role/serve choreography at all (LowFive::MetadataVOL has no
    // serve_on_close/serve_all()), so this line is deliberately
    // shorter than the distributed branch's own -- there is nothing
    // else to report here that the distributed trace's own fields
    // would mean anything for.
    if (TraceEnabledCached()) {
      std::fprintf(stderr,
                   "TRACE(bruss_lowfive_attach): bridge_rank=%d distributed=0 "
                   "memory_patterns=%s files.size=%zu\n",
                   this_rank,
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
  // The LOCAL per-event order assertion, checked and
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
  // The policy decision and its own reason, BEFORE
  // calling serve_state_step() (which mutates serve_state.deferred_
  // pending but not completed_any_read for THIS event) -- decide_
  // serve_action()'s own condition, read directly, not re-derived.
  const bool cold_start_first_reader =
      s.serve_state.role == fuse_examples::lowfive_attach::ServeRole::kFirstReader &&
      !s.serve_state.completed_any_read;
  const long long this_event_index = s.event_index++;
  if (TraceEnabledCached()) {
    std::fprintf(stderr,
                 "TRACE(bruss_lowfive_serve_after_write): event_index=%lld exchange=%d policy=%s\n",
                 this_event_index, s.order_tracker.exchange_index,
                 cold_start_first_reader ? "cold-start-first-reader" : "steady-state");
  }
  // The WRITE side's own event/action pair (kExportWrittenAndClosed ->
  // kServeNow/kDeferServe); only the READ side's own flush placement is
  // subtle. See lowfive_attach_decision.hpp's own "RECEIVE AND RELEASE"
  // comment.
  auto action = fuse_examples::lowfive_attach::serve_state_step(
      &s.serve_state, fuse_examples::lowfive_attach::ServeEvent::kExportWrittenAndClosed);
  if (!action.ok()) {
    CopyErrorMessage("[" + action.error().code + "] " + action.error().message, errbuf,
                     errbuf_len);
    return 1;
  }
  // Trace: one line before and after each serve_all() call, with the
  // action taken and monotonic elapsed seconds. A serve_all() that returns
  // in microseconds served nothing; one that blocks until the peer opens
  // is the healthy case. The kDeferServe branch also traces (no
  // serve_all() call at all), so a hook that returned another action
  // shows up as a trace line with NO matching "exit elapsed=" line, as
  // distinct from a near-zero elapsed line.
  switch (action.value()) {
    case fuse_examples::lowfive_attach::ServeAction::kServeNow: {
      if (TraceEnabledCached())
        std::fprintf(stderr, "TRACE(bruss_lowfive_serve_after_write): action=kServeNow enter\n");
      const auto t0 = std::chrono::steady_clock::now();
      s.dist_vol->serve_all();
      // This component's own exported file(s) are only now safe to drop
      // -- SERVED, not merely written -- so the next exchange's own
      // H5Fcreate(TRUNC) of the same name does not meet a stale copy
      // (this file is recreated on every write, i.e. every exchange, not
      // once per coupling window; lowfive_attach_decision.hpp's own
      // "ROOT CAUSE" comment has the full citation on file_create()'s own
      // emplace()-on-duplicate-key behaviour).
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
  // This hook flushes nothing and does not call broadcast_files() either
  // -- lowfive_attach_decision.hpp's own "RECEIVE AND RELEASE" comment has
  // the full account, with LowFive-source citations, of why: the real
  // receive is H5Fopen itself (about to run, right after this call
  // returns), and broadcast_files() propagates an already-known file
  // registry to a multi-rank component's OWN sibling ranks over `local`,
  // never to the peer component over the intercomm -- not needed for this
  // row's own one-rank-per-component layout, and not reachable from any
  // code path inside DistMetadataVOL::file_open()/file_close() themselves
  // (grep confirms zero internal callers in the LowFive checkout this
  // cites). `serve_state_step()` is still called, purely so the event
  // sequence stays validated end to end (a future mutation that makes
  // kImportAboutToOpen return anything other than kReceiveOnly is caught
  // by this switch's own `default` case, and by
  // lowfive_attach_model_check.hpp's own search).
  //
  // Checked (NOT advanced -- CheckAndAdvanceOrder()'s own doc comment says
  // why kImportAboutToOpen only checks) BEFORE this process's own H5Fopen,
  // which runs right after this call returns -- "before the next blocking
  // call" for the READ side specifically.
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
                 "TRACE(bruss_lowfive_receive_before_read): event_index=%lld exchange=%d "
                 "policy=receive-only\n",
                 this_event_index, s.order_tracker.exchange_index);
  }
  auto action = fuse_examples::lowfive_attach::serve_state_step(
      &s.serve_state, fuse_examples::lowfive_attach::ServeEvent::kImportAboutToOpen);
  if (!action.ok()) {
    CopyErrorMessage("[" + action.error().code + "] " + action.error().message, errbuf,
                     errbuf_len);
    return 1;
  }
  // This hook never calls serve_all() at all (see the "RECEIVE AND
  // RELEASE" account above), so its own trace line is a single "enter/is
  // about to open" marker with no matching elapsed-time line -- a real
  // H5Fopen-blocks-here signature in the surrounding write/read trace
  // (shim.cpp) is what settles whether the peer is actually serving at
  // this point, not this hook.
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
          "(expected kReceiveOnly always; see lowfive_attach_decision.hpp's own RECEIVE AND "
          "RELEASE comment)",
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
  // This process has now completed its OWN open-and-close of the peer's
  // file, which -- read from LowFive's own source
  // (lowfive_attach_decision.hpp's own "RECEIVE AND RELEASE" comment) --
  // is exactly the event that would release a PEER's own pending
  // serve_all(); only now is it safe for a first-reader to enter its own
  // (the pre-open hook above deliberately flushes nothing).
  //
  // Checked and advanced BEFORE serve_state_step(), same discipline as
  // the write side above -- a deviation is refused before the
  // kFlushDeferred branch's own blocking serve_all() call.
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
                 "TRACE(bruss_lowfive_after_read_close): event_index=%lld exchange=%d "
                 "policy=%s\n",
                 this_event_index, s.order_tracker.exchange_index,
                 s.serve_state.deferred_pending ? "flush-deferred" : "noop-after-import");
  }
  auto action = fuse_examples::lowfive_attach::serve_state_step(
      &s.serve_state, fuse_examples::lowfive_attach::ServeEvent::kImportClosed);
  if (!action.ok()) {
    CopyErrorMessage("[" + action.error().code + "] " + action.error().message, errbuf,
                     errbuf_len);
    return 1;
  }
  // Trace: same before/after/
  // elapsed-time shape as serve_after_write()'s own kServeNow case
  // above -- this is the FIRST-READER's own (deferred) serve_all()
  // call, the other of the two places this process ever blocks in it.
  switch (action.value()) {
    case fuse_examples::lowfive_attach::ServeAction::kFlushDeferred: {
      if (TraceEnabledCached())
        std::fprintf(stderr, "TRACE(bruss_lowfive_after_read_close): action=kFlushDeferred enter\n");
      const auto t0 = std::chrono::steady_clock::now();
      s.dist_vol->serve_all();
      // Same reasoning as serve_after_write()'s own kServeNow
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
  if (!s.attached || !s.distributed) {
#if FUSE_EXAMPLES_LOWFIVE_ATTACH_HAVE_DIST
    // Free whatever THIS rank itself created even on this early-return
    // path -- a rank that
    // joined the split (bruss_lowfive_join_local_split()) but never
    // reached a successful bruss_lowfive_attach() (the only way
    // `s.attached` is false with `local_comm` non-null: u_main.c/
    // v_main.f90 abort the process immediately on any attach failure,
    // so this is a belt-and-suspenders case in practice, not a reachable
    // steady state) must not leak the communicator it already formed.
    if (s.local_comm != MPI_COMM_NULL) MPI_Comm_free(&s.local_comm);
    if (s.intercomm != MPI_COMM_NULL) MPI_Comm_free(&s.intercomm);
#endif
    return 0;
  }
#if FUSE_EXAMPLES_LOWFIVE_ATTACH_HAVE_DIST
  auto action = fuse_examples::lowfive_attach::serve_state_step(
      &s.serve_state, fuse_examples::lowfive_attach::ServeEvent::kFinalize);
  if (!action.ok()) {
    CopyErrorMessage("[" + action.error().code + "] " + action.error().message, errbuf,
                     errbuf_len);
    return 1;
  }
  // A DETERMINISTIC, explicit H5close() -- NOT an
  // atexit()-registered one -- called here, strictly BEFORE this
  // function's own caller's MPI_Finalize() (confirmed by reading both
  // u_main.c's and v_main.f90's own call sites: bruss_lowfive_
  // finalize() always precedes MPI_Finalize() there), while the
  // process is still in ORDINARY execution. decide_exit_hygiene_plan()
  // says an attaching participant (U or V, the only role this
  // function's own caller ever runs in) always wants this -- see
  // lowfive_attach_decision.hpp's own "EXIT HYGIENE" comment for why
  // this is sound
  // (spdlog's own static registry cannot have been torn down yet,
  // since no exit()-time unwind has begun) and why the PRE-EXISTING
  // std::atexit(+[]{H5close();}) in bruss_lowfive_attach() is KEPT,
  // not replaced, as a backstop for an abrupt-exit path that skips
  // this function entirely.
  const auto hygiene = fuse_examples::lowfive_attach::decide_exit_hygiene_plan(
      fuse_examples::lowfive_attach::LowFiveProcessRole::kAttachingParticipant);
  if (hygiene.explicit_h5close_before_mpi_finalize) {
    if (TraceEnabledCached()) {
      std::fprintf(stderr,
                   "TRACE(bruss_lowfive_finalize): calling H5close() before MPI_Finalize "
                   "(deterministic VOL teardown, role=kAttachingParticipant)\n");
    }
    H5close();
  }
  // Free every communicator this rank creates, in the order that cannot
  // hang -- AFTER H5close() (the VOL class is torn down; neither
  // communicator needs to stay alive for that call) and BEFORE the
  // caller's own MPI_Finalize (read, not assumed: both u_main.c's and
  // v_main.f90's own call sites call this function strictly before
  // MPI_Finalize(), the same ordering the H5close() above
  // already relies on). Intercomm first, local second -- the reverse of
  // creation order, the usual stack discipline, though MPI itself places
  // no ordering requirement between freeing two otherwise-unrelated
  // communicators.
  if (s.intercomm != MPI_COMM_NULL) MPI_Comm_free(&s.intercomm);
  if (s.local_comm != MPI_COMM_NULL) MPI_Comm_free(&s.local_comm);
  return 0;
#else
  (void)errbuf;
  (void)errbuf_len;
  return 0;
#endif
}

// Trace helpers: lowfive_attach.h's own doc comment
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

// See lowfive_attach.h's own doc comment on this function for why it
// exists (H5Lexists() is unreliable for this check) and why it is
// FILE-level, not dataset-level.
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

// UNLIKE every other stub in this branch, this one SUCCEEDS -- lowfive_attach.h's own doc comment on
// bruss_lowfive_join_local_split() states the contract plainly: without
// -DFUSE_ENABLE_LOWFIVE=ON, no graph this build could ever load carries
// an hdf5_path field, so every caller (physics rank or coupler/
// placeholder alike) reaches the IDENTICAL "nothing to join" answer the
// real implementation already gives an ordinary, non-LowFive graph, with
// no MPI call either way -- never an error, and never a reason for
// brusselator_coupler_main to behave differently in this configuration
// than it already does.
int bruss_lowfive_join_local_split(const char*, long long, long long* local_comm_f_out,
                                   char*, long long) {
  if (local_comm_f_out) *local_comm_f_out = 0;
  return 0;
}

int bruss_lowfive_attach(const char*, long long, long long, long long* fapl_id_out, char* errbuf,
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

// The three functions below do not
// depend on HDF5/LowFive at all (plain getenv()/fprintf()), so they
// work the same in this build configuration -- u_main.c's/v_main.f90's
// own skip-guard trace fires regardless of
// -DFUSE_ENABLE_LOWFIVE, same as bruss_lowfive_available() above.
// This duplicates (rather than shares) the env-parsing logic the
// FUSE_HAVE_LOWFIVE branch's own TraceEnabledCached() uses, because
// that branch's own pure half (ParseTraceEnabledEnv(),
// lowfive_attach_decision.hpp) is only included under `#if
// FUSE_HAVE_LOWFIVE` above (it pulls in fuse/base/result.hpp and
// friends this translation unit has no other reason to need here) --
// restructuring that include guard just to share one string
// comparison is not worth the risk.
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

// Same reasoning as bruss_lowfive_attach_file_count() above.
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
