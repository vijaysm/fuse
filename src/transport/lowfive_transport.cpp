// Task 8.4: the LowFive HDF5-VOL transport (spec §5 mechanism 4, Demo B).
//
// THIS FILE IS COMPILED IN EVERY CONFIGURATION. Only its body is conditional.
// `LowFiveTransport` is declared unconditionally in the public header, so
// every one of its members must link everywhere; on a build without LowFive
// they return E-TRANSPORT-NOLOWFIVE and name the CMake option. Task 5.2's
// failure was the other arrangement -- an always-visible header declaring a
// function that only exists in some configurations.
//
// THREE THINGS IN HERE ARE NOT OBVIOUS, AND EACH ONE WAS MEASURED.
//
// 1. The connector is attached programmatically, never through the
//    HDF5_VOL_CONNECTOR environment variable. The environment-variable path
//    aborts the process after main returns (docs/user/lowfive.md has the trace (this repo, unlike upstream/soma, keeps no docs/BUILDING.md):
//    HDF5's atexit handler terminates the connector after LowFive's spdlog
//    statics are already destroyed).
//
// 2. There is exactly ONE registration per process and it is never undone.
//    `LowFive::MetadataVOL::create_MetadataVOL()` returns a singleton, and
//    `VOLBase::register_plugin()` is not reference counted: two live
//    `H5VOLProperty` objects get the SAME vol_id, and whichever is destroyed
//    first unregisters it out from under the other, which then fails with
//    "not a VOL connector ID". Verified directly. So the property object is
//    created once, on demand, and intentionally leaked -- the same reasoning
//    that leaks `TransportRegistry::instance()`.
//
// 3. Interception is PROVED, not assumed. A VOL that failed to attach does
//    not report an error: HDF5 simply writes a real file, everything returns
//    0, and the coupling silently becomes slow disk I/O. Two independent
//    checks run at setup and again on every move: LowFive's
//    `after_dataset_write` callback must have fired (that is connector code
//    executing, which cannot happen through the native VOL), and `stat()` on
//    the path must fail. `build_plan` additionally proves the directory IS
//    writable first, because "no file on disk" is worthless evidence if a
//    file could not have been created there anyway.

#include "fuse/base/config.hpp"
#include "fuse/transport/transport.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#if FUSE_HAVE_LOWFIVE
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <hdf5.h>
#include <lowfive/H5VOLProperty.hpp>
#include <lowfive/vol-metadata.hpp>
#endif

// Distributed mode (Demo B, two executables -- see set_intercomm()'s
// declaration in fuse/transport/transport.hpp for what this is and the
// UNEXECUTED caveat that applies to all of it). Needs MPI for the
// intercommunicator itself, which is why this is a second, narrower guard
// than plain FUSE_HAVE_LOWFIVE: a LowFive build with FUSE_ENABLE_MPI=OFF has
// the single-process path above but not this one.
//
// ALSO needs FUSE_LOWFIVE_DIST_LIB_FOUND (cmake/FuseDependencies.cmake's
// LowFive block, via cmake/FindLowFive.cmake's LOWFIVE_HAVE_DIST):
// LowFive::DistMetadataVOL is compiled into a SEPARATE static archive
// (liblowfive-dist.a) on both installs measured on this lab's host, not
// into liblowfive itself -- linking against vol-dist-metadata.hpp's
// declarations without that archive fails at the FINAL link of whatever
// links this translation unit, not here, which is exactly what happened to
// fuse_transport_lowfive_test (2026-10-02) before this second guard
// existed. A build with MPI but no discovered dist archive therefore still
// gets the single-process path only, same as a build with no MPI at all.
#if FUSE_HAVE_LOWFIVE && FUSE_HAVE_MPI && defined(FUSE_LOWFIVE_DIST_LIB_FOUND)
#include <mpi.h>
#include <lowfive/vol-dist-metadata.hpp>
#define FUSE_LOWFIVE_HAVE_DIST 1
#else
#define FUSE_LOWFIVE_HAVE_DIST 0
#endif

namespace fuse {
namespace {

#if !FUSE_HAVE_LOWFIVE
[[nodiscard]] Failure no_lowfive(const char* what) {
  return Failure{"E-TRANSPORT-NOLOWFIVE",
               std::string(what) +
                   " requires a LowFive-enabled build; this one was configured "
                   "without it. Reconfigure with -DFUSE_ENABLE_LOWFIVE=ON and "
                   "-DFUSE_ENABLE_MOAB=ON, -DCMAKE_PREFIX_PATH pointed at a "
                   "LowFive built against the SAME HDF5 prefix MOAB uses -- "
                   // R-NN (this track, 2026-10-02): the previous guidance here
                   // named "/opt/lowfive-shared;/opt/hdf5-shared", which is
                   // EXACTLY the pairing tools/probe-hdf5-images.sh measures
                   // as hazardous on this lab's machine (two distinct libhdf5
                   // images load when it is linked alongside MOAB; see
                   // docs/user/lowfive.md and KNOWN_ISSUES.md section D).
                   // cmake/FuseDependencies.cmake now refuses that pairing at
                   // configure time rather than let it reach here, but a
                   // string baked into this message is not refused by
                   // anything, so it must not repeat the mistake.
                   "cmake/FuseDependencies.cmake's configure-time probe will "
                   "refuse any pairing that is not (see docs/user/lowfive.md)"};
}
#endif

#if FUSE_HAVE_LOWFIVE

[[nodiscard]] bool path_exists(const char* p) noexcept {
  struct ::stat s {};
  return ::stat(p, &s) == 0;
}

/// The process-wide VOL registration. Constructed on first use and never
/// destroyed; see note 2 at the top of this file for why there can only be
/// one, and why unregistering it is not worth the hazard.
struct VolSingleton {
  LowFive::MetadataVOL&    vol;
  LowFive::H5VOLProperty*  prop = nullptr;  // intentionally leaked with `this`
  hid_t                    vol_id = H5I_INVALID_HID;
  /// Bumped by LowFive itself, from inside `MetadataVOL::dataset_write`.
  /// This counter advancing is the evidence that the connector is in the
  /// stack; nothing in the native HDF5 path can move it.
  std::size_t              writes = 0;
  bool                     registered = false;
  /// File names currently claimed by a live transport. The VOL's in-memory
  /// file map is shared, so two transports on one name would overwrite each
  /// other's datasets with no diagnostic at all.
  std::vector<std::string> claimed;

  VolSingleton() : vol(LowFive::MetadataVOL::create_MetadataVOL()) {
    prop = new LowFive::H5VOLProperty(vol);
    vol_id = prop->vol_id;
    // "*","*" = every file, every dataset: this transport owns the whole
    // namespace it uses, and a pattern that failed to match would fall
    // through to real file I/O, which is the outcome this file exists to
    // prevent.
    vol.set_memory("*", "*");
    // Without keep, LowFive discards a file's metadata at H5Fclose and the
    // read leg's H5Fopen finds nothing.
    vol.set_keep(true);
    vol.set_after_dataset_write([this] { ++writes; });
    registered = H5VLis_connector_registered_by_name("lowfive") == 1 &&
                 vol_id >= 0;

    // ABORT AT EXIT, AND WHY THIS LINE IS HERE.
    //
    // docs/user/lowfive.md records the process dying AFTER main returns with
    //   libc++abi: terminating due to uncaught exception of type
    //   std::__1::system_error: mutex lock failed: Invalid argument
    // and attributes it to the HDF5_VOL_CONNECTOR plugin-loading path. It is
    // not confined to that path. The programmatic path reproduces it too --
    // observed here on exactly one test, the one that makes a NATIVE-VOL HDF5
    // call before registering the connector, and reproduced standalone in ~30
    // lines with the same ordering. All output is produced first; the abort is
    // in teardown, when HDF5's atexit handler terminates the connector after
    // LowFive's spdlog statics and their mutex are already gone.
    //
    // The remedy BUILDING.md already documents is to call H5close() while
    // those statics are still alive. Registering it with std::atexit HERE puts
    // it after HDF5's own registration in the LIFO chain, so it runs first.
    // Verified: the standalone reproducer goes exit=134 -> exit=0, and the
    // test binary goes "Subprocess aborted" -> pass.
    //
    // Terminating HDF5 on the application's behalf is a heavy thing for a
    // library to do; it is scoped to a process that has actually registered
    // the LowFive connector, and it runs at exit, after which no HDF5 call
    // can follow.
    std::atexit(+[] { H5close(); });
  }

  VolSingleton(const VolSingleton&) = delete;
  VolSingleton& operator=(const VolSingleton&) = delete;
};

VolSingleton& vol_singleton() {
  static VolSingleton* g = new VolSingleton;  // leaked, deliberately
  return *g;
}

#if FUSE_LOWFIVE_HAVE_DIST
/// The process-wide DISTRIBUTED VOL registration -- set_intercomm()'s
/// counterpart to VolSingleton above, and mutually exclusive with it: a
/// process attaches ONE LowFive connector (see the note on VolSingleton),
/// so whichever of `vol_singleton()` / `dist_vol_singleton()` is reached
/// FIRST decides which kind, and the other must never be reached afterward
/// in the same process. `set_intercomm()` enforces this by being refused
/// once `set_location()`/`build_plan()` have already created a
/// (non-distributed) VolSingleton.
///
/// UNEXECUTED: constructed from LowFive::DistMetadataVOL's public header
/// contract (lowfive/vol-dist-metadata.hpp) only; MPI cannot start in the
/// sandbox this was written in, so nothing below has run. See
/// docs/user/lowfive.md and this track's report for exactly what that means
/// for confidence in this path versus the single-process one above, which
/// IS measured end to end.
struct DistVolSingleton {
  LowFive::DistMetadataVOL& vol;
  LowFive::H5VOLProperty*   prop = nullptr;  // intentionally leaked with `this`
  hid_t                     vol_id = H5I_INVALID_HID;
  bool                      registered = false;
  std::vector<std::string>  claimed;

  DistVolSingleton(MPI_Comm local, MPI_Comm intercomm)
      : vol(LowFive::DistMetadataVOL::create_DistMetadataVOL(local, intercomm)) {
    prop = new LowFive::H5VOLProperty(vol);
    vol_id = prop->vol_id;
    // Same reasoning as VolSingleton::VolSingleton: this process's own HDF5
    // traffic for the claimed file(s) is served from memory/over the
    // intercomm, nothing is kept on close (serve_all() below does the
    // handoff explicitly instead), and the same post-main teardown hazard
    // documented on VolSingleton applies here too.
    vol.set_memory("*", "*");
    registered = H5VLis_connector_registered_by_name("lowfive") == 1 &&
                 vol_id >= 0;
    std::atexit(+[] { H5close(); });
  }

  DistVolSingleton(const DistVolSingleton&) = delete;
  DistVolSingleton& operator=(const DistVolSingleton&) = delete;
};

/// Unlike `vol_singleton()`, this takes the communicators on its FIRST call
/// (construction time) and ignores them on every later call -- there is only
/// ever one. A second `set_intercomm()` with different handles in the same
/// process is refused by comparing against `first_local`/`first_intercomm`
/// before this is reached a second time; see `LowFiveTransport::set_intercomm`.
DistVolSingleton& dist_vol_singleton(MPI_Comm local, MPI_Comm intercomm) {
  static DistVolSingleton* g = new DistVolSingleton(local, intercomm);  // leaked
  return *g;
}
#endif  // FUSE_LOWFIVE_HAVE_DIST

/// UPSTREAM BUG, worked around here. `LowFive::MetadataVOL::drop(std::string)`
/// does not check `files.find(name)` against `files.end()`; called with a name
/// that is not in the in-memory map it dereferences the end iterator and the
/// process dies with EXC_BAD_ACCESS inside libc++'s red-black tree code.
/// Reproduced standalone against /opt/lowfive-shared with a two-line program:
/// create the VOL, call `drop("never-existed.h5")`, exit 139. It bites here
/// because the file is dropped after every `move()` and again in the
/// destructor. `MetadataVOL::files` is a public member, so the membership test
/// the library should be doing is available to us.
void safe_drop(LowFive::MetadataVOL& vol, const std::string& file) {
  if (vol.files.find(file) != vol.files.end()) vol.drop(file);
}

/// RAII for the ids `move` and the setup probe open, so an early return on an
/// HDF5 failure cannot leak an hid_t. HDF5 ids are a process-global resource;
/// leaking one per step is a slow-motion resource exhaustion that no test
/// notices.
class DsetHandle {
 public:
  explicit DsetHandle(hid_t id) noexcept : id_(id) {}
  ~DsetHandle() { if (id_ >= 0) H5Dclose(id_); }
  DsetHandle(const DsetHandle&) = delete;
  DsetHandle& operator=(const DsetHandle&) = delete;
  [[nodiscard]] hid_t get() const noexcept { return id_; }
  [[nodiscard]] bool ok() const noexcept { return id_ >= 0; }

 private:
  hid_t id_;
};

class FileHandle {
 public:
  explicit FileHandle(hid_t id) noexcept : id_(id) {}
  ~FileHandle() { if (id_ >= 0) H5Fclose(id_); }
  FileHandle(const FileHandle&) = delete;
  FileHandle& operator=(const FileHandle&) = delete;
  [[nodiscard]] hid_t get() const noexcept { return id_; }
  [[nodiscard]] bool ok() const noexcept { return id_ >= 0; }

 private:
  hid_t id_;
};

#endif  // FUSE_HAVE_LOWFIVE

}  // namespace

// ---------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------

struct LowFiveTransport::Impl {
  std::string file;
  std::string dataset;
  bool        located = false;
  bool        planned = false;

  // The planned spec's identity. prepare() compares against these and nothing
  // else; every comparison is a scalar, so it allocates nothing.
  std::size_t p_elements = 0;
  int         p_components = 1;
  int         p_src_rank_set = -1;
  int         p_dst_rank_set = -1;
  std::size_t p_n_ids = 0;
  std::size_t p_n_src_ranks = 0;
  std::size_t p_n_dst_ranks = 0;
  std::size_t p_total = 0;  // elements * components: the `n` move() demands

  std::size_t writes_seen = 0;

  // Task 8.4 continuation (Demo B): set once by set_intercomm(), before
  // set_location(). Decides which VOL singleton build_plan() attaches to --
  // see the note on DistVolSingleton for why there is only ever one kind
  // per process. UNEXECUTED path; see transport.hpp's set_intercomm() doc.
  bool distributed = false;
#if FUSE_LOWFIVE_HAVE_DIST
  MPI_Comm    local_comm = MPI_COMM_NULL;
  MPI_Comm    intercomm = MPI_COMM_NULL;
#endif

#if FUSE_HAVE_LOWFIVE
  hid_t       fapl = H5I_INVALID_HID;
  hid_t       space = H5I_INVALID_HID;
  // Cached so move() does not touch std::string. `file` and `dataset` are
  // never mutated after set_location, and Impl lives behind a unique_ptr, so
  // moving the transport does not move the strings.
  const char* file_c = nullptr;
  const char* dataset_c = nullptr;

  ~Impl() {
    if (space >= 0) H5Sclose(space);
    if (fapl >= 0) H5Pclose(fapl);
    if (located) {
#if FUSE_LOWFIVE_HAVE_DIST
      if (distributed) {
        // dist_vol_singleton() is only ever constructed via set_intercomm(),
        // which is the only path that can have set `distributed`; by the
        // time a located, distributed Impl is destroyed the singleton
        // already exists, so this does not risk creating one with the wrong
        // communicators as a destructor side effect.
        auto& g = dist_vol_singleton(local_comm, intercomm);
        auto it = std::find(g.claimed.begin(), g.claimed.end(), file);
        if (it != g.claimed.end()) g.claimed.erase(it);
        return;
      }
#endif
      auto& g = vol_singleton();
      safe_drop(g.vol, file);
      auto it = std::find(g.claimed.begin(), g.claimed.end(), file);
      if (it != g.claimed.end()) g.claimed.erase(it);
    }
  }
#else
  ~Impl() = default;
#endif

  Impl() = default;
  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;
};

LowFiveTransport::LowFiveTransport() : impl_(std::make_unique<Impl>()) {}
LowFiveTransport::~LowFiveTransport() = default;
LowFiveTransport::LowFiveTransport(LowFiveTransport&&) noexcept = default;
LowFiveTransport& LowFiveTransport::operator=(LowFiveTransport&&) noexcept =
    default;

bool LowFiveTransport::is_available() noexcept {
#if FUSE_HAVE_LOWFIVE
  return true;
#else
  return false;
#endif
}

long long LowFiveTransport::access_plist_handle() const noexcept {
#if FUSE_HAVE_LOWFIVE
  return impl_ ? static_cast<long long>(impl_->fapl) : -1;
#else
  return -1;
#endif
}

std::size_t LowFiveTransport::intercepted_writes() const noexcept {
  return impl_ ? impl_->writes_seen : 0;
}

std::string LowFiveTransport::file_path() const {
  return impl_ ? impl_->file : std::string{};
}

// ---------------------------------------------------------------------------
// set_intercomm / serve_as_producer / receive_as_consumer -- distributed
// mode, Demo B (Task 8.4 continuation). UNEXECUTED: see the doc comment on
// set_intercomm() in fuse/transport/transport.hpp.
// ---------------------------------------------------------------------------

Result<void> LowFiveTransport::set_intercomm(int fortran_local_comm,
                                             int fortran_intercomm) {
#if !FUSE_HAVE_LOWFIVE
  (void)fortran_local_comm;
  (void)fortran_intercomm;
  return no_lowfive("LowFiveTransport::set_intercomm()");
#elif !FUSE_LOWFIVE_HAVE_DIST
  (void)fortran_local_comm;
  (void)fortran_intercomm;
  return Failure{"E-TRANSPORT-NOMPI",
               "LowFiveTransport::set_intercomm() needs a build configured "
               "with both -DFUSE_ENABLE_LOWFIVE=ON and -DFUSE_ENABLE_MPI=ON; "
               "this one has LowFive but not MPI, so only the single-process "
               "path (set_location/build_plan/move) is available"};
#else
  Impl* p = impl_.get();
  if (p == nullptr) {
    return Failure{"E-TRANSPORT-LF-LOCATION",
                 "set_intercomm() on a moved-from LowFiveTransport"};
  }
  if (p->located) {
    return Failure{"E-TRANSPORT-LF-LOCATION",
                 "set_intercomm() after set_location()/build_plan(); it "
                 "decides which VOL connector this process attaches and "
                 "cannot switch once the single-process one is already live"};
  }
  int initialized = 0;
  if (MPI_Initialized(&initialized) != MPI_SUCCESS || initialized == 0) {
    return Failure{"E-TRANSPORT-MPIINIT",
                 "MPI is not initialised; MPI_Comm_f2c on a handle from an "
                 "uninitialised MPI is not meaningful"};
  }
  int finalized = 0;
  if (MPI_Finalized(&finalized) == MPI_SUCCESS && finalized != 0) {
    return Failure{"E-TRANSPORT-MPIINIT", "MPI has already been finalised"};
  }
  // Refuse a null handle BEFORE any query on it: MPICH's default error
  // handler aborts the process inside MPI_Comm_test_inter on an invalid
  // handle instead of returning (seen on the host 2026-10-04, handle 0).
  if (fortran_local_comm == 0 || fortran_intercomm == 0 ||
      MPI_Comm_f2c(static_cast<MPI_Fint>(fortran_local_comm)) == MPI_COMM_NULL ||
      MPI_Comm_f2c(static_cast<MPI_Fint>(fortran_intercomm)) == MPI_COMM_NULL) {
    return Failure{"E-TRANSPORT-COMM",
                 "set_intercomm() needs two live communicators; a handle of 0 "
                 "or MPI_COMM_NULL is not one"};
  }
  const MPI_Comm local = MPI_Comm_f2c(static_cast<MPI_Fint>(fortran_local_comm));
  const MPI_Comm inter = MPI_Comm_f2c(static_cast<MPI_Fint>(fortran_intercomm));
  if (local == MPI_COMM_NULL) {
    return Failure{"E-TRANSPORT-COMM",
                 "local communicator handle " +
                     std::to_string(fortran_local_comm) +
                     " converts to MPI_COMM_NULL"};
  }
  if (inter == MPI_COMM_NULL) {
    return Failure{"E-TRANSPORT-COMM",
                 "intercommunicator handle " +
                     std::to_string(fortran_intercomm) +
                     " converts to MPI_COMM_NULL"};
  }
  int is_inter = 0;
  MPI_Comm_test_inter(inter, &is_inter);
  if (!is_inter) {
    return Failure{"E-TRANSPORT-COMM",
                 "the handle given as the intercommunicator is an ordinary "
                 "(intra-)communicator; form one with MPI_Comm_connect / "
                 "MPI_Comm_accept or an MPMD launch's split, not with "
                 "MPI_Comm_dup/MPI_Comm_split"};
  }
  p->distributed = true;
  p->local_comm = local;
  p->intercomm = inter;
  return Result<void>::success();
#endif
}

Result<void> LowFiveTransport::serve_as_producer() {
#if !FUSE_HAVE_LOWFIVE
  return no_lowfive("LowFiveTransport::serve_as_producer()");
#elif !FUSE_LOWFIVE_HAVE_DIST
  return Failure{"E-TRANSPORT-NOMPI",
               "serve_as_producer() needs -DFUSE_ENABLE_MPI=ON alongside "
               "-DFUSE_ENABLE_LOWFIVE=ON"};
#else
  Impl* p = impl_.get();
  if (p == nullptr || !p->distributed) {
    return Failure{"E-TRANSPORT-LF-NOTDIST",
                 "serve_as_producer() without a prior successful "
                 "set_intercomm(); there is no intercommunicator to serve "
                 "across"};
  }
  // Public API shape only (vol-dist-metadata.hpp): push whatever this
  // process has written across the intercommunicator. UNEXECUTED -- see
  // this method's declaration.
  dist_vol_singleton(p->local_comm, p->intercomm).vol.serve_all();
  return Result<void>::success();
#endif
}

Result<void> LowFiveTransport::receive_as_consumer() {
#if !FUSE_HAVE_LOWFIVE
  return no_lowfive("LowFiveTransport::receive_as_consumer()");
#elif !FUSE_LOWFIVE_HAVE_DIST
  return Failure{"E-TRANSPORT-NOMPI",
               "receive_as_consumer() needs -DFUSE_ENABLE_MPI=ON alongside "
               "-DFUSE_ENABLE_LOWFIVE=ON"};
#else
  Impl* p = impl_.get();
  if (p == nullptr || !p->distributed) {
    return Failure{"E-TRANSPORT-LF-NOTDIST",
                 "receive_as_consumer() without a prior successful "
                 "set_intercomm()"};
  }
  // UNEXECUTED -- see this method's declaration in transport.hpp.
  dist_vol_singleton(p->local_comm, p->intercomm).vol.broadcast_files();
  return Result<void>::success();
#endif
}

// ---------------------------------------------------------------------------
// set_location
// ---------------------------------------------------------------------------

Result<void> LowFiveTransport::set_location(std::string file,
                                            std::string dataset) {
#if !FUSE_HAVE_LOWFIVE
  (void)file;
  (void)dataset;
  return no_lowfive("LowFiveTransport::set_location()");
#else
  Impl* p = impl_.get();
  if (p == nullptr) {
    return Failure{"E-TRANSPORT-LF-LOCATION",
                 "set_location() on a moved-from LowFiveTransport"};
  }
  if (p->planned) {
    return Failure{"E-TRANSPORT-LF-LOCATION",
                 "set_location() after build_plan(); the property list and "
                 "the interception proof are built from the location and "
                 "cannot be rebuilt in the loop"};
  }
  if (file.empty() || dataset.empty()) {
    return Failure{"E-TRANSPORT-LF-LOCATION",
                 "both a file name and a dataset name are required"};
  }

  // Distributed mode (set_intercomm() already called) tracks claimed file
  // names on the DIST singleton, never the non-distributed one -- the two
  // have independent in-memory file maps. Non-distributed is the common
  // case, so its lookup is not behind the FUSE_LOWFIVE_HAVE_DIST guard.
  std::vector<std::string>* claimed = nullptr;
#if FUSE_LOWFIVE_HAVE_DIST
  if (p->distributed) {
    claimed = &dist_vol_singleton(p->local_comm, p->intercomm).claimed;
  }
#endif
  VolSingleton* non_dist = nullptr;
  if (claimed == nullptr) {
    non_dist = &vol_singleton();
    claimed = &non_dist->claimed;
  }
  if (std::find(claimed->begin(), claimed->end(), file) != claimed->end()) {
    return Failure{"E-TRANSPORT-LF-LOCATION",
                 "another live LowFiveTransport already uses the file name '" +
                     file +
                     "'; LowFive's in-memory file map is process-wide, so the "
                     "two would silently overwrite each other"};
  }

  // "No file on disk" is only evidence if a file COULD have appeared there.
  // Prove the directory is writable by actually creating and removing a
  // sentinel next to the path we are about to claim. Meaningful in
  // distributed mode too: the producer side still promises nothing reaches
  // this path, and "could not have written here anyway" would be worthless
  // evidence of that promise in either mode.
  const std::string sentinel = file + ".soma-writable-probe";
  const int fd = ::open(sentinel.c_str(), O_CREAT | O_EXCL | O_WRONLY, 0600);
  if (fd < 0) {
    return Failure{"E-TRANSPORT-LF-LOCATION",
                 "cannot create a file next to '" + file +
                     "', so the absence of that file would prove nothing "
                     "about interception. Point the transport at a writable "
                     "directory"};
  }
  ::close(fd);
  ::unlink(sentinel.c_str());

  if (p->located) {
    auto it = std::find(claimed->begin(), claimed->end(), p->file);
    if (it != claimed->end()) claimed->erase(it);
  }
  p->file = std::move(file);
  p->dataset = std::move(dataset);
  p->file_c = p->file.c_str();
  p->dataset_c = p->dataset.c_str();
  p->located = true;
  claimed->push_back(p->file);
  return Result<void>::success();
#endif
}

// ---------------------------------------------------------------------------
// build_plan
// ---------------------------------------------------------------------------

Result<void> LowFiveTransport::build_plan(const TransportSpec& spec) {
#if !FUSE_HAVE_LOWFIVE
  (void)spec;
  return no_lowfive("LowFiveTransport::build_plan()");
#else
  Impl* p = impl_.get();
  if (p == nullptr) {
    return Failure{"E-TRANSPORT-LF-LOCATION",
                 "build_plan() on a moved-from LowFiveTransport"};
  }
  if (!p->located) {
    return Failure{"E-TRANSPORT-LF-LOCATION",
                 "build_plan() before set_location()"};
  }
  // A retry after a failed build_plan must not leak the ids the failed
  // attempt created.
  if (p->space >= 0) {
    H5Sclose(p->space);
    p->space = H5I_INVALID_HID;
  }
  if (p->fapl >= 0) {
    H5Pclose(p->fapl);
    p->fapl = H5I_INVALID_HID;
  }
  p->planned = false;
  if (spec.components < 1) {
    return Failure{"E-TRANSPORT-SIZE",
                 "components must be at least 1, got " +
                     std::to_string(spec.components)};
  }
  const auto nc = static_cast<std::size_t>(spec.components);
  if (spec.elements == 0) {
    return Failure{"E-TRANSPORT-SIZE",
                 "a LowFive channel with zero elements has no I/O to "
                 "intercept; there is nothing for this backend to do"};
  }
  if (spec.elements > (static_cast<std::size_t>(-1) / nc)) {
    return Failure{"E-TRANSPORT-SIZE",
                 "elements * components overflows std::size_t"};
  }
  const std::size_t total = spec.elements * nc;

#if FUSE_LOWFIVE_HAVE_DIST
  if (p->distributed) {
    // ---------------------------------------------------------------------
    // DISTRIBUTED build_plan. Cannot run the single-process round-trip probe
    // below: the read leg happens in the PEER process, after that process's
    // own receive_as_consumer()/H5Fopen, which has not happened yet (may
    // never happen on this rank at all -- a producer-only process has no
    // read leg ever). What IS provable here, without any I/O: the connector
    // registered under its OWN name check, and a property list this process
    // creates actually carries it (the same two checks the non-distributed
    // path makes before touching HDF5 data, just without the write+read
    // either side of them). The bit-for-bit, zero-bytes-on-disk claim is
    // Demo B's own test's job (examples/demo_b), run on the host.
    // UNEXECUTED -- see set_intercomm()'s doc in transport.hpp.
    // ---------------------------------------------------------------------
    auto& dg = dist_vol_singleton(p->local_comm, p->intercomm);
    if (!dg.registered) {
      return Failure{"E-TRANSPORT-LF-VOL",
                   "the distributed LowFive VOL connector did not register; "
                   "H5VLis_connector_registered_by_name(\"lowfive\") is not 1"};
    }
    const hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    if (fapl < 0) {
      return Failure{"E-TRANSPORT-HDF5", "H5Pcreate(H5P_FILE_ACCESS) failed"};
    }
    dg.prop->apply(fapl);
    hid_t      on_list = H5I_INVALID_HID;
    const bool got = H5Pget_vol_id(fapl, &on_list) >= 0;
    const bool matches = got && on_list == dg.vol_id;
    if (on_list >= 0) H5VLclose(on_list);
    if (!matches) {
      H5Pclose(fapl);
      return Failure{"E-TRANSPORT-LF-VOL",
                   "H5Pset_vol did not put the distributed LowFive connector "
                   "on the file-access property list"};
    }
    const auto dim = static_cast<hsize_t>(total);
    const hid_t space = H5Screate_simple(1, &dim, nullptr);
    if (space < 0) {
      H5Pclose(fapl);
      return Failure{"E-TRANSPORT-HDF5",
                   "H5Screate_simple for " + std::to_string(total) +
                       " elements failed"};
    }
    p->fapl = fapl;
    p->space = space;
    p->p_elements = spec.elements;
    p->p_components = spec.components;
    p->p_src_rank_set = spec.src_rank_set;
    p->p_dst_rank_set = spec.dst_rank_set;
    p->p_n_ids = spec.global_ids.size();
    p->p_n_src_ranks = spec.src_ranks.size();
    p->p_n_dst_ranks = spec.dst_ranks.size();
    p->p_total = total;
    p->planned = true;
    return Result<void>::success();
  }
#endif  // FUSE_LOWFIVE_HAVE_DIST

  auto& g = vol_singleton();
  if (!g.registered) {
    return Failure{"E-TRANSPORT-LF-VOL",
                 "the LowFive VOL connector did not register; "
                 "H5VLis_connector_registered_by_name(\"lowfive\") is not 1. "
                 "Nothing would be intercepted and HDF5 would fall back to "
                 "real files"};
  }

  // Fresh property list and dataspace, both owned for the transport's life so
  // that move() creates neither.
  const hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
  if (fapl < 0) {
    return Failure{"E-TRANSPORT-HDF5", "H5Pcreate(H5P_FILE_ACCESS) failed"};
  }
  g.prop->apply(fapl);

  // Not "the call returned 0": read the connector back off the list and
  // compare it with the one we registered.
  // H5Pget_vol_id hands back a reference, not just a value; releasing it is
  // what keeps the connector's reference count from creeping up once per
  // build_plan.
  hid_t      on_list = H5I_INVALID_HID;
  const bool got = H5Pget_vol_id(fapl, &on_list) >= 0;
  const bool matches = got && on_list == g.vol_id;
  if (on_list >= 0) H5VLclose(on_list);
  if (!matches) {
    H5Pclose(fapl);
    return Failure{"E-TRANSPORT-LF-VOL",
                 "H5Pset_vol did not put the LowFive connector on the "
                 "file-access property list; H5Pget_vol_id reports " +
                     std::to_string(static_cast<long long>(on_list)) +
                     ", expected " +
                     std::to_string(static_cast<long long>(g.vol_id))};
  }

  const auto dim = static_cast<hsize_t>(total);
  const hid_t space = H5Screate_simple(1, &dim, nullptr);
  if (space < 0) {
    H5Pclose(fapl);
    return Failure{"E-TRANSPORT-HDF5",
                 "H5Screate_simple for " + std::to_string(total) +
                     " elements failed"};
  }

  p->fapl = fapl;
  p->space = space;

  // ---------------------------------------------------------------------
  // Prove interception, here and not in the loop. A single probe value is
  // written and read back through the connector; the callback counter and
  // the filesystem are both checked. Anything short of this passes against
  // plain file I/O.
  // ---------------------------------------------------------------------
  const std::size_t before = g.writes;
  const double      probe_in = 1.0;
  double            probe_out = 0.0;
  const hsize_t     one = 1;
  const hid_t       probe_space = H5Screate_simple(1, &one, nullptr);
  if (probe_space < 0) {
    return Failure{"E-TRANSPORT-HDF5", "H5Screate_simple(1) failed"};
  }
  // R-46 (REVIEW.md): residual upstream name, harmless (it never leaves this
  // process's own scratch HDF5 file) but a leftover porting.md renaming would
  // have caught.
  const char* kProbe = "__fuse_lowfive_probe__";
  bool        hdf5_ok = true;
  {
    FileHandle f(H5Fcreate(p->file_c, H5F_ACC_TRUNC, H5P_DEFAULT, p->fapl));
    if (!f.ok()) {
      hdf5_ok = false;
    } else {
      DsetHandle d(H5Dcreate2(f.get(), kProbe, H5T_IEEE_F64LE, probe_space,
                              H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
      hdf5_ok = d.ok() && H5Dwrite(d.get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                                   H5P_DEFAULT, &probe_in) >= 0;
    }
  }
  if (hdf5_ok) {
    FileHandle f(H5Fopen(p->file_c, H5F_ACC_RDONLY, p->fapl));
    if (!f.ok()) {
      hdf5_ok = false;
    } else {
      DsetHandle d(H5Dopen2(f.get(), kProbe, H5P_DEFAULT));
      hdf5_ok = d.ok() && H5Dread(d.get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                                  H5P_DEFAULT, &probe_out) >= 0;
    }
  }
  const bool        fired = g.writes > before;
  const bool        on_disk = path_exists(p->file_c);
  H5Sclose(probe_space);
  safe_drop(g.vol, p->file);
  if (on_disk) ::unlink(p->file_c);  // do not leave the evidence behind

  if (on_disk) {
    return Failure{"E-TRANSPORT-LF-ONDISK",
                 "HDF5 created a real file at '" + p->file +
                     "'. The VOL connector is not intercepting and this "
                     "channel would degrade to disk I/O"};
  }
  if (!fired) {
    return Failure{"E-TRANSPORT-LF-NOINTERCEPT",
                 "LowFive's after_dataset_write callback did not fire for the "
                 "setup probe, so the write did not go through the connector"};
  }
  if (!hdf5_ok) {
    return Failure{"E-TRANSPORT-HDF5",
                 "the setup probe's HDF5 round trip through '" + p->file +
                     "' failed"};
  }
  if (probe_out != probe_in) {
    return Failure{"E-TRANSPORT-LF-NOINTERCEPT",
                 "the setup probe read back " + std::to_string(probe_out) +
                     " instead of " + std::to_string(probe_in) +
                     "; the connector is attached but not carrying data"};
  }

  p->p_elements = spec.elements;
  p->p_components = spec.components;
  p->p_src_rank_set = spec.src_rank_set;
  p->p_dst_rank_set = spec.dst_rank_set;
  p->p_n_ids = spec.global_ids.size();
  p->p_n_src_ranks = spec.src_ranks.size();
  p->p_n_dst_ranks = spec.dst_ranks.size();
  p->p_total = total;
  p->planned = true;
  return Result<void>::success();
#endif
}

// ---------------------------------------------------------------------------
// prepare -- the hot loop. Scalars only.
// ---------------------------------------------------------------------------

Result<void> LowFiveTransport::prepare(const TransportSpec& spec) {
  const Impl* p = impl_.get();
  if (p == nullptr || !p->planned) {
#if !FUSE_HAVE_LOWFIVE
    return no_lowfive("LowFiveTransport::prepare()");
#else
    return Failure{"E-TRANSPORT-UNPLANNED",
                 "prepare() before a successful build_plan(); registering the "
                 "VOL connector and proving interception cannot happen here, "
                 "inside the hot loop"};
#endif
  }
  if (spec.elements != p->p_elements || spec.components != p->p_components ||
      spec.src_rank_set != p->p_src_rank_set ||
      spec.dst_rank_set != p->p_dst_rank_set ||
      spec.global_ids.size() != p->p_n_ids ||
      spec.src_ranks.size() != p->p_n_src_ranks ||
      spec.dst_ranks.size() != p->p_n_dst_ranks) {
    return Failure{"E-TRANSPORT-RESPEC",
                 "prepare() was given a different TransportSpec than "
                 "build_plan() planned; the property list and dataspace are "
                 "sized from it and must not be rebuilt inside the loop"};
  }
  return Result<void>::success();
}

// ---------------------------------------------------------------------------
// move
// ---------------------------------------------------------------------------

Result<void> LowFiveTransport::move(const double* src, double* dst,
                                    std::size_t n) {
  Impl* p = impl_.get();
  if (p == nullptr || !p->planned) {
#if !FUSE_HAVE_LOWFIVE
    return no_lowfive("LowFiveTransport::move()");
#else
    return Failure{"E-TRANSPORT-UNPLANNED",
                 "move() before a successful build_plan()"};
#endif
  }
#if !FUSE_HAVE_LOWFIVE
  (void)src;
  (void)dst;
  (void)n;
  return no_lowfive("LowFiveTransport::move()");
#else
  if (n != p->p_total) {
    return Failure{"E-TRANSPORT-SIZE",
                 "move of " + std::to_string(n) +
                     " doubles, but the planned dataspace holds exactly " +
                     std::to_string(p->p_total) + " (elements * components)"};
  }
  if (src == nullptr || dst == nullptr) {
    return Failure{"E-TRANSPORT-NULL",
                 "lowfive transport given a null pointer for " +
                     std::to_string(n) + " doubles"};
  }
#if FUSE_LOWFIVE_HAVE_DIST
  if (p->distributed) {
    return Failure{"E-TRANSPORT-LF-DISTRIBUTED",
                 "move() writes AND reads back within this one call, which "
                 "is only meaningful when both ends of the channel are in "
                 "this process. In distributed mode (set_intercomm() was "
                 "called) the producer and consumer are two different "
                 "processes: drive the actual write/read through each "
                 "executable's own, unmodified H5Dwrite/H5Dread "
                 "(examples/demo_b), with serve_as_producer()/"
                 "receive_as_consumer() bracketing them -- not through this "
                 "call"};
  }
#endif

  auto&             g = vol_singleton();
  const std::size_t before = g.writes;

  // --- write leg: exactly what an unmodified HDF5 component would emit ---
  {
    FileHandle f(H5Fcreate(p->file_c, H5F_ACC_TRUNC, H5P_DEFAULT, p->fapl));
    if (!f.ok()) {
      return Failure{"E-TRANSPORT-HDF5", "H5Fcreate('" + p->file + "') failed"};
    }
    DsetHandle d(H5Dcreate2(f.get(), p->dataset_c, H5T_IEEE_F64LE, p->space,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    if (!d.ok()) {
      return Failure{"E-TRANSPORT-HDF5",
                   "H5Dcreate2('" + p->dataset + "') failed"};
    }
    if (H5Dwrite(d.get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                 src) < 0) {
      return Failure{"E-TRANSPORT-HDF5", "H5Dwrite failed"};
    }
  }

  // Re-checked every step, not just at setup: a field transfer that stopped being
  // intercepted mid-run must fail, not quietly start writing files.
  if (g.writes == before) {
    return Failure{"E-TRANSPORT-LF-NOINTERCEPT",
                 "the write to '" + p->file +
                     "' did not go through the LowFive connector"};
  }
  // The delta, not ++: the counter is the VOL's, and reporting how many
  // dataset writes it actually saw is the diagnostic worth having.
  p->writes_seen += g.writes - before;
  if (path_exists(p->file_c)) {
    ::unlink(p->file_c);
    return Failure{"E-TRANSPORT-LF-ONDISK",
                 "a real file appeared at '" + p->file +
                     "'; the connector stopped intercepting and this channel "
                     "would degrade to disk I/O"};
  }

  // --- read leg: opening a path that does not exist on disk and getting the
  // data back is itself only possible through the connector ---
  {
    FileHandle f(H5Fopen(p->file_c, H5F_ACC_RDONLY, p->fapl));
    if (!f.ok()) {
      return Failure{"E-TRANSPORT-HDF5", "H5Fopen('" + p->file + "') failed"};
    }
    DsetHandle d(H5Dopen2(f.get(), p->dataset_c, H5P_DEFAULT));
    if (!d.ok()) {
      return Failure{"E-TRANSPORT-HDF5",
                   "H5Dopen2('" + p->dataset + "') failed"};
    }
    if (H5Dread(d.get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                dst) < 0) {
      return Failure{"E-TRANSPORT-HDF5", "H5Dread failed"};
    }
  }

  // Release the in-memory file so the next step starts from a clean map;
  // set_keep(true) would otherwise accumulate one per step.
  safe_drop(g.vol, p->file);
  return Result<void>::success();
#endif
}

}  // namespace fuse
