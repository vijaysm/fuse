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
//    aborts the process after main returns (docs/BUILDING.md has the trace:
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

#include "soma/transport/transport.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifdef SOMA_HAVE_LOWFIVE
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <hdf5.h>
#include <lowfive/H5VOLProperty.hpp>
#include <lowfive/vol-metadata.hpp>
#endif

namespace soma {
namespace {

#ifndef SOMA_HAVE_LOWFIVE
[[nodiscard]] Error no_lowfive(const char* what) {
  return Error{"E-TRANSPORT-NOLOWFIVE",
               std::string(what) +
                   " requires a LowFive-enabled build; this one was configured "
                   "without it. Reconfigure with -DSOMA_ENABLE_LOWFIVE=ON and "
                   "-DCMAKE_PREFIX_PATH=\"/opt/lowfive-shared;/opt/hdf5-shared\" "
                   "(see docs/BUILDING.md)"};
}
#endif

#ifdef SOMA_HAVE_LOWFIVE

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
    // docs/BUILDING.md records the process dying AFTER main returns with
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

#endif  // SOMA_HAVE_LOWFIVE

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

#ifdef SOMA_HAVE_LOWFIVE
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
#ifdef SOMA_HAVE_LOWFIVE
  return true;
#else
  return false;
#endif
}

long long LowFiveTransport::access_plist_handle() const noexcept {
#ifdef SOMA_HAVE_LOWFIVE
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
// set_location
// ---------------------------------------------------------------------------

Result<void> LowFiveTransport::set_location(std::string file,
                                            std::string dataset) {
#ifndef SOMA_HAVE_LOWFIVE
  (void)file;
  (void)dataset;
  return no_lowfive("LowFiveTransport::set_location()");
#else
  Impl* p = impl_.get();
  if (p == nullptr) {
    return Error{"E-TRANSPORT-LF-LOCATION",
                 "set_location() on a moved-from LowFiveTransport"};
  }
  if (p->planned) {
    return Error{"E-TRANSPORT-LF-LOCATION",
                 "set_location() after build_plan(); the property list and "
                 "the interception proof are built from the location and "
                 "cannot be rebuilt in the loop"};
  }
  if (file.empty() || dataset.empty()) {
    return Error{"E-TRANSPORT-LF-LOCATION",
                 "both a file name and a dataset name are required"};
  }

  auto& g = vol_singleton();
  if (std::find(g.claimed.begin(), g.claimed.end(), file) != g.claimed.end()) {
    return Error{"E-TRANSPORT-LF-LOCATION",
                 "another live LowFiveTransport already uses the file name '" +
                     file +
                     "'; LowFive's in-memory file map is process-wide, so the "
                     "two would silently overwrite each other"};
  }

  // "No file on disk" is only evidence if a file COULD have appeared there.
  // Prove the directory is writable by actually creating and removing a
  // sentinel next to the path we are about to claim.
  const std::string sentinel = file + ".soma-writable-probe";
  const int fd = ::open(sentinel.c_str(), O_CREAT | O_EXCL | O_WRONLY, 0600);
  if (fd < 0) {
    return Error{"E-TRANSPORT-LF-LOCATION",
                 "cannot create a file next to '" + file +
                     "', so the absence of that file would prove nothing "
                     "about interception. Point the transport at a writable "
                     "directory"};
  }
  ::close(fd);
  ::unlink(sentinel.c_str());

  if (p->located) {
    auto it = std::find(g.claimed.begin(), g.claimed.end(), p->file);
    if (it != g.claimed.end()) g.claimed.erase(it);
  }
  p->file = std::move(file);
  p->dataset = std::move(dataset);
  p->file_c = p->file.c_str();
  p->dataset_c = p->dataset.c_str();
  p->located = true;
  g.claimed.push_back(p->file);
  return Result<void>::success();
#endif
}

// ---------------------------------------------------------------------------
// build_plan
// ---------------------------------------------------------------------------

Result<void> LowFiveTransport::build_plan(const TransportSpec& spec) {
#ifndef SOMA_HAVE_LOWFIVE
  (void)spec;
  return no_lowfive("LowFiveTransport::build_plan()");
#else
  Impl* p = impl_.get();
  if (p == nullptr) {
    return Error{"E-TRANSPORT-LF-LOCATION",
                 "build_plan() on a moved-from LowFiveTransport"};
  }
  if (!p->located) {
    return Error{"E-TRANSPORT-LF-LOCATION",
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
    return Error{"E-TRANSPORT-SIZE",
                 "components must be at least 1, got " +
                     std::to_string(spec.components)};
  }
  const auto nc = static_cast<std::size_t>(spec.components);
  if (spec.elements == 0) {
    return Error{"E-TRANSPORT-SIZE",
                 "a LowFive channel with zero elements has no I/O to "
                 "intercept; there is nothing for this backend to do"};
  }
  if (spec.elements > (static_cast<std::size_t>(-1) / nc)) {
    return Error{"E-TRANSPORT-SIZE",
                 "elements * components overflows std::size_t"};
  }
  const std::size_t total = spec.elements * nc;

  auto& g = vol_singleton();
  if (!g.registered) {
    return Error{"E-TRANSPORT-LF-VOL",
                 "the LowFive VOL connector did not register; "
                 "H5VLis_connector_registered_by_name(\"lowfive\") is not 1. "
                 "Nothing would be intercepted and HDF5 would fall back to "
                 "real files"};
  }

  // Fresh property list and dataspace, both owned for the transport's life so
  // that move() creates neither.
  const hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
  if (fapl < 0) {
    return Error{"E-TRANSPORT-HDF5", "H5Pcreate(H5P_FILE_ACCESS) failed"};
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
    return Error{"E-TRANSPORT-LF-VOL",
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
    return Error{"E-TRANSPORT-HDF5",
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
    return Error{"E-TRANSPORT-HDF5", "H5Screate_simple(1) failed"};
  }
  const char* kProbe = "__soma_lowfive_probe__";
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
    return Error{"E-TRANSPORT-LF-ONDISK",
                 "HDF5 created a real file at '" + p->file +
                     "'. The VOL connector is not intercepting and this "
                     "channel would degrade to disk I/O"};
  }
  if (!fired) {
    return Error{"E-TRANSPORT-LF-NOINTERCEPT",
                 "LowFive's after_dataset_write callback did not fire for the "
                 "setup probe, so the write did not go through the connector"};
  }
  if (!hdf5_ok) {
    return Error{"E-TRANSPORT-HDF5",
                 "the setup probe's HDF5 round trip through '" + p->file +
                     "' failed"};
  }
  if (probe_out != probe_in) {
    return Error{"E-TRANSPORT-LF-NOINTERCEPT",
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
#ifndef SOMA_HAVE_LOWFIVE
    return no_lowfive("LowFiveTransport::prepare()");
#else
    return Error{"E-TRANSPORT-UNPLANNED",
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
    return Error{"E-TRANSPORT-RESPEC",
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
#ifndef SOMA_HAVE_LOWFIVE
    return no_lowfive("LowFiveTransport::move()");
#else
    return Error{"E-TRANSPORT-UNPLANNED",
                 "move() before a successful build_plan()"};
#endif
  }
#ifndef SOMA_HAVE_LOWFIVE
  (void)src;
  (void)dst;
  (void)n;
  return no_lowfive("LowFiveTransport::move()");
#else
  if (n != p->p_total) {
    return Error{"E-TRANSPORT-SIZE",
                 "move of " + std::to_string(n) +
                     " doubles, but the planned dataspace holds exactly " +
                     std::to_string(p->p_total) + " (elements * components)"};
  }
  if (src == nullptr || dst == nullptr) {
    return Error{"E-TRANSPORT-NULL",
                 "lowfive transport given a null pointer for " +
                     std::to_string(n) + " doubles"};
  }

  auto&             g = vol_singleton();
  const std::size_t before = g.writes;

  // --- write leg: exactly what an unmodified HDF5 component would emit ---
  {
    FileHandle f(H5Fcreate(p->file_c, H5F_ACC_TRUNC, H5P_DEFAULT, p->fapl));
    if (!f.ok()) {
      return Error{"E-TRANSPORT-HDF5", "H5Fcreate('" + p->file + "') failed"};
    }
    DsetHandle d(H5Dcreate2(f.get(), p->dataset_c, H5T_IEEE_F64LE, p->space,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    if (!d.ok()) {
      return Error{"E-TRANSPORT-HDF5",
                   "H5Dcreate2('" + p->dataset + "') failed"};
    }
    if (H5Dwrite(d.get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                 src) < 0) {
      return Error{"E-TRANSPORT-HDF5", "H5Dwrite failed"};
    }
  }

  // Re-checked every step, not just at setup: a channel that stopped being
  // intercepted mid-run must fail, not quietly start writing files.
  if (g.writes == before) {
    return Error{"E-TRANSPORT-LF-NOINTERCEPT",
                 "the write to '" + p->file +
                     "' did not go through the LowFive connector"};
  }
  // The delta, not ++: the counter is the VOL's, and reporting how many
  // dataset writes it actually saw is the diagnostic worth having.
  p->writes_seen += g.writes - before;
  if (path_exists(p->file_c)) {
    ::unlink(p->file_c);
    return Error{"E-TRANSPORT-LF-ONDISK",
                 "a real file appeared at '" + p->file +
                     "'; the connector stopped intercepting and this channel "
                     "would degrade to disk I/O"};
  }

  // --- read leg: opening a path that does not exist on disk and getting the
  // data back is itself only possible through the connector ---
  {
    FileHandle f(H5Fopen(p->file_c, H5F_ACC_RDONLY, p->fapl));
    if (!f.ok()) {
      return Error{"E-TRANSPORT-HDF5", "H5Fopen('" + p->file + "') failed"};
    }
    DsetHandle d(H5Dopen2(f.get(), p->dataset_c, H5P_DEFAULT));
    if (!d.ok()) {
      return Error{"E-TRANSPORT-HDF5",
                   "H5Dopen2('" + p->dataset + "') failed"};
    }
    if (H5Dread(d.get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                dst) < 0) {
      return Error{"E-TRANSPORT-HDF5", "H5Dread failed"};
    }
  }

  // Release the in-memory file so the next step starts from a clean map;
  // set_keep(true) would otherwise accumulate one per step.
  safe_drop(g.vol, p->file);
  return Result<void>::success();
#endif
}

}  // namespace soma
