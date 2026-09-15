// Task 8.4. What this file is trying to be immune to:
//
//   A test that asserts move() returned success proves NOTHING here. Plain
//   HDF5 file I/O also returns success, is also correct, and is exactly the
//   outcome this backend exists to avoid. So every claim about interception
//   is made against a CONTROL that runs the identical HDF5 call sequence with
//   the default (native) property list on the SAME path, and shows a real
//   file appearing there. Absence of a file is only evidence once presence
//   has been demonstrated to be possible.

#include <gtest/gtest.h>
#include <soma/transport/transport.hpp>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <new>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

using soma::LowFiveTransport;
using soma::TransportRegistry;
using soma::TransportSpec;

namespace {

[[nodiscard]] bool on_disk(const std::string& p) {
  struct ::stat s {};
  return ::stat(p.c_str(), &s) == 0;
}

[[nodiscard]] std::string tmp_path(const char* leaf) {
  std::string dir = ::testing::TempDir();
  if (dir.empty() || dir.back() != '/') dir += '/';
  return dir + leaf;
}

}  // namespace

// ---------------------------------------------------------------------------
// True in every configuration.
// ---------------------------------------------------------------------------

TEST(LowFiveTransport, NamedAndNotLocal) {
  const LowFiveTransport t;
  EXPECT_EQ(t.name(), "lowfive");
  // The data leaves ordinary memory for the HDF5 stack; a local transport is
  // what CopyTransport is for.
  EXPECT_FALSE(t.is_local());
}

// The registry entry lives in an always-linked translation unit
// (src/transport/copy_transport.cpp) and is unconditional, so "lowfive" is
// creatable in every build. soma_core is a static archive: an entry
// registered from a self-initialising, otherwise unreferenced object file
// would be present or absent depending on what else the executable linked.
TEST(TransportRegistry, LowFiveIsRegisteredInEveryConfiguration) {
  const auto t = TransportRegistry::instance().create("lowfive");
  ASSERT_TRUE(t.ok()) << t.error().message;
  EXPECT_EQ(t.value()->name(), "lowfive");
  EXPECT_FALSE(t.value()->is_local());
}

TEST(LowFiveTransport, AvailabilityMatchesTheBuild) {
#ifdef SOMA_HAVE_LOWFIVE
  EXPECT_TRUE(LowFiveTransport::is_available());
#else
  EXPECT_FALSE(LowFiveTransport::is_available());
#endif
}

#ifndef SOMA_HAVE_LOWFIVE

// ---------------------------------------------------------------------------
// Compiled out. Every entry point must still LINK and must say why it cannot
// work, naming the option that turns it on.
// ---------------------------------------------------------------------------

TEST(LowFiveTransport, EveryEntryPointFailsLoudlyWithoutLowFive) {
  LowFiveTransport t;

  const auto loc = t.set_location("/tmp/soma-nolowfive.h5", "f");
  ASSERT_FALSE(loc.ok());
  EXPECT_EQ(loc.error().code, "E-TRANSPORT-NOLOWFIVE");
  // The message has to be actionable, not just present.
  EXPECT_NE(loc.error().message.find("SOMA_ENABLE_LOWFIVE"), std::string::npos)
      << loc.error().message;

  const auto plan = t.build_plan(TransportSpec{.elements = 4});
  ASSERT_FALSE(plan.ok());
  EXPECT_EQ(plan.error().code, "E-TRANSPORT-NOLOWFIVE");

  const auto prep = t.prepare(TransportSpec{.elements = 4});
  ASSERT_FALSE(prep.ok());
  EXPECT_EQ(prep.error().code, "E-TRANSPORT-NOLOWFIVE");

  std::vector<double> a(4, 1.0), b(4, 0.0);
  const auto mv = t.move(a.data(), b.data(), 4);
  ASSERT_FALSE(mv.ok());
  EXPECT_EQ(mv.error().code, "E-TRANSPORT-NOLOWFIVE");

  EXPECT_LT(t.access_plist_handle(), 0);
  EXPECT_EQ(t.intercepted_writes(), 0u);
}

// It must not silently degrade to file I/O when it cannot work either.
TEST(LowFiveTransport, WritesNoFileWhenCompiledOut) {
  const std::string path = tmp_path("soma-nolowfive-degrade.h5");
  ::unlink(path.c_str());
  LowFiveTransport t;
  (void)t.set_location(path, "field");
  (void)t.build_plan(TransportSpec{.elements = 4});
  std::vector<double> a(4, 1.0), b(4, 0.0);
  (void)t.move(a.data(), b.data(), 4);
  EXPECT_FALSE(on_disk(path));
}

#else  // SOMA_HAVE_LOWFIVE

#include <hdf5.h>

namespace {

// ---------------------------------------------------------------------------
// An "unmodified HDF5 component". It knows a path, a dataset name and a
// file-access property list id. It has never heard of LowFive or of SOMA.
// ---------------------------------------------------------------------------
bool component_writes_hdf5(const std::string& path, const std::string& dset,
                           hid_t fapl, const std::vector<double>& data) {
  const auto  dim = static_cast<hsize_t>(data.size());
  const hid_t sp = H5Screate_simple(1, &dim, nullptr);
  const hid_t f = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
  if (f < 0) {
    H5Sclose(sp);
    return false;
  }
  const hid_t d = H5Dcreate2(f, dset.c_str(), H5T_IEEE_F64LE, sp, H5P_DEFAULT,
                             H5P_DEFAULT, H5P_DEFAULT);
  bool ok = d >= 0 && H5Dwrite(d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                               H5P_DEFAULT, data.data()) >= 0;
  if (d >= 0) H5Dclose(d);
  H5Sclose(sp);
  H5Fclose(f);
  return ok;
}

bool component_reads_hdf5(const std::string& path, const std::string& dset,
                          hid_t fapl, std::vector<double>& out) {
  const hid_t f = H5Fopen(path.c_str(), H5F_ACC_RDONLY, fapl);
  if (f < 0) return false;
  const hid_t d = H5Dopen2(f, dset.c_str(), H5P_DEFAULT);
  bool ok = d >= 0 && H5Dread(d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                              H5P_DEFAULT, out.data()) >= 0;
  if (d >= 0) H5Dclose(d);
  H5Fclose(f);
  return ok;
}

// Allocation counting, the mechanism test/unit/runner_test.cpp and
// test/parallel/mpi_transport_test.cpp both use. Replacing global operator
// new is the only way to measure the hot-loop claim rather than assert it by
// inspection.
std::size_t g_allocations = 0;
bool        g_counting = false;

class AllocationCounter {
 public:
  AllocationCounter() {
    g_allocations = 0;
    g_counting = true;
  }
  ~AllocationCounter() { g_counting = false; }
  AllocationCounter(const AllocationCounter&) = delete;
  AllocationCounter& operator=(const AllocationCounter&) = delete;
  [[nodiscard]] std::size_t count() const { return g_allocations; }
};

}  // namespace

void* operator new(std::size_t n) {
  if (g_counting) ++g_allocations;
  void* p = std::malloc(n == 0 ? 1 : n);
  if (p == nullptr) throw std::bad_alloc();
  return p;
}
void* operator new[](std::size_t n) { return ::operator new(n); }
void* operator new(std::size_t n, const std::nothrow_t&) noexcept {
  if (g_counting) ++g_allocations;
  return std::malloc(n == 0 ? 1 : n);
}
void* operator new[](std::size_t n, const std::nothrow_t& t) noexcept {
  return ::operator new(n, t);
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

// ---------------------------------------------------------------------------
// THE CONTROL. Without this, every "no file on disk" assertion below is
// unfalsifiable: it would also hold if the path were unwritable, if HDF5 were
// broken, or if nothing had been written at all.
// ---------------------------------------------------------------------------

TEST(LowFiveInterception, PlainHdf5OnTheSamePathDoesCreateAFile) {
  const std::string path = tmp_path("soma-lf-control.h5");
  ::unlink(path.c_str());
  ASSERT_FALSE(on_disk(path));

  const std::vector<double> data{1.0, 2.0, 3.0, 4.0};
  // H5P_DEFAULT: the native VOL, no interception.
  ASSERT_TRUE(component_writes_hdf5(path, "field", H5P_DEFAULT, data));

  EXPECT_TRUE(on_disk(path))
      << "the identical HDF5 call sequence with the native VOL must produce a "
         "real file, or the interception tests below assert nothing";
  struct ::stat s {};
  ASSERT_EQ(::stat(path.c_str(), &s), 0);
  EXPECT_GT(s.st_size, 0);

  std::vector<double> back(4, 0.0);
  ASSERT_TRUE(component_reads_hdf5(path, "field", H5P_DEFAULT, back));
  EXPECT_EQ(back, data);
  ::unlink(path.c_str());
}

// ---------------------------------------------------------------------------
// The claim: same path, same calls, connector attached -> data arrives and no
// file exists.
// ---------------------------------------------------------------------------

TEST(LowFiveInterception, SamePathIsWrittenByPlainHdf5AndNotByTheTransport) {
  const std::string path = tmp_path("soma-lf-samepath.h5");
  ::unlink(path.c_str());

  const std::vector<double> data{10.5, -2.25, 0.0, 7.125};

  // (a) Native VOL on this exact path: a file appears.
  ASSERT_TRUE(component_writes_hdf5(path, "field", H5P_DEFAULT, data));
  ASSERT_TRUE(on_disk(path)) << "control leg failed; path is not writable";
  ::unlink(path.c_str());
  ASSERT_FALSE(on_disk(path));

  // (b) Same path through the transport: data round trips, nothing appears.
  LowFiveTransport t;
  ASSERT_TRUE(t.set_location(path, "field").ok());
  const auto plan = t.build_plan(TransportSpec{.elements = 4});
  ASSERT_TRUE(plan.ok()) << plan.error().code << ": " << plan.error().message;
  ASSERT_TRUE(t.prepare(TransportSpec{.elements = 4}).ok());

  std::vector<double> dst(4, -999.0);
  const auto          mv = t.move(data.data(), dst.data(), 4);
  ASSERT_TRUE(mv.ok()) << mv.error().code << ": " << mv.error().message;

  EXPECT_EQ(dst, data) << "the data did not arrive through the VOL path";
  EXPECT_FALSE(on_disk(path))
      << "a real file was created at " << path
      << "; the connector did not intercept and the channel degraded to disk";
  // LowFive's own dataset_write handler ran. Nothing in the native HDF5 path
  // can move this counter.
  EXPECT_EQ(t.intercepted_writes(), 1u);
}

// The FAPL handed to an otherwise untouched HDF5 program is the whole point
// of the backend: the component is not rewritten, only its file-access
// property list is.
TEST(LowFiveInterception, UnmodifiedHdf5ComponentIsIntercepted) {
  const std::string path = tmp_path("soma-lf-component.h5");
  ::unlink(path.c_str());

  LowFiveTransport t;
  ASSERT_TRUE(t.set_location(path, "temperature").ok());
  ASSERT_TRUE(t.build_plan(TransportSpec{.elements = 6}).ok());

  const long long handle = t.access_plist_handle();
  ASSERT_GE(handle, 0);
  const auto fapl = static_cast<hid_t>(handle);

  const std::vector<double> produced{0.5, 1.5, 2.5, 3.5, 4.5, 5.5};
  ASSERT_TRUE(component_writes_hdf5(path, "temperature", fapl, produced));
  EXPECT_FALSE(on_disk(path)) << "the component's write hit the filesystem";

  std::vector<double> consumed(6, 0.0);
  ASSERT_TRUE(component_reads_hdf5(path, "temperature", fapl, consumed));
  EXPECT_EQ(consumed, produced);
  EXPECT_FALSE(on_disk(path));
}

// Reading a path that does not exist is impossible without interception --
// this is the read-side counterpart of the write-side callback check.
TEST(LowFiveInterception, ReadOfANonexistentPathSucceedsOnlyBecauseOfTheVol) {
  const std::string path = tmp_path("soma-lf-readonly.h5");
  ::unlink(path.c_str());

  LowFiveTransport t;
  ASSERT_TRUE(t.set_location(path, "field").ok());
  ASSERT_TRUE(t.build_plan(TransportSpec{.elements = 3}).ok());
  const auto fapl = static_cast<hid_t>(t.access_plist_handle());

  const std::vector<double> produced{9.0, 8.0, 7.0};
  ASSERT_TRUE(component_writes_hdf5(path, "field", fapl, produced));
  ASSERT_FALSE(on_disk(path));

  // The native VOL cannot open it: there is no file.
  H5E_auto2_t old_fn = nullptr;
  void*       old_data = nullptr;
  H5Eget_auto2(H5E_DEFAULT, &old_fn, &old_data);
  H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
  std::vector<double> native(3, 0.0);
  const bool native_ok = component_reads_hdf5(path, "field", H5P_DEFAULT, native);
  H5Eset_auto2(H5E_DEFAULT, old_fn, old_data);
  EXPECT_FALSE(native_ok)
      << "the native VOL opened " << path
      << ", which means a real file exists and nothing was intercepted";

  // The intercepting FAPL can.
  std::vector<double> through(3, 0.0);
  ASSERT_TRUE(component_reads_hdf5(path, "field", fapl, through));
  EXPECT_EQ(through, produced);
}

// ---------------------------------------------------------------------------
// The hot loop. Task 4.6's constraint, measured, not asserted by inspection.
//
// Only prepare() is claimed to be allocation-free, and that is the whole of
// the contract in include/soma/transport/transport.hpp. move() cannot be:
// it calls into HDF5 and LowFive, both of which allocate internally, and no
// amount of restructuring on this side of the boundary changes that. The
// count from move() is printed rather than asserted, so a regression that
// makes it far worse is at least visible.
// ---------------------------------------------------------------------------

TEST(LowFiveTransport, PrepareAllocatesNothingPerStep) {
  const std::string path = tmp_path("soma-lf-alloc.h5");
  ::unlink(path.c_str());

  LowFiveTransport t;
  const TransportSpec spec{.src_rank_set = 0, .dst_rank_set = 1, .elements = 4};
  ASSERT_TRUE(t.set_location(path, "field").ok());
  ASSERT_TRUE(t.build_plan(spec).ok());

  // A warm-up outside the counter, exactly as the runner's
  // AllocatesNothingWhileRunning does, so something allocated lazily on first
  // use is still caught by the counted steps.
  ASSERT_TRUE(t.prepare(spec).ok());

  std::size_t allocations = 0;
  bool        ok = true;
  {
    const AllocationCounter counter;
    for (int step = 0; step < 8; ++step) ok = ok && t.prepare(spec).ok();
    allocations = counter.count();
  }
  ASSERT_TRUE(ok);
  std::printf("[alloc] %zu allocation(s) across 8 prepare()\n", allocations);
  std::fflush(stdout);
  EXPECT_EQ(allocations, 0u)
      << "prepare() allocated " << allocations
      << " times over 8 steps; the VOL registration, the property list and "
         "the dataspace must be built in build_plan(), not in the hot loop";
}

TEST(LowFiveTransport, AllocationCounterActuallyCounts) {
  // A check that cannot fail proves nothing; show the counter is live.
  std::size_t allocations = 0;
  {
    const AllocationCounter counter;
    std::vector<double> v(4096, 1.0);
    allocations = counter.count();
    EXPECT_DOUBLE_EQ(v[0], 1.0);
  }
  EXPECT_GE(allocations, 1u);
}

TEST(LowFiveTransport, MoveAllocationCountIsReportedNotZero) {
  // Documenting the honest limit: move() goes through HDF5 and LowFive, so it
  // allocates. If this ever came out zero the measurement would be broken.
  const std::string path = tmp_path("soma-lf-movealloc.h5");
  LowFiveTransport t;
  const TransportSpec spec{.elements = 4};
  ASSERT_TRUE(t.set_location(path, "field").ok());
  ASSERT_TRUE(t.build_plan(spec).ok());
  const std::vector<double> src{1.0, 2.0, 3.0, 4.0};
  std::vector<double>       dst(4, 0.0);
  ASSERT_TRUE(t.move(src.data(), dst.data(), 4).ok());  // warm up

  std::size_t allocations = 0;
  bool        ok = true;
  {
    const AllocationCounter counter;
    ok = t.move(src.data(), dst.data(), 4).ok();
    allocations = counter.count();
  }
  ASSERT_TRUE(ok);
  std::printf("[alloc] %zu allocation(s) in one move()\n", allocations);
  std::fflush(stdout);
  EXPECT_GT(allocations, 0u);
}

// ---------------------------------------------------------------------------
// Repetition: the same transport, many steps, changing data. A backend that
// cached the first step's dataset would pass a single-step test.
// ---------------------------------------------------------------------------

TEST(LowFiveInterception, EachStepCarriesItsOwnData) {
  const std::string path = tmp_path("soma-lf-steps.h5");
  ::unlink(path.c_str());

  LowFiveTransport    t;
  const TransportSpec spec{.elements = 5};
  ASSERT_TRUE(t.set_location(path, "field").ok());
  ASSERT_TRUE(t.build_plan(spec).ok());

  for (int step = 0; step < 8; ++step) {
    std::vector<double> src(5);
    for (std::size_t i = 0; i < src.size(); ++i) {
      src[i] = static_cast<double>(step) * 100.0 + static_cast<double>(i);
    }
    std::vector<double> dst(5, -1.0);
    ASSERT_TRUE(t.prepare(spec).ok()) << "step " << step;
    const auto mv = t.move(src.data(), dst.data(), 5);
    ASSERT_TRUE(mv.ok()) << "step " << step << ": " << mv.error().message;
    EXPECT_EQ(dst, src) << "step " << step;
    ASSERT_FALSE(on_disk(path)) << "step " << step;
  }
  EXPECT_EQ(t.intercepted_writes(), 8u);
}

// Multiple components per entity: the dataspace is elements * components.
TEST(LowFiveInterception, MultiComponentFieldsRoundTrip) {
  const std::string path = tmp_path("soma-lf-multicomp.h5");
  LowFiveTransport  t;
  const TransportSpec spec{.elements = 3, .components = 2};
  ASSERT_TRUE(t.set_location(path, "velocity").ok());
  ASSERT_TRUE(t.build_plan(spec).ok());

  const std::vector<double> src{1.0, -1.0, 2.0, -2.0, 3.0, -3.0};
  std::vector<double>       dst(6, 0.0);
  ASSERT_TRUE(t.prepare(spec).ok());
  ASSERT_TRUE(t.move(src.data(), dst.data(), 6).ok());
  EXPECT_EQ(dst, src);
  EXPECT_FALSE(on_disk(path));

  // One short of the planned size is still rejected -- enforcement, not an
  // assert, so it holds under -O3 -DNDEBUG too.
  const auto r = t.move(src.data(), dst.data(), 5);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-TRANSPORT-SIZE");
}

// ---------------------------------------------------------------------------
// Error paths.
// ---------------------------------------------------------------------------

TEST(LowFiveTransport, PrepareBeforeBuildPlanIsAnError) {
  LowFiveTransport t;
  const auto       r = t.prepare(TransportSpec{.elements = 4});
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-TRANSPORT-UNPLANNED");
}

TEST(LowFiveTransport, MoveBeforeBuildPlanIsAnError) {
  LowFiveTransport    t;
  std::vector<double> a(4, 1.0), b(4, 0.0);
  const auto          r = t.move(a.data(), b.data(), 4);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-TRANSPORT-UNPLANNED");
}

TEST(LowFiveTransport, BuildPlanBeforeSetLocationIsAnError) {
  LowFiveTransport t;
  const auto       r = t.build_plan(TransportSpec{.elements = 4});
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-TRANSPORT-LF-LOCATION");
}

TEST(LowFiveTransport, PrepareRejectsADifferentSpec) {
  const std::string path = tmp_path("soma-lf-respec.h5");
  LowFiveTransport  t;
  ASSERT_TRUE(t.set_location(path, "field").ok());
  ASSERT_TRUE(t.build_plan(TransportSpec{.elements = 4}).ok());
  const auto r = t.prepare(TransportSpec{.elements = 5});
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-TRANSPORT-RESPEC");
}

TEST(LowFiveTransport, MoveRejectsNullAndWrongSize) {
  const std::string path = tmp_path("soma-lf-guards.h5");
  LowFiveTransport  t;
  ASSERT_TRUE(t.set_location(path, "field").ok());
  ASSERT_TRUE(t.build_plan(TransportSpec{.elements = 4}).ok());
  std::vector<double> a(4, 1.0), b(4, 0.0);

  auto r = t.move(nullptr, b.data(), 4);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-TRANSPORT-NULL");
  r = t.move(a.data(), nullptr, 4);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-TRANSPORT-NULL");
  r = t.move(a.data(), b.data(), 8);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-TRANSPORT-SIZE");
}

// "No file on disk" must not be achievable by pointing the transport at a
// place where no file could be created in the first place.
TEST(LowFiveTransport, RejectsAnUnwritableLocation) {
  LowFiveTransport t;
  const auto r = t.set_location("/soma-no-such-directory-8f3a/x.h5", "field");
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-TRANSPORT-LF-LOCATION");
  EXPECT_NE(r.error().message.find("prove nothing"), std::string::npos)
      << r.error().message;
}

TEST(LowFiveTransport, RejectsEmptyNames) {
  LowFiveTransport t;
  EXPECT_EQ(t.set_location("", "field").error().code,
            "E-TRANSPORT-LF-LOCATION");
  EXPECT_EQ(t.set_location(tmp_path("soma-lf-empty.h5"), "").error().code,
            "E-TRANSPORT-LF-LOCATION");
}

// LowFive's in-memory file map is process-wide. Two transports on one name
// would overwrite each other's datasets with no diagnostic at all.
TEST(LowFiveTransport, TwoTransportsCannotClaimOneFileName) {
  const std::string path = tmp_path("soma-lf-shared-name.h5");
  LowFiveTransport  a;
  ASSERT_TRUE(a.set_location(path, "field").ok());
  {
    LowFiveTransport b;
    const auto       r = b.set_location(path, "field");
    ASSERT_FALSE(r.ok());
    EXPECT_EQ(r.error().code, "E-TRANSPORT-LF-LOCATION");
  }
}

// ...and the claim is released when the transport dies, or a run that creates
// channels in a loop would fail on its second pass.
TEST(LowFiveTransport, TheNameIsReleasedWhenTheTransportDies) {
  const std::string path = tmp_path("soma-lf-reclaim.h5");
  {
    LowFiveTransport a;
    ASSERT_TRUE(a.set_location(path, "field").ok());
  }
  LowFiveTransport b;
  EXPECT_TRUE(b.set_location(path, "field").ok());
}

TEST(LowFiveTransport, RejectsAChannelWithNothingToIntercept) {
  const std::string path = tmp_path("soma-lf-empty-spec.h5");
  LowFiveTransport  t;
  ASSERT_TRUE(t.set_location(path, "field").ok());
  const auto r = t.build_plan(TransportSpec{.elements = 0});
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-TRANSPORT-SIZE");
}

TEST(LowFiveTransport, SetLocationAfterBuildPlanIsAnError) {
  const std::string path = tmp_path("soma-lf-relocate.h5");
  LowFiveTransport  t;
  ASSERT_TRUE(t.set_location(path, "field").ok());
  ASSERT_TRUE(t.build_plan(TransportSpec{.elements = 4}).ok());
  const auto r = t.set_location(tmp_path("soma-lf-relocate2.h5"), "field");
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-TRANSPORT-LF-LOCATION");
}

// The registry-created instance is a real one, not a stub.
TEST(LowFiveInterception, TheRegistryHandsBackAWorkingTransport) {
  const auto made = TransportRegistry::instance().create("lowfive");
  ASSERT_TRUE(made.ok()) << made.error().message;
  auto* t = dynamic_cast<LowFiveTransport*>(made.value().get());
  ASSERT_NE(t, nullptr);

  const std::string path = tmp_path("soma-lf-registry.h5");
  ::unlink(path.c_str());
  ASSERT_TRUE(t->set_location(path, "field").ok());
  ASSERT_TRUE(t->build_plan(TransportSpec{.elements = 2}).ok());
  const std::vector<double> src{42.0, -42.0};
  std::vector<double>       dst(2, 0.0);
  ASSERT_TRUE(t->prepare(TransportSpec{.elements = 2}).ok());
  ASSERT_TRUE(t->move(src.data(), dst.data(), 2).ok());
  EXPECT_EQ(dst, src);
  EXPECT_FALSE(on_disk(path));
}

#endif  // SOMA_HAVE_LOWFIVE
