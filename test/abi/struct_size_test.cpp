// ABI-SHORT-STRUCT, ABI-LONG-STRUCT (build-evidence/abi-matrix.toml).
//
// Both rows are about `struct_size` as a FIELD VALUE, not about how much
// memory the caller actually allocated: a real fuse_setup_options/fuse_capability_set
// object is always fully, safely constructed here, and only its
// `struct_size` member is overwritten to the value under test. That is
// exactly rule 2's contract -- the callee decides from the number the
// caller wrote, never from the caller's real allocation, which this test
// has no way to inspect anyway.
#include "fuse/base/config.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

#include "fuse/abi/participant.h"
#include "fuse/model/descriptor.hpp"
#include "fuse/model/coupling_graph.hpp"

#include "graph_fixture.hpp"

namespace {

const fuse_participant_abi_v1* Table() {
  const auto* t = fuse_coupler_abi(FUSE_ABI_VERSION);
  return t;
}

/// A loaded (but not yet initialized) context: enough state for
/// fuse_setup_options()'s struct_size behaviour, nothing more.
fuse_ctx* LoadedContext(const fuse_participant_abi_v1* t) {
  fuse_context_out co{.struct_size = sizeof(co)};
  EXPECT_EQ(t->create(nullptr, nullptr, &co), FUSE_OK);
  const std::vector<std::byte> bytes = fuse::serialize(fuse::CouplingGraph{});
  fuse_graph_source src{.struct_size = sizeof(src)};
  src.bytes = bytes.data();
  src.byte_count = bytes.size();
  EXPECT_EQ(t->load_graph(co.ctx, &src, nullptr), FUSE_OK);
  return co.ctx;
}

}  // namespace

TEST(AbiStructSize, ShortStructIsRefused) {
  const auto* t = Table();
  ASSERT_NE(t, nullptr);

  // The OUT direction: fuse_capability_set, through an entry (ctx may be
  // NULL) that needs no live context at all.
  for (const std::uint32_t bad : {0u, 1u, 2u, 3u}) {
    fuse_capability_set caps{.struct_size = sizeof(fuse_capability_set)};
    caps.struct_size = bad;
    EXPECT_EQ(t->capabilities(nullptr, nullptr, &caps), FUSE_ERR_VERSION)
        << "struct_size = " << bad;
  }
  {
    fuse_capability_set caps{};
    caps.struct_size = FUSE_ABI_MAX_STRUCT_SIZE + 1;
    EXPECT_EQ(t->capabilities(nullptr, nullptr, &caps), FUSE_ERR_VERSION);
  }
  {
    fuse_capability_set caps{};
    caps.struct_size = 0xFFFFFFFFu;
    EXPECT_EQ(t->capabilities(nullptr, nullptr, &caps), FUSE_ERR_VERSION);
  }

  // The IN direction: fuse_setup_options, on a real loaded context.
  fuse_ctx* ctx = LoadedContext(t);
  fuse_setup_options setup{.struct_size = sizeof(fuse_setup_options)};
  setup.communicator = 0;
  setup.has_end_time = 0;
  setup.struct_size  = 2;  // shorter than any frozen layout (>= 4)
  EXPECT_EQ(t->setup(ctx, &setup, nullptr), FUSE_ERR_VERSION);
  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}

TEST(AbiStructSize, LongerStructFromANewerCallerIsAccepted) {
  const auto* t = Table();
  ASSERT_NE(t, nullptr);

  // OUT direction: a caller whose header declares fuse_capability_set with 64
  // extra trailing bytes this library's layout does not have must still get
  // a correct answer in the bytes this library DOES know about.
  fuse_capability_set caps{.struct_size = sizeof(fuse_capability_set) + 64};
  EXPECT_EQ(t->capabilities(nullptr, nullptr, &caps), FUSE_OK);
  EXPECT_NE(caps.flags, 0u) << "FUSE_CAP_YAML is always set";

  // IN direction: fuse_setup_options, same story.
  fuse_ctx* ctx = LoadedContext(t);
  fuse_setup_options setup{.struct_size = sizeof(fuse_setup_options) + 64};
  setup.communicator = 0;
  setup.has_end_time = 0;
  EXPECT_EQ(t->setup(ctx, &setup, nullptr), FUSE_OK);
  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}

// AbiStructSize.OldCallerStructSizeNeverReadsMyComponent (fuse_setup_options'
// own growth, examples/brusselator/participants: Conformance Unit 2).
// Heap-allocates EXACTLY the pre-growth byte count -- the same
// offsetof-derived expression src/abi/participant.cpp's own
// FirstFrozenStructSize<fuse_setup_options> specialization uses -- so a
// read past it is a genuine heap-buffer-overflow, not a benign read of
// slack the allocator happened to round up to. An old caller's buffer
// really is exactly this size; my_component did not exist when it was
// compiled, so fuse_setup() must never read the bytes where it would live.
TEST(AbiStructSize, OldCallerStructSizeNeverReadsMyComponent) {
  constexpr std::size_t kOldSize =
      offsetof(fuse_setup_options, end_time) + sizeof(fuse_exact_time);
  static_assert(kOldSize < sizeof(fuse_setup_options),
               "my_component must have grown fuse_setup_options (rule 3)");

  const auto* t = Table();
  ASSERT_NE(t, nullptr);
  fuse_ctx* ctx = LoadedContext(t);

  auto buf = std::make_unique<std::byte[]>(kOldSize);
  auto* old_opts = reinterpret_cast<fuse_setup_options*>(buf.get());
  old_opts->struct_size  = static_cast<std::uint32_t>(kOldSize);
  old_opts->communicator = 0;
  old_opts->has_end_time = 0;
  old_opts->reserved     = 0;
  old_opts->end_time     = fuse_exact_time{.numerator = 0, .denominator = 1};
  EXPECT_EQ(t->setup(ctx, old_opts, nullptr), FUSE_OK);

  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}

// AbiStructSize.OldCallerStructSizeNeverReadsMyComponent (above) allocates
// exactly kOldSize bytes and checks only that setup()/destroy() still
// return FUSE_OK -- it never inspects the bytes impl_setup() would read if
// the struct_size_covers() guard were removed, so a plain (non-ASan) build
// does not reliably observe the out-of-bounds read: whatever garbage byte
// pattern the allocator happens to leave past the buffer is read into a
// local variable and, on most runs, never dereferenced in a way the test
// can see. The mutation gate's own row (`abi-setup-reads-my-component-
// past-an-old-callers-struct-size`, build-evidence/mutations/abi.toml)
// therefore SURVIVES a tier-0 run: no sanitizer, no fault.
//
// This test makes the same out-of-bounds read fault DETERMINISTICALLY, on
// every run, with no sanitizer required: `old_opts` is placed so that
// every byte from kOldSize onward -- including all 8 bytes of the
// `my_component` pointer field itself -- lies on a SEPARATE mmap'd page
// immediately following it, and that page is `mprotect`'d PROT_NONE. The
// correct code, gated by `struct_size_covers()`, never reads at or past
// `kOldSize` and returns normally; code that removes the gate (the
// mutation) touches the protected page and the process receives SIGBUS/
// SIGSEGV immediately, which ctest reports as a failed test -- a real
// kill, not a probabilistic one.
TEST(AbiStructSize, OldCallerStructSizeNeverReadsPastItsOwnAllocation) {
  constexpr std::size_t kOldSize =
      offsetof(fuse_setup_options, end_time) + sizeof(fuse_exact_time);
  static_assert(kOldSize < sizeof(fuse_setup_options),
               "my_component must have grown fuse_setup_options (rule 3)");
  static_assert(kOldSize + sizeof(fuse_setup_options::my_component) <=
                    sizeof(fuse_setup_options),
               "my_component must fit the frozen struct's own tail");

  const long page = ::sysconf(_SC_PAGESIZE);
  ASSERT_GT(page, 0);
  ASSERT_LE(static_cast<long>(kOldSize), page)
      << "the old-caller prefix must fit in one page below the guard page";

  void* region = ::mmap(nullptr, static_cast<std::size_t>(2) * static_cast<std::size_t>(page),
                        PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(region, MAP_FAILED);
  std::byte* page0 = reinterpret_cast<std::byte*>(region);
  std::byte* page1 = page0 + page;
  // Every byte at or past kOldSize (my_component's own 8 bytes included)
  // sits on page1; page1 is unreadable, so any access to it faults.
  ASSERT_EQ(::mprotect(page1, static_cast<std::size_t>(page), PROT_NONE), 0);
  std::byte* raw = page1 - kOldSize;

  auto* old_opts = reinterpret_cast<fuse_setup_options*>(raw);
  old_opts->struct_size  = static_cast<std::uint32_t>(kOldSize);
  old_opts->communicator = 0;
  old_opts->has_end_time = 0;
  old_opts->reserved     = 0;
  old_opts->end_time     = fuse_exact_time{.numerator = 0, .denominator = 1};

  const auto* t = Table();
  ASSERT_NE(t, nullptr);
  fuse_ctx* ctx = LoadedContext(t);
  EXPECT_EQ(t->setup(ctx, old_opts, nullptr), FUSE_OK);
  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);

  ::munmap(region, static_cast<std::size_t>(2) * static_cast<std::size_t>(page));
}

// KNOWN_ISSUES.md D76: has_communicator's own growth, same discipline as
// OldCallerStructSizeNeverReadsMyComponent just above, one field later.
// Heap-allocates EXACTLY the pre-D76 byte count (my_component's own last
// pre-growth member, the same expression participant.cpp reads
// has_communicator under via struct_size_covers()) -- a caller built
// before D76 never allocated has_communicator, so fuse_setup() must never
// read the bytes where it would live, and must fall back to the ORIGINAL
// sentinel (communicator == 0 means none supplied).
TEST(AbiStructSize, OldCallerStructSizeNeverReadsHasCommunicator) {
  constexpr std::size_t kOldSize =
      offsetof(fuse_setup_options, my_component) + sizeof(const char*);
  static_assert(kOldSize < sizeof(fuse_setup_options),
               "has_communicator must have grown fuse_setup_options (rule 3)");

  const auto* t = Table();
  ASSERT_NE(t, nullptr);
  fuse_ctx* ctx = LoadedContext(t);

  auto buf = std::make_unique<std::byte[]>(kOldSize);
  auto* old_opts = reinterpret_cast<fuse_setup_options*>(buf.get());
  old_opts->struct_size  = static_cast<std::uint32_t>(kOldSize);
  old_opts->communicator = 0;
  old_opts->has_end_time = 0;
  old_opts->reserved     = 0;
  old_opts->end_time     = fuse_exact_time{.numerator = 0, .denominator = 1};
  old_opts->my_component = nullptr;
  // communicator == 0, has_communicator never allocated: setup() must read
  // this exactly as it did before D76 -- no communicator supplied, no
  // refusal, no read past kOldSize.
  EXPECT_EQ(t->setup(ctx, old_opts, nullptr), FUSE_OK);

  fuse_base_communicator_info info{.struct_size = sizeof(info)};
  EXPECT_EQ(t->base_communicator(ctx, nullptr, &info), FUSE_OK);
  EXPECT_EQ(info.formed, 0) << "an old caller's communicator == 0 must still mean "
                               "\"none supplied\", not D76's new flagged-zero case";

  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}


// D14 Stage B (KNOWN_ISSUES.md): base_communicator(), appended after
// view_field (entries.def) -- a local, non-collective read of
// c.communicator/c.communicator_set, legal at ANY context state
// (src/abi/participant.cpp's own impl_base_communicator() doc comment).
// Exercised here, alongside the OTHER struct_size rows this file
// already covers, because it needs no real MPI_Init() at all when
// my_component is left NULL (impl_setup()'s own my_component-gated
// connect() branch is then never reached) -- exactly this file's own
// "a real object, only struct_size overwritten" discipline.

TEST(AbiStructSize, BaseCommunicatorDefaultsToZeroFormedBeforeSetup) {
  const auto* t = Table();
  ASSERT_NE(t, nullptr);
  fuse_ctx* ctx = LoadedContext(t);

  fuse_base_communicator_info info{.struct_size = sizeof(info)};
  info.communicator = 0xDEAD;  // poisoned, so a no-op callee could not
  info.formed = 1;             // accidentally read back as "formed"
  EXPECT_EQ(t->base_communicator(ctx, nullptr, &info), FUSE_OK);
  EXPECT_EQ(info.formed, 0);
  EXPECT_EQ(info.communicator, 0);

  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}

// setup()'s own communicator refusal (participant.cpp:609-614,
// E-ABI-SETUP-NOMPI) means a NONZERO communicator is refused outright
// whenever this build has no MPI -- exactly this tier-0 build
// (FUSE_HAVE_MPI is 0 here; struct_size_test.cpp's own other tests
// above already only ever pass communicator = 0 for the same reason).
// A communicator = 0 setup() is still a real, complete setup() call,
// so it proves the OTHER half of base_communicator()'s own contract:
// an InProcess-style setup() that never supplied a communicator at all
// must leave `formed` at 0 afterwards, not just before setup() as
// BaseCommunicatorDefaultsToZeroFormedBeforeSetup already covers.
TEST(AbiStructSize, BaseCommunicatorStaysUnformedAfterACommunicatorlessSetup) {
  const auto* t = Table();
  ASSERT_NE(t, nullptr);
  fuse_ctx* ctx = LoadedContext(t);

  fuse_setup_options setup{.struct_size = sizeof(setup)};
  setup.communicator = 0;
  setup.has_end_time = 0;
  EXPECT_EQ(t->setup(ctx, &setup, nullptr), FUSE_OK);

  fuse_base_communicator_info info{.struct_size = sizeof(info)};
  EXPECT_EQ(t->base_communicator(ctx, nullptr, &info), FUSE_OK);
  EXPECT_EQ(info.formed, 0);
  EXPECT_EQ(info.communicator, 0);

  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}

// struct_size negotiation (rule 2): a struct_size shorter than the
// first frozen layout is FUSE_ERR_VERSION, mirroring ShortStructIsRefused
// above for fuse_capability_set. Exercised at the (formed=0) state every
// path above already reaches -- struct_size negotiation is about the
// SHAPE of the OUT struct, not the value underneath it.
TEST(AbiStructSize, BaseCommunicatorInfoShortStructIsRefused) {
  const auto* t = Table();
  ASSERT_NE(t, nullptr);
  fuse_ctx* ctx = LoadedContext(t);
  for (const std::uint32_t bad : {0u, 1u, 2u, 3u}) {
    fuse_base_communicator_info info{};
    info.struct_size = bad;
    EXPECT_EQ(t->base_communicator(ctx, nullptr, &info), FUSE_ERR_VERSION)
        << "struct_size = " << bad;
  }
  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}

// An OLD caller's own, pre-growth struct_size (this struct's own FIRST
// frozen layout IS its current one -- it was appended whole, D14 Stage
// B -- so this is the same `sizeof` both the caller and the library
// agree on; included for parity with OldCallerStructSizeNeverReadsMyComponent
// above, proving a caller built against exactly this layout, not a
// future-grown one, still gets a correct answer).
TEST(AbiStructSize, BaseCommunicatorInfoExactFrozenSizeIsAccepted) {
  const auto* t = Table();
  ASSERT_NE(t, nullptr);
  fuse_ctx* ctx = LoadedContext(t);

  fuse_base_communicator_info info{};
  info.struct_size = sizeof(fuse_base_communicator_info);
  EXPECT_EQ(t->base_communicator(ctx, nullptr, &info), FUSE_OK);
  EXPECT_EQ(info.formed, 0);
  EXPECT_EQ(info.communicator, 0);

  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}

// A NEWER caller's own, longer struct (64 extra trailing bytes this
// library's layout does not have) must still get a correct answer in
// the bytes this library DOES know about -- mirrors
// LongerStructFromANewerCallerIsAccepted above.
TEST(AbiStructSize, BaseCommunicatorInfoLongerStructFromANewerCallerIsAccepted) {
  const auto* t = Table();
  ASSERT_NE(t, nullptr);
  fuse_ctx* ctx = LoadedContext(t);

  fuse_base_communicator_info info{.struct_size = sizeof(fuse_base_communicator_info) + 64};
  EXPECT_EQ(t->base_communicator(ctx, nullptr, &info), FUSE_OK);
  EXPECT_EQ(info.formed, 0);
  EXPECT_EQ(info.communicator, 0);

  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}

// The `formed = 1` / real-value-readback half of base_communicator()'s
// contract needs a NONZERO fuse_setup_options::communicator, which
// setup() refuses outright in any build without MPI (E-ABI-SETUP-NOMPI,
// participant.cpp:609-614) -- exactly this translation unit's own tier-0
// build (FUSE_HAVE_MPI is 0 here). Guarded so it exists only in an
// MPI-enabled build; even there it calls no actual MPI function: with
// my_component left NULL, impl_setup()'s my_component-gated connect()
// branch (the only place that ever calls MPI_Comm_f2c on this value,
// participant.cpp:888-894 / :1061) is never reached, so the literal 99
// below is stored and read back VERBATIM without MPI_Init() ever
// running -- safe to BUILD-VERIFY here; per this track's own standing
// rule against running anything that touches real MPI, it is left for
// the owner's host run (host_pending), not executed in this sandbox.
#if FUSE_HAVE_MPI
TEST(AbiStructSize, BaseCommunicatorReadsBackAFormedCommunicatorWhenMpiIsAvailable) {
  const auto* t = Table();
  ASSERT_NE(t, nullptr);
  fuse_ctx* ctx = LoadedContext(t);

  fuse_setup_options setup{.struct_size = sizeof(setup)};
  setup.communicator = 99;
  setup.has_end_time = 0;
  EXPECT_EQ(t->setup(ctx, &setup, nullptr), FUSE_OK);

  fuse_base_communicator_info info{.struct_size = sizeof(info)};
  EXPECT_EQ(t->base_communicator(ctx, nullptr, &info), FUSE_OK);
  EXPECT_EQ(info.formed, 1);
  EXPECT_EQ(info.communicator, 99);

  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}
#endif  // FUSE_HAVE_MPI

// AbiStructSize.OldCallerFuseStepRequestIsAccepted (Review 2026-10-11 AB-2):
// FirstFrozenStructSize<fuse_step_request> was missing, so check_struct_size
// refused struct_size=16 (the layout shipped before t_begin/t_end/step/
// iteration/requires_checkpoint/requires_restore/reserved were appended in
// 612b153, per c-abi.md's growth table "16 -> 72") with FUSE_ERR_VERSION,
// even though begin_step()/end_step() never read any of the grown members
// (entries.def documents them provider-direction-only). Heap-allocates
// EXACTLY the pre-growth byte count, mirroring
// OldCallerStructSizeNeverReadsMyComponent above, so a read past it is a
// genuine heap-buffer-overflow.
TEST(AbiStructSize, OldCallerFuseStepRequestIsAccepted) {
  constexpr std::size_t kOldSize =
      offsetof(fuse_step_request, component) + sizeof(const char*);
  static_assert(kOldSize < sizeof(fuse_step_request),
               "the provider-direction fields must have grown "
               "fuse_step_request (rule 3)");

  const auto* t = Table();
  ASSERT_NE(t, nullptr);

  fuse_context_out co{.struct_size = sizeof(co)};
  ASSERT_EQ(t->create(nullptr, nullptr, &co), FUSE_OK);
  fuse_ctx* ctx = co.ctx;
  const auto bytes = fuse::abi_test::build_two_component_lag1_bytes();
  fuse_graph_source src{.struct_size = sizeof(src)};
  src.bytes = bytes.data();
  src.byte_count = bytes.size();
  ASSERT_EQ(t->load_graph(ctx, &src, nullptr), FUSE_OK);

  double up_storage[4] = {1, 2, 3, 4};
  double down_storage[4] = {0, 0, 0, 0};
  fuse_field_name fname{.struct_size = sizeof(fname)};
  fname.component = "upstream";
  fname.field = "out";
  fuse_field_ref fref{.struct_size = sizeof(fref)};
  ASSERT_EQ(t->field_handle(ctx, &fname, &fref), FUSE_OK);
  fuse_field_view fview{.struct_size = sizeof(fview)};
  fview.handle = fref.handle;
  fview.element_type = FUSE_ELEMENT_F64;
  fview.memspace = FUSE_MEMSPACE_HOST;
  fview.ownership = FUSE_OWNERSHIP_OWNED;
  fview.components = 1;
  fview.count = 4;
  fview.owned_count = 4;
  fview.stride = 1;
  fview.data = up_storage;
  ASSERT_EQ(t->register_field(ctx, &fview, nullptr), FUSE_OK);

  fname.component = "downstream";
  fname.field = "in";
  ASSERT_EQ(t->field_handle(ctx, &fname, &fref), FUSE_OK);
  fview.handle = fref.handle;
  fview.data = down_storage;
  ASSERT_EQ(t->register_field(ctx, &fview, nullptr), FUSE_OK);

  ASSERT_EQ(t->initialize(ctx, nullptr, nullptr), FUSE_OK);

  auto buf = std::make_unique<std::byte[]>(kOldSize);
  auto* old_req = reinterpret_cast<fuse_step_request*>(buf.get());
  old_req->struct_size = static_cast<std::uint32_t>(kOldSize);
  old_req->component   = "upstream";

  fuse_step_report rep{.struct_size = sizeof(rep)};
  EXPECT_EQ(t->begin_step(ctx, old_req, &rep), FUSE_OK);
  EXPECT_EQ(rep.step, 0);

  old_req->component = "upstream";  // end_step's own struct_size check
  fuse_step_report end_rep{.struct_size = sizeof(end_rep)};
  EXPECT_EQ(t->end_step(ctx, old_req, &end_rep), FUSE_OK);

  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}

// The write-side mirror of OldCallerStructSizeNeverReadsPastItsOwnAllocation
// (docs/dev/abi-design.md rule 2): a caller compiled before next_component
// was appended to fuse_step_report (2026-10-03) hands begin_step() a report
// whose struct_size stops at t_end. begin_step() must not write
// next_component (or any later member) into it. The report is placed so that
// its first byte past the old size is the first byte of a PROT_NONE page: the
// correct, struct_size_covers()-gated code returns normally; code that writes
// next_component unconditionally (build-evidence/mutations/abi.toml,
// abi-begin-step-writes-next-component-past-an-old-callers-struct-size)
// faults on every run, instead of corrupting a heap allocation that may or
// may not crash at teardown (lead, 2026-10-19: that row survived the gate in a
// clean tree because the plain-build heap overflow it relied on is
// allocator-dependent).
TEST(AbiStructSize, OldCallerStepReportIsNeverWrittenPastItsOwnStructSize) {
  constexpr std::size_t kOldSize = offsetof(fuse_step_report, next_component);
  static_assert(kOldSize == offsetof(fuse_step_report, t_end) + sizeof(fuse_exact_time),
                "next_component must directly follow t_end (rule 3)");
  static_assert(kOldSize < sizeof(fuse_step_report),
                "next_component and its successors must have grown fuse_step_report");

  const long page = ::sysconf(_SC_PAGESIZE);
  ASSERT_GT(page, 0);
  ASSERT_LE(static_cast<long>(kOldSize), page);

  void* region = ::mmap(nullptr, static_cast<std::size_t>(2) * static_cast<std::size_t>(page),
                        PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(region, MAP_FAILED);
  std::byte* page0 = reinterpret_cast<std::byte*>(region);
  std::byte* page1 = page0 + page;
  ASSERT_EQ(::mprotect(page1, static_cast<std::size_t>(page), PROT_NONE), 0);
  auto* old_rep = reinterpret_cast<fuse_step_report*>(page1 - kOldSize);
  old_rep->struct_size = static_cast<std::uint32_t>(kOldSize);

  const auto* t = Table();
  ASSERT_NE(t, nullptr);

  fuse_context_out co{.struct_size = sizeof(co)};
  ASSERT_EQ(t->create(nullptr, nullptr, &co), FUSE_OK);
  fuse_ctx* ctx = co.ctx;
  const auto bytes = fuse::abi_test::build_two_component_lag1_bytes();
  fuse_graph_source src{.struct_size = sizeof(src)};
  src.bytes = bytes.data();
  src.byte_count = bytes.size();
  ASSERT_EQ(t->load_graph(ctx, &src, nullptr), FUSE_OK);

  double up_storage[4] = {1, 2, 3, 4};
  double down_storage[4] = {0, 0, 0, 0};
  fuse_field_name fname{.struct_size = sizeof(fname)};
  fname.component = "upstream";
  fname.field = "out";
  fuse_field_ref fref{.struct_size = sizeof(fref)};
  ASSERT_EQ(t->field_handle(ctx, &fname, &fref), FUSE_OK);
  fuse_field_view fview{.struct_size = sizeof(fview)};
  fview.handle = fref.handle;
  fview.element_type = FUSE_ELEMENT_F64;
  fview.memspace = FUSE_MEMSPACE_HOST;
  fview.ownership = FUSE_OWNERSHIP_OWNED;
  fview.components = 1;
  fview.count = 4;
  fview.owned_count = 4;
  fview.stride = 1;
  fview.data = up_storage;
  ASSERT_EQ(t->register_field(ctx, &fview, nullptr), FUSE_OK);
  fname.component = "downstream";
  fname.field = "in";
  ASSERT_EQ(t->field_handle(ctx, &fname, &fref), FUSE_OK);
  fview.handle = fref.handle;
  fview.data = down_storage;
  ASSERT_EQ(t->register_field(ctx, &fview, nullptr), FUSE_OK);
  ASSERT_EQ(t->initialize(ctx, nullptr, nullptr), FUSE_OK);

  fuse_step_request req{.struct_size = sizeof(req)};
  req.component = "upstream";
  EXPECT_EQ(t->begin_step(ctx, &req, old_rep), FUSE_OK);
  EXPECT_EQ(old_rep->step, 0);

  fuse_step_report end_rep{.struct_size = sizeof(end_rep)};
  EXPECT_EQ(t->end_step(ctx, &req, &end_rep), FUSE_OK);
  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
  ::munmap(region, static_cast<std::size_t>(2) * static_cast<std::size_t>(page));
}
