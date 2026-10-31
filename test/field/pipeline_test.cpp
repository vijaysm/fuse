// The lazy field pipeline (usability rules U1, U3, U5, U6).
//
// Every rule here is INVISIBLE IN A PASSING EAGER IMPLEMENTATION: a version
// that copied the data at every stage would satisfy each functional
// expectation below and merely be slower. That is why the laziness test is an
// allocation fence rather than an inspection, and why the lifetime rule is
// pinned by a deleted overload rather than by a comment.

#include "fuse/field/pipeline.hpp"

#include <atomic>
#include <cstdint>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>

#include "fuse/field/field_ref.hpp"
#include <utility>
#include <string>
#include "fuse/model/coupling_graph.hpp"

namespace {

/// Counts Kokkos allocations while it is alive.
///
/// Kokkos routes View allocations through its own allocator, so a global
/// `operator new` counter would miss them; this uses Kokkos' own profiling
/// callback, which is the only thing that sees every View allocation
/// regardless of memory space.
class AllocationFence {
 public:
  AllocationFence() {
    count().store(0);
    Kokkos::Tools::Experimental::set_allocate_data_callback(
        [](const Kokkos_Profiling_SpaceHandle, const char*, const void*,
           const std::uint64_t) { count().fetch_add(1); });
  }
  ~AllocationFence() {
    Kokkos::Tools::Experimental::set_allocate_data_callback(nullptr);
  }
  AllocationFence(const AllocationFence&) = delete;
  AllocationFence& operator=(const AllocationFence&) = delete;

  [[nodiscard]] static std::atomic<int>& count() {
    static std::atomic<int> n{0};
    return n;
  }
  [[nodiscard]] int allocations() const { return count().load(); }
};

/// A registered field over caller-owned storage: 3 components, interleaved,
/// which is the layout a component using a `std::vector<double>` of xyz
/// triples actually has.
struct Interleaved {
  std::vector<double> storage;
  fuse::FieldRef      ref;

  explicit Interleaved(std::size_t n)
      : storage(n * 3),
        ref(storage.data(),
            fuse::Layout{.stride = 3, .component_stride = 1, .count = n,
                         .components = 3},
            fuse::MemorySpace::Host) {
    // value = element * 100 + component, so any mis-sliced value is obvious
    // rather than plausible.
    for (std::size_t i = 0; i < n; ++i) {
      for (int c = 0; c < 3; ++c) {
        storage[i * 3 + static_cast<std::size_t>(c)] =
            static_cast<double>(i) * 100.0 + static_cast<double>(c);
      }
    }
  }
};

}  // namespace

// ---- U3: laziness -------------------------------------------------------

TEST(Pipeline, AFourStageChainOnALargeFieldAllocatesNothing) {
  // Ten million elements. An eager implementation would copy 30 million
  // doubles here, three times over; this must not move one.
  Interleaved f(10'000'000);

  int allocations = 0;
  double value = 0.0;
  {
    const AllocationFence fence;
    const auto chain = fuse::FieldSpan<>::from(f.ref)
                           .owned(9'000'000)
                           .slice({1'000, fuse::end}, fuse::all)
                           .component(2);
    // A terminal, so the chain is not eliminated as dead code: the point is
    // that reaching a value cost no allocation, not that nothing ran.
    value = chain.at(0);
    allocations = fence.allocations();
  }
  EXPECT_EQ(allocations, 0)
      << "the chain allocated " << allocations << " time(s); the stages are "
         "supposed to be strided views over the caller's own storage";
  // element 1000, component 2 -> 1000*100 + 2
  EXPECT_DOUBLE_EQ(value, 100'002.0) << "laziness achieved by doing nothing";
}

TEST(Pipeline, StagesSelectTheValuesTheyClaimTo) {
  // Laziness is worthless if the view is wrong, so the arithmetic is checked
  // independently of the allocation count.
  Interleaved f(10);
  const auto s = fuse::FieldSpan<>::from(f.ref).slice({2, 5}, fuse::all);
  ASSERT_EQ(s.element_count(), 3u);
  ASSERT_EQ(s.component_count(), 3u);
  EXPECT_DOUBLE_EQ(s.at(0, 0), 200.0);
  EXPECT_DOUBLE_EQ(s.at(2, 1), 401.0);

  const auto c = fuse::FieldSpan<>::from(f.ref).component(1);
  ASSERT_EQ(c.element_count(), 10u);
  ASSERT_EQ(c.component_count(), 1u);
  EXPECT_DOUBLE_EQ(c.at(7), 701.0);

  // A chain's stages compose: slice then component, in that order.
  const auto both =
      fuse::FieldSpan<>::from(f.ref).slice({4, fuse::end}, fuse::all).component(2);
  ASSERT_EQ(both.element_count(), 6u);
  EXPECT_DOUBLE_EQ(both.at(0), 402.0);
}

// ---- U5: the spelling, and what a bad extent does -----------------------

TEST(Pipeline, ExtentSpellingsMeanWhatMdspanMeans) {
  Interleaved f(10);
  const auto whole = fuse::FieldSpan<>::from(f.ref);
  EXPECT_EQ(whole.slice(fuse::all, fuse::all).element_count(), 10u);
  EXPECT_EQ(whole.slice({3, fuse::end}, fuse::all).element_count(), 7u)
      << "{3, end} is half-open from 3 to the last element inclusive";
  EXPECT_EQ(whole.slice({3, 5}, fuse::all).element_count(), 2u)
      << "{3, 5} is half-open: two elements, not three";
  EXPECT_EQ(whole.slice(7, fuse::all).element_count(), 1u)
      << "a bare index is the one-element range, as submdspan(m, 7, all) is";

  // `{lo, end}` resolves against the CURRENT extent, so it stays correct
  // after an earlier stage has narrowed the dimension.
  EXPECT_EQ(whole.slice({0, 6}, fuse::all).slice({2, fuse::end}, fuse::all)
                .element_count(),
            4u);
}

TEST(Pipeline, AReversedExtentIsRefusedWithItsBounds) {
  Interleaved f(10);
  const auto whole = fuse::FieldSpan<>::from(f.ref);
  try {
    (void)whole.slice({7, 3}, fuse::all);
    FAIL() << "a reversed extent was accepted";
  } catch (const fuse::Refusal& e) {
    EXPECT_EQ(std::string(e.code()), "E-FIELD-EXTENT-REVERSED");
    const std::string m = e.what();
    EXPECT_NE(m.find("[7, 3)"), std::string::npos)
        << "the offending bounds are not in the message: " << m;
  }
}

TEST(Pipeline, AnOutOfRangeExtentIsRefusedRatherThanClamped) {
  Interleaved f(10);
  const auto whole = fuse::FieldSpan<>::from(f.ref);
  try {
    (void)whole.slice({8, 25}, fuse::all);
    FAIL() << "clamping turns a caller's wrong arithmetic into a shorter loop "
              "that still produces numbers";
  } catch (const fuse::Refusal& e) {
    EXPECT_EQ(std::string(e.code()), "E-FIELD-EXTENT-RANGE");
    const std::string m = e.what();
    EXPECT_NE(m.find("[8, 25)"), std::string::npos) << m;
    EXPECT_NE(m.find("10"), std::string::npos)
        << "the available count is not in the message: " << m;
  }
  // The component dimension is checked the same way, not just the element one.
  EXPECT_THROW((void)whole.slice(fuse::all, {0, 4}), fuse::Refusal);
}

TEST(Pipeline, AnUnboundFieldIsRefusedNotDereferenced) {
  // A default-constructed `Layout{}` has `count == 0`, which
  // is bit-for-bit the same state as a legitimately-bound empty partition
  // (see AnEmptyFieldFromABoundZeroCountFieldRefDoesNotRefuse below) -- so it
  // cannot be used to pin "genuinely unbound" here. A non-empty null FieldRef
  // is the case `FieldRegistry::bind()` itself still rejects unconditionally.
  const fuse::FieldRef unbound(
      nullptr, fuse::Layout{.stride = 1, .count = 4}, fuse::MemorySpace::Host);
  EXPECT_THROW((void)fuse::FieldSpan<>::from(unbound), fuse::Refusal);
}

// The complement of the above: a rank that legitimately owns
// zero elements of a mesh entity set binds a null pointer with `count == 0`
// (FieldRegistry::bind()'s own allowance, registry.hpp:38-46), and
// FieldSpan::from() must accept it as an ordinary, empty view rather than
// rejecting it -- rejecting it here is an uncaught throw on every rank with a
// genuinely empty partition, on the ordinary path of a partitioned run.
TEST(Pipeline, AnEmptyFieldFromABoundZeroCountFieldRefDoesNotRefuse) {
  const fuse::FieldRef empty(
      nullptr, fuse::Layout{.stride = 1, .count = 0, .components = 3},
      fuse::MemorySpace::Host);
  const auto span = fuse::FieldSpan<>::from(empty);
  EXPECT_EQ(span.element_count(), 0u);
  EXPECT_EQ(span.component_count(), 3u);
  // slice/local_sum/fill on the empty result must be no-ops, not throws.
  EXPECT_NO_THROW(span.fill(1.0));
  EXPECT_DOUBLE_EQ(span.local_sum(), 0.0);
  EXPECT_NO_THROW((void)span.slice(fuse::all, fuse::all));
}

// ---- U6: lifetime -------------------------------------------------------

TEST(Pipeline, AnRvalueChainOwnsItsResultAndOutlivesTheTemporary) {
  // The classic dangling shape: the field is a temporary and the chain's
  // result is kept. `Kokkos::subview` of a MANAGED view is managed and shares
  // the reference count, so this is valid -- the allocation outlives the
  // temporary without a copy. Under the ASan leg this test is the proof.
  auto kept = fuse::OwnedField<>("temp", 100, 3).slice({10, 20}, fuse::all);
  ASSERT_EQ(kept.element_count(), 10u);
  ASSERT_EQ(kept.component_count(), 3u);
  kept.span().fill(2.5);
  EXPECT_DOUBLE_EQ(kept.span().at(9, 2), 2.5)
      << "the buffer behind an rvalue chain did not survive";
}

TEST(Pipeline, AnRvalueChainSharesStorageRatherThanCopyingIt) {
  // Owning must not mean copying: the subview keeps a reference, so taking one
  // allocates nothing.
  int allocations = 0;
  {
    fuse::OwnedField<> f("big", 1'000'000, 3);
    const AllocationFence fence;                 // after the field is built
    auto part = std::move(f).slice({1, 999'999}, fuse::all);
    allocations = fence.allocations();
    EXPECT_EQ(part.element_count(), 999'998u);
  }
  EXPECT_EQ(allocations, 0) << "an owning chain copied its storage";
}

TEST(Pipeline, AnLvalueChainBorrows) {
  fuse::OwnedField<> f("f", 10, 2);
  f.span().fill(1.25);
  const auto borrowed = f.slice({2, 4}, fuse::all);   // lvalue -> FieldSpan
  static_assert(std::is_same_v<decltype(borrowed), const fuse::FieldSpan<>>,
                "an lvalue chain must borrow, not allocate an owning value");
  EXPECT_DOUBLE_EQ(borrowed.at(0, 0), 1.25);
}

// ---- U1: the two kinds of chaining stay apart ---------------------------

namespace {
template <class T> concept HasPipelineStage = requires(const T& t) {
  t.slice(fuse::all, fuse::all);
};
template <class T> concept HasAuthoringCall = requires(T& t) {
  t.add_component(std::declval<fuse::ComponentDecl>());
};
}  // namespace

TEST(Pipeline, NoTypeIsBothABuilderAndAPipeline) {
  // U1. The two chain kinds read alike and mean opposite things -- a builder
  // call records an intention and validates nothing; a pipeline stage
  // validates immediately and records nothing. A type offering both would make
  // which-one-is-this a question at every call site.
  static_assert(HasPipelineStage<fuse::FieldSpan<>>);
  static_assert(HasAuthoringCall<fuse::CouplingGraph>);
  static_assert(!HasAuthoringCall<fuse::FieldSpan<>>,
                "a pipeline type acquired an authoring method (U1)");
  static_assert(!HasPipelineStage<fuse::CouplingGraph>,
                "the graph builder acquired a pipeline stage (U1)");
  SUCCEED();
}

// ---- Layout::well_formed()'s padded/strided boundary ---------------------
//
// These pin the tightened disjointness bound in field_ref.hpp. They live
// here rather than in test/runtime/field_ref_test.cpp -- Layout is
// header-only and pipeline_test.cpp already includes field_ref.hpp directly.

TEST(Layout, WellFormedAcceptsATightPaddedInterleavedLayout) {
  // offsets: component 0 at {0,5,10,15}, component 1 at {2,7,12,17} -- disjoint,
  // confirmed by brute force. The old bound
  // (stride >= nc*cs = 6) wrongly rejected this.
  EXPECT_TRUE((fuse::Layout{.stride = 5, .component_stride = 2, .count = 4,
                            .components = 3}
                   .well_formed()));
  // One unit tighter than the boundary must still be rejected.
  EXPECT_FALSE((fuse::Layout{.stride = 4, .component_stride = 2, .count = 4,
                             .components = 3}
                    .well_formed()));
}

TEST(Layout, WellFormedAcceptsATightPaddedBlockedLayout) {
  // Symmetric blocked case: component blocks at {0,2,4}, {5,7,9} -- disjoint.
  // The old bound (cs >= count*stride = 6) wrongly rejected this.
  EXPECT_TRUE((fuse::Layout{.stride = 2, .component_stride = 5, .count = 3,
                            .components = 2}
                   .well_formed()));
  // One unit tighter than the boundary must still be rejected.
  EXPECT_FALSE((fuse::Layout{.stride = 2, .component_stride = 4, .count = 3,
                             .components = 2}
                    .well_formed()));
}
