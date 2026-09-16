#include <gtest/gtest.h>
#include <fuse/field/registry.hpp>
#include <fuse/base/error.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

using fuse::FieldId;
using fuse::FieldRef;
using fuse::FieldRegistry;
using fuse::Layout;
using fuse::MemorySpace;

TEST(FieldRef, DefaultIsInvalid) {
  const FieldRef f;
  EXPECT_FALSE(f.valid());
  EXPECT_EQ(f.data(), nullptr);
}

TEST(FieldRef, WrapsContiguousScalarData) {
  std::vector<double> v{1.0, 2.0, 3.0, 4.0};
  const FieldRef f{v.data(), Layout{.stride = 1, .count = 4}, MemorySpace::Host};
  ASSERT_TRUE(f.valid());
  EXPECT_EQ(f.size(), 4u);
  EXPECT_DOUBLE_EQ(f.at(2), 3.0);
}

TEST(FieldRef, WritesThroughToOwnerStorage) {
  std::vector<double> v(4, 0.0);
  FieldRef f{v.data(), Layout{.stride = 1, .count = 4}, MemorySpace::Host};
  f.at(1) = 7.5;
  EXPECT_DOUBLE_EQ(v[1], 7.5);   // the component sees the write
}

TEST(FieldRef, InterleavedVectorLayout) {
  // x0 y0 z0 x1 y1 z1
  std::vector<double> v{1, 2, 3, 4, 5, 6};
  const FieldRef f{v.data(),
                   Layout{.stride = 3, .component_stride = 1, .count = 2,
                          .components = 3},
                   MemorySpace::Host};
  EXPECT_DOUBLE_EQ(f.at(0, 0), 1.0);
  EXPECT_DOUBLE_EQ(f.at(0, 2), 3.0);
  EXPECT_DOUBLE_EQ(f.at(1, 0), 4.0);
  EXPECT_DOUBLE_EQ(f.at(1, 2), 6.0);
}

TEST(FieldRef, BlockedVectorLayout) {
  // x0 x1 | y0 y1 | z0 z1
  std::vector<double> v{1, 4, 2, 5, 3, 6};
  const FieldRef f{v.data(),
                   Layout{.stride = 1, .component_stride = 2, .count = 2,
                          .components = 3},
                   MemorySpace::Host};
  EXPECT_DOUBLE_EQ(f.at(0, 0), 1.0);
  EXPECT_DOUBLE_EQ(f.at(0, 2), 3.0);
  EXPECT_DOUBLE_EQ(f.at(1, 0), 4.0);
  EXPECT_DOUBLE_EQ(f.at(1, 2), 6.0);
}

TEST(FieldRegistry, BindsAndRetrieves) {
  std::vector<double> v(8, 1.0);
  FieldRegistry reg;
  const auto r = reg.bind(FieldId{0},
                          FieldRef{v.data(), Layout{.stride = 1, .count = 8},
                                   MemorySpace::Host});
  ASSERT_TRUE(r.ok());
  EXPECT_TRUE(reg.is_bound(FieldId{0}));
  EXPECT_EQ(reg.bound_count(), 1u);

  const auto got = reg.get(FieldId{0});
  ASSERT_TRUE(got.ok());
  EXPECT_EQ(got.value().data(), v.data());
}

TEST(FieldRegistry, RejectsRebinding) {
  std::vector<double> a(4, 0.0), b(4, 0.0);
  FieldRegistry reg;
  ASSERT_TRUE(reg.bind(FieldId{0}, FieldRef{a.data(),
                                            Layout{.stride = 1, .count = 4},
                                            MemorySpace::Host}).ok());
  const auto second = reg.bind(FieldId{0},
                               FieldRef{b.data(),
                                        Layout{.stride = 1, .count = 4},
                                        MemorySpace::Host});
  ASSERT_FALSE(second.ok());
  EXPECT_EQ(second.error().code, "E-FIELD-REBIND");
}

TEST(FieldRegistry, RejectsNullData) {
  FieldRegistry reg;
  const auto r = reg.bind(FieldId{0}, FieldRef{nullptr,
                                               Layout{.stride = 1, .count = 4},
                                               MemorySpace::Host});
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-FIELD-NULL");
}

// The complement of RejectsNullData above, and the reason its premise is
// "null with something to point at" rather than "null". A rank that owns no
// elements of a region is ordinary on a partitioned mesh (spec §4.1), and the
// empty vector it hands over has a null data(). Refusing that would make a
// component unbindable on exactly the ranks the MPI transport exists to
// serve.
TEST(FieldRegistry, AcceptsANullPointerForAnEmptyField) {
  const std::vector<double> empty;
  ASSERT_EQ(empty.data(), nullptr)
      << "this platform's empty vector::data() is non-null, so this test no "
         "longer exercises the case it was written for";

  FieldRegistry reg;
  const auto r = reg.bind(FieldId{0}, FieldRef{nullptr,
                                               Layout{.stride = 1, .count = 0},
                                               MemorySpace::Host});
  ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
  EXPECT_TRUE(reg.is_bound(FieldId{0}));

  const auto got = reg.get(FieldId{0});
  ASSERT_TRUE(got.ok());
  EXPECT_EQ(got.value().data(), nullptr);
  EXPECT_EQ(got.value().size(), 0u);
}

// Zero elements, several components: still nothing to address.
TEST(FieldRegistry, AcceptsANullPointerForAnEmptyVectorField) {
  FieldRegistry reg;
  const auto r = reg.bind(
      FieldId{1}, FieldRef{nullptr,
                           Layout{.stride = 3, .count = 0, .components = 3},
                           MemorySpace::Host});
  ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
}

// One element is not zero elements.
TEST(FieldRegistry, StillRejectsNullForASingleElement) {
  FieldRegistry reg;
  const auto r = reg.bind(FieldId{0}, FieldRef{nullptr,
                                               Layout{.stride = 1, .count = 1},
                                               MemorySpace::Host});
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-FIELD-NULL");
}

TEST(FieldRegistry, GetUnboundFieldIsAnError) {
  const FieldRegistry reg;
  const auto r = reg.get(FieldId{3});
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-FIELD-UNBOUND");
}

TEST(FieldRegistry, ReportsWhichRequiredFieldsAreMissing) {
  std::vector<double> v(4, 0.0);
  FieldRegistry reg;
  ASSERT_TRUE(reg.bind(FieldId{0}, FieldRef{v.data(),
                                            Layout{.stride = 1, .count = 4},
                                            MemorySpace::Host}).ok());
  const auto missing = reg.unbound({FieldId{0}, FieldId{1}, FieldId{2}});
  ASSERT_EQ(missing.size(), 2u);
  EXPECT_EQ(missing[0], FieldId{1});
  EXPECT_EQ(missing[1], FieldId{2});
}

// --------------------------------------------------------------------------
// Additions beyond the brief. Every check below must hold in Release
// (-O3 -DNDEBUG) as well as Debug, so none of them may rest on assert().
// --------------------------------------------------------------------------

TEST(FieldRef, IsANonOwningTriviallyCopyableView) {
  // Compile-time proof of spec 4.2: no ownership, no allocation, no
  // user-provided copy/destroy semantics.
  static_assert(std::is_trivially_copyable_v<FieldRef>);
  static_assert(std::is_trivially_destructible_v<FieldRef>);
  static_assert(std::is_trivially_copyable_v<Layout>);
  SUCCEED();
}

TEST(FieldRef, CopyingAViewDoesNotCopyTheData) {
  std::vector<double> v(4, 0.0);
  FieldRef a{v.data(), Layout{.stride = 1, .count = 4}, MemorySpace::Host};
  FieldRef b = a;               // NOLINT: copy is the point of the test
  b.at(3) = 2.5;
  EXPECT_EQ(a.data(), b.data());
  EXPECT_DOUBLE_EQ(v[3], 2.5);  // the write landed in the owner's storage
  EXPECT_DOUBLE_EQ(a.at(3), 2.5);
}

TEST(FieldRef, RecordsMemorySpace) {
  std::vector<double> v(2, 0.0);
  const FieldRef host{v.data(), Layout{.stride = 1, .count = 2},
                      MemorySpace::Host};
  const FieldRef dev{v.data(), Layout{.stride = 1, .count = 2},
                     MemorySpace::Device};
  EXPECT_EQ(host.space(), MemorySpace::Host);
  EXPECT_EQ(dev.space(), MemorySpace::Device);
  EXPECT_STREQ(fuse::to_cstring(MemorySpace::Host), "Host");
  EXPECT_STREQ(fuse::to_cstring(MemorySpace::Device), "Device");
}

TEST(FieldRef, LayoutIsRetrievable) {
  std::vector<double> v(6, 0.0);
  const FieldRef f{v.data(),
                   Layout{.stride = 3, .component_stride = 1, .count = 2,
                          .components = 3},
                   MemorySpace::Host};
  EXPECT_EQ(f.layout().stride, 3u);
  EXPECT_EQ(f.layout().component_stride, 1u);
  EXPECT_EQ(f.layout().count, 2u);
  EXPECT_EQ(f.layout().components, 3);
  EXPECT_EQ(f.layout().effective_component_stride(), 1u);
}

TEST(FieldRef, ZeroComponentStrideMeansInterleaved) {
  const Layout blocked{.stride = 1, .component_stride = 4, .count = 4};
  const Layout interleaved{.stride = 1, .component_stride = 0, .count = 4};
  EXPECT_EQ(blocked.effective_component_stride(), 4u);
  EXPECT_EQ(interleaved.effective_component_stride(), 1u);
}

// at() is bounds-checked unconditionally: an out-of-range index under
// -O3 -DNDEBUG would otherwise be a silent heap overflow.
TEST(FieldRef, AtThrowsOnOutOfRangeElement) {
  std::vector<double> v(4, 0.0);
  FieldRef f{v.data(), Layout{.stride = 1, .count = 4}, MemorySpace::Host};
  EXPECT_THROW((void)f.at(4), fuse::InvalidArgument);
  EXPECT_THROW((void)f.at(9999), fuse::InvalidArgument);
  const FieldRef cf = f;
  EXPECT_THROW((void)cf.at(4), fuse::InvalidArgument);
  EXPECT_NO_THROW((void)f.at(3));
}

TEST(FieldRef, AtThrowsOnOutOfRangeComponent) {
  std::vector<double> v(6, 0.0);
  FieldRef f{v.data(),
             Layout{.stride = 3, .component_stride = 1, .count = 2,
                    .components = 3},
             MemorySpace::Host};
  EXPECT_THROW((void)f.at(0, 3), fuse::InvalidArgument);
  EXPECT_THROW((void)f.at(0, -1), fuse::InvalidArgument);  // would wrap to a huge offset
  EXPECT_NO_THROW((void)f.at(1, 2));
}

TEST(FieldRef, AtOnAnUnboundViewThrows) {
  const FieldRef f;
  EXPECT_THROW((void)f.at(0), fuse::InvalidArgument);
}

TEST(FieldRef, OffsetMatchesAtForEveryValidIndex) {
  std::vector<double> v{1, 4, 2, 5, 3, 6};
  const Layout blocked{.stride = 1, .component_stride = 2, .count = 2,
                       .components = 3};
  const FieldRef f{v.data(), blocked, MemorySpace::Host};
  for (std::size_t i = 0; i < blocked.count; ++i) {
    for (int c = 0; c < blocked.components; ++c) {
      EXPECT_DOUBLE_EQ(f.at(i, c), f.data()[f.offset(i, c)]);
    }
  }
}

TEST(Layout, WellFormedAcceptsScalarInterleavedAndBlocked) {
  EXPECT_TRUE((Layout{.stride = 1, .count = 4}.well_formed()));
  EXPECT_TRUE((Layout{.stride = 3, .component_stride = 1, .count = 2,
                      .components = 3}
                   .well_formed()));
  EXPECT_TRUE((Layout{.stride = 1, .component_stride = 2, .count = 2,
                      .components = 3}
                   .well_formed()));
  // A view into a wider array: gaps are fine, overlaps are not.
  EXPECT_TRUE((Layout{.stride = 8, .component_stride = 1, .count = 3,
                      .components = 3}
                   .well_formed()));
  // Empty on this rank.
  EXPECT_TRUE((Layout{.stride = 1, .count = 0}.well_formed()));
}

TEST(Layout, WellFormedRejectsAliasingAndNonPositiveComponents) {
  // stride < components with interleaved packing: element 1 component 0 and
  // element 0 component 2 would be the same double.
  EXPECT_FALSE((Layout{.stride = 1, .count = 2, .components = 3}.well_formed()));
  // Blocked but the component blocks overlap the element extent.
  EXPECT_FALSE((Layout{.stride = 1, .component_stride = 1, .count = 4,
                       .components = 2}
                   .well_formed()));
  // A zero element stride collapses every element onto one slot.
  EXPECT_FALSE((Layout{.stride = 0, .component_stride = 1, .count = 4,
                       .components = 1}
                   .well_formed()));
  EXPECT_FALSE((Layout{.stride = 1, .count = 4, .components = 0}.well_formed()));
  EXPECT_FALSE((Layout{.stride = 1, .count = 4, .components = -1}.well_formed()));
}

TEST(Layout, WellFormedRejectsOverflowingExtents) {
  constexpr std::size_t kMax = std::numeric_limits<std::size_t>::max();
  EXPECT_FALSE((Layout{.stride = kMax / 2, .component_stride = 1, .count = 8,
                       .components = 1}
                   .well_formed()));
}

TEST(FieldRegistry, RejectsAnInvalidFieldId) {
  std::vector<double> v(4, 0.0);
  FieldRegistry reg;
  const auto r = reg.bind(FieldId{},  // default-constructed == kInvalid
                          FieldRef{v.data(), Layout{.stride = 1, .count = 4},
                                   MemorySpace::Host});
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-FIELD-ID");
  EXPECT_EQ(reg.bound_count(), 0u);
}

TEST(FieldRegistry, RejectsAMalformedLayout) {
  std::vector<double> v(6, 0.0);
  FieldRegistry reg;
  const auto r = reg.bind(
      FieldId{1},
      FieldRef{v.data(), Layout{.stride = 1, .count = 2, .components = 3},
               MemorySpace::Host});
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-FIELD-LAYOUT");
  EXPECT_FALSE(reg.is_bound(FieldId{1}));
}

TEST(FieldRegistry, RejectedBindLeavesTheRegistryUnchanged) {
  std::vector<double> a(4, 0.0), b(4, 0.0);
  FieldRegistry reg;
  ASSERT_TRUE(reg.bind(FieldId{7}, FieldRef{a.data(),
                                            Layout{.stride = 1, .count = 4},
                                            MemorySpace::Host}).ok());
  ASSERT_FALSE(reg.bind(FieldId{7}, FieldRef{b.data(),
                                             Layout{.stride = 1, .count = 4},
                                             MemorySpace::Host}).ok());
  EXPECT_EQ(reg.bound_count(), 1u);
  const auto got = reg.get(FieldId{7});
  ASSERT_TRUE(got.ok());
  EXPECT_EQ(got.value().data(), a.data());  // the first binding still stands
}

// Standing rule: a handle handed out earlier must survive later registrations.
TEST(FieldRegistry, HandleSurvivesManySubsequentBinds) {
  std::vector<double> first{10.0, 11.0, 12.0, 13.0};
  std::vector<std::vector<double>> owners;
  owners.reserve(256);

  FieldRegistry reg;
  ASSERT_TRUE(reg.bind(FieldId{0}, FieldRef{first.data(),
                                            Layout{.stride = 1, .count = 4},
                                            MemorySpace::Host}).ok());
  const auto early = reg.get(FieldId{0});
  ASSERT_TRUE(early.ok());
  const FieldRef held = early.value();  // by value: no reference into the map

  // Force many rehashes of the registry's internal table.
  for (std::uint32_t i = 1; i < 256; ++i) {
    owners.emplace_back(4, static_cast<double>(i));
    ASSERT_TRUE(reg.bind(FieldId{i},
                         FieldRef{owners.back().data(),
                                  Layout{.stride = 1, .count = 4},
                                  MemorySpace::Host})
                    .ok());
  }
  EXPECT_EQ(reg.bound_count(), 256u);

  // The handle taken before 255 more binds still addresses the same storage.
  EXPECT_EQ(held.data(), first.data());
  EXPECT_DOUBLE_EQ(held.at(2), 12.0);

  const auto late = reg.get(FieldId{0});
  ASSERT_TRUE(late.ok());
  EXPECT_EQ(late.value().data(), first.data());
  EXPECT_DOUBLE_EQ(late.value().at(3), 13.0);
}

TEST(FieldRegistry, GetReturnsAnIndependentCopyOfTheView) {
  std::vector<double> v(4, 0.0);
  FieldRegistry reg;
  ASSERT_TRUE(reg.bind(FieldId{2}, FieldRef{v.data(),
                                            Layout{.stride = 1, .count = 4},
                                            MemorySpace::Host}).ok());
  auto got = reg.get(FieldId{2});
  ASSERT_TRUE(got.ok());
  got.value().at(0) = 3.25;   // writes go to the owner, not to a registry copy
  EXPECT_DOUBLE_EQ(v[0], 3.25);

  const auto again = reg.get(FieldId{2});
  ASSERT_TRUE(again.ok());
  EXPECT_DOUBLE_EQ(again.value().at(0), 3.25);
}

TEST(FieldRegistry, UnboundIsEmptyWhenEverythingIsBound) {
  std::vector<double> v(4, 0.0);
  FieldRegistry reg;
  ASSERT_TRUE(reg.bind(FieldId{0}, FieldRef{v.data(),
                                            Layout{.stride = 1, .count = 4},
                                            MemorySpace::Host}).ok());
  ASSERT_TRUE(reg.bind(FieldId{1}, FieldRef{v.data(),
                                            Layout{.stride = 1, .count = 4},
                                            MemorySpace::Host}).ok());
  EXPECT_TRUE(reg.unbound({FieldId{0}, FieldId{1}}).empty());
  EXPECT_TRUE(reg.unbound({}).empty());
}

TEST(FieldRegistry, UnboundPreservesRequestOrderAndDuplicates) {
  FieldRegistry reg;
  const auto missing = reg.unbound({FieldId{5}, FieldId{2}, FieldId{5}});
  ASSERT_EQ(missing.size(), 3u);
  EXPECT_EQ(missing[0], FieldId{5});
  EXPECT_EQ(missing[1], FieldId{2});
  EXPECT_EQ(missing[2], FieldId{5});
}

TEST(FieldRegistry, IsBoundIsFalseForAnInvalidId) {
  EXPECT_FALSE(FieldRegistry{}.is_bound(FieldId{}));
}

TEST(FieldRegistry, ErrorMessagesNameTheField) {
  FieldRegistry reg;
  const auto r = reg.get(FieldId{42});
  ASSERT_FALSE(r.ok());
  EXPECT_NE(r.error().message.find("42"), std::string::npos);
}
