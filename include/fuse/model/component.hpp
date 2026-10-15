#pragma once
#include <utility>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <algorithm>
#include <limits>

#include <string>
#include <vector>

#include "fuse/base/error.hpp"
#include "fuse/base/ids.hpp"
#include "fuse/base/rational.hpp"
#include "fuse/model/launch.hpp"

namespace fuse {

/// Contiguous range of world-communicator ranks owned by a component.
/// Disjoint rank sets are the normal case (spec Section 4.1); overlap is legal.
/// Which ranks a component runs on.
///
/// An explicit sorted list of distinct ranks, not a range. Three layouts that
/// real coupled jobs use and a range cannot express:
///
///   * a component on strided ranks (every other core, one per socket);
///   * a component on a subset chosen by the launcher rather than a block;
///   * two components sharing ranks, which is how a job fits both in memory
///     when neither needs a whole partition to itself.
///
/// The last one also means `overlaps()` is a question with a useful answer
/// rather than an error condition: a channel between two components that share
/// some ranks moves data between the ranks they do not share and copies on the
/// ranks they do.
///
/// `{first, count}` still constructs a contiguous set, because that is the
/// common case and because it keeps every existing declaration reading the
/// same way. The members are gone, though, so code that used to compute
/// `first + count` has to say what it means.
class RankSet {
 public:
  // REVIEW.md R-17 was investigated and REJECTED here: see the review
  // report for the git-history evidence (ComponentDecl.DefaultsToUnitInterval
  // OnRankZero, test/model/component_test.cpp, predates this class and pins
  // rank 0 as the deliberate default of the pre-refactor `struct RankSet
  // { int first = 0; int count = 1; }` this class replaced). "Empty =
  // unplaced" is RankSet(int, int)'s documented contract for an EXPLICIT
  // count <= 0, not a claim about the niladic default, which has always
  // meant "rank 0" and is tested as such.
  RankSet() : ranks_{0} {}

  /// The contiguous set `[first, first + count)`. `count <= 0` yields an empty
  /// set, which is legal: a component may be declared and placed nowhere in a
  /// particular run.
  ///
  /// LEGAL ONLY FOR A COMPONENT WITH NO CHANNELS, and `validate()` enforces
  /// that with `E-PLACEMENT-EMPTY`. A channel out of or into a component that
  /// exists on no rank asks to move data to or from nowhere, which is a
  /// modelling error rather than a configuration to be accommodated: there is
  /// no rank to read from, and no rank that could receive. So "placed nowhere"
  /// means "switched off this run", and switching a component off means its
  /// channels go with it.
  ///
  /// This sentence used to stop at "legal", which left the model layer calling
  /// a configuration legal while `MpiTransport::build_plan` refused it
  /// (`E-TRANSPORT-RANKS`) two layers down -- the ruling of 2026-09-22 is that
  /// the refusal is right and this contract was the incomplete half.
  RankSet(int first, int count) {
    if (count <= 0) return;  // empty, as documented above.
    // BM-14 (review 2026-10-11): `first` and `count` arrive unchecked from
    // any caller, including a hand-written YAML document (src/model/
    // yaml_io.cpp) that has not yet reached any other validation gate. A
    // huge `count` would `reserve()` gigabytes before anything looks at
    // it, and `first + count` overflowing `int` is undefined behavior in
    // the loop below -- refused here, before either happens, the same
    // "refuse rather than let the arithmetic go wrong" rule
    // `detail::narrow_exponent()` (fuse/base/units.hpp) and `Rational`'s
    // own overflow checks already follow. 16 million ranks is already
    // far more than any placement a real coupled run has ever declared.
    constexpr int kMaxRankSetCount = 1 << 24;
    if (count > kMaxRankSetCount) {
      FUSE_REFUSE("E-RANKSET-RANGE",
                 "RankSet(first={}, count={}): count exceeds the sane bound {}",
                 first, count, kMaxRankSetCount);
    }
    const std::int64_t last =
        static_cast<std::int64_t>(first) + static_cast<std::int64_t>(count) - 1;
    if (last > std::numeric_limits<int>::max()) {
      FUSE_REFUSE("E-RANKSET-RANGE",
                 "RankSet(first={}, count={}): first + count overflows int",
                 first, count);
    }
    ranks_.reserve(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) ranks_.push_back(first + i);
  }

  /// An arbitrary set. Sorted and de-duplicated on construction, because every
  /// consumer depends on the order (`CommGraph` matches by position, the
  /// descriptor is compared byte-for-byte across ranks, and the agreement
  /// reduction derives its schedule from a rank's index in the list), and
  /// because a caller handing over ranks in launcher order should not have to
  /// know that.
  explicit RankSet(std::vector<int> ranks) : ranks_(std::move(ranks)) {
    std::sort(ranks_.begin(), ranks_.end());
    ranks_.erase(std::unique(ranks_.begin(), ranks_.end()), ranks_.end());
  }

  [[nodiscard]] const std::vector<int>& ranks() const noexcept { return ranks_; }
  [[nodiscard]] std::size_t size() const noexcept { return ranks_.size(); }
  [[nodiscard]] bool empty() const noexcept { return ranks_.empty(); }
  [[nodiscard]] auto begin() const noexcept { return ranks_.begin(); }
  [[nodiscard]] auto end() const noexcept { return ranks_.end(); }

  /// The smallest rank, or -1 when empty. Named for what it is: callers that
  /// want "where the block starts" are usually about to assume contiguity.
  [[nodiscard]] int min_rank() const noexcept {
    return ranks_.empty() ? -1 : ranks_.front();
  }
  [[nodiscard]] int max_rank() const noexcept {
    return ranks_.empty() ? -1 : ranks_.back();
  }

  [[nodiscard]] bool contains(int rank) const noexcept {
    return std::binary_search(ranks_.begin(), ranks_.end(), rank);
  }

  /// Whether the two sets share at least one rank.
  [[nodiscard]] bool overlaps(const RankSet& o) const noexcept {
    auto a = ranks_.begin();
    auto b = o.ranks_.begin();
    while (a != ranks_.end() && b != o.ranks_.end()) {
      if (*a == *b) return true;
      if (*a < *b) ++a; else ++b;
    }
    return false;
  }

  /// Whether the two sets are the same ranks. Distinct from `overlaps`: a
  /// channel between identical placements needs no data move at all, while one
  /// between partially shared placements needs both a move and a local copy.
  [[nodiscard]] bool same_as(const RankSet& o) const noexcept {
    return ranks_ == o.ranks_;
  }

  /// Whether this set is a block `[min_rank, max_rank]` with nothing missing.
  /// Asked by code that can take a faster path on a contiguous layout; never
  /// assumed.
  [[nodiscard]] bool is_contiguous() const noexcept {
    if (ranks_.empty()) return true;
    return ranks_.back() - ranks_.front() + 1 == static_cast<int>(ranks_.size());
  }

  /// The position of `rank` in this set, or -1. This, not the rank number, is
  /// what indexes per-rank data in a set that may be strided.
  [[nodiscard]] int index_of(int rank) const noexcept {
    const auto it = std::lower_bound(ranks_.begin(), ranks_.end(), rank);
    if (it == ranks_.end() || *it != rank) return -1;
    return static_cast<int>(it - ranks_.begin());
  }

  [[nodiscard]] bool operator==(const RankSet& o) const noexcept {
    return ranks_ == o.ranks_;
  }

 private:
  std::vector<int> ranks_;
};

/// The union of two placements, sorted: the ranks that take part in an exchange
/// between them.
[[nodiscard]] inline RankSet rank_union(const RankSet& a, const RankSet& b) {
  std::vector<int> all(a.ranks());
  all.insert(all.end(), b.ranks().begin(), b.ranks().end());
  return RankSet{std::move(all)};
}

/// The ranks the two placements share.
[[nodiscard]] inline RankSet rank_intersection(const RankSet& a, const RankSet& b) {
  std::vector<int> both;
  std::set_intersection(a.ranks().begin(), a.ranks().end(),
                        b.ranks().begin(), b.ranks().end(),
                        std::back_inserter(both));
  return RankSet{std::move(both)};
}

/// Which of two things a placement-space entry is.
///
/// The IR's placement space is the component index (`ValueType::rank_set`,
/// `ComponentId` everywhere in the Runner, `placed_view(MeshId, ComponentId)`),
/// so a separately-placed coupler is represented as another entry in the same
/// array rather than a new kind of id. `kind` is what keeps that decision
/// honest: every loop that means "a physics component with state and a
/// clock" -- advancing it, building it a `Clock`, matching it to a supplied
/// `Component`, counting it as a rollback participant -- must ask `kind`
/// rather than assuming every entry qualifies. See docs/dev/coupler-
/// placement.md for the enumerated call sites this decision touches.
enum class ComponentKind { Physics, Coupler };

[[nodiscard]] constexpr const char* to_cstring(ComponentKind k) noexcept {
  switch (k) {
    case ComponentKind::Physics: return "Physics";
    case ComponentKind::Coupler: return "Coupler";
  }
  return "Unknown";
}

/// Spec Section 8.1: `interval` is the component's own coupling interval.
/// Independent intervals are the default; lockstep is all intervals equal.
struct ComponentDecl {
  std::string           name;
  RankSet               ranks;
  Rational              interval{1};
  std::vector<FieldId>  fields;
  /// `Coupler` entries own no fields (`validate()`'s `E-COUPLER-FIELDS`),
  /// have no clock, and are never advanced -- see the enumerated call sites
  /// in docs/dev/coupler-placement.md. Default `Physics` so every existing
  /// `ComponentDecl{...}` aggregate-initializer keeps meaning what it meant
  /// before this field existed.
  ComponentKind          kind = ComponentKind::Physics;
  // Deliberately NOT serialized by either the flat descriptor
  // (include/fuse/model/descriptor.hpp) or the YAML front-end
  // (src/graph/yaml_io.cpp). Nothing in the current graph API populates
  // this field -- there is no add_region_to_component()-style call -- so
  // Task 1.5-1.7 leaves it as a placeholder for whatever Phase 2 introduces
  // to populate it, at which point serialization support should be added
  // alongside the writer. Pinned by
  // Descriptor.RegionsFieldIsIntentionallyNotSerialized so a later
  // populate-without-serialize regression is caught rather than silently
  // passing through structurally_equal() (which is defined in terms of
  // serialize(), so it cannot see a field neither side writes).
  std::vector<RegionId> regions;

  /// How THIS component's executable finds the rest of the graph when it is
  /// not run in-process with everyone else (fuse/model/launch.hpp). Default
  /// `InProcess` keeps every existing `ComponentDecl{...}` aggregate
  /// initializer and every descriptor written before S6's interfaces track
  /// meaning what it meant before this field existed (descriptor v9, see
  /// descriptor.hpp's kGraphVersion history).
  LaunchSpec launch;
};

/// `g.add_coupler(CouplerDecl{"C", ranks})`: a placement with no fields and
/// no clock, that a channel's `RemapSpec::remap_on` (channel.hpp) can name.
/// Lowers to a `ComponentDecl` with `kind = ComponentKind::Coupler` (see
/// `CouplingGraph::add_coupler`) -- there is no separate coupler array, so
/// every existing `component_count()`/`component(ComponentId)` reader keeps
/// working unchanged, and `kind` is what a caller that needs to tell the two
/// apart asks.
struct CouplerDecl {
  std::string name;
  RankSet     ranks;

  /// How this coupler's own ranks are launched, when it runs as its own
  /// program rather than inside a component's executable. A coupler owns
  /// no fields and is never advanced, but it still takes part in every
  /// collective of the base communicator -- including the one `connect()`
  /// itself makes -- so a coupler launched as its own MPMD segment must
  /// say so here, exactly as a component does. Left at its default
  /// (`LaunchMode::InProcess`) a coupler shares whichever executable its
  /// ranks already belong to, which is the accept-mode arrangement.
  /// Measured consequence of not carrying this, 2026-10-08: the
  /// Brusselator's MPMD rows hung with no output at all, U and V waiting
  /// in `connect_mpmd()`'s collective for a coupler rank that never
  /// called `connect()`.
  LaunchSpec  launch;
};

/// The time span an accumulation or space-time projection covers.
struct Window {
  enum class Kind { CouplingPeriod, FromComponent, Explicit };

  Kind        kind = Kind::CouplingPeriod;
  ComponentId component;   // valid only when kind == FromComponent
  Rational    length{0};   // valid only when kind == Explicit

  [[nodiscard]] static Window coupling_period() { return Window{}; }
  [[nodiscard]] static Window from(ComponentId c) {
    return Window{Kind::FromComponent, c, Rational{0}};
  }
  [[nodiscard]] static Window explicit_length(Rational len) {
    return Window{Kind::Explicit, ComponentId{}, len};
  }
};

}  // namespace fuse
