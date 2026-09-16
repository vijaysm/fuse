#pragma once

#include <cstddef>
#include <cstring>
#include <span>
#include <string>
#include <vector>

#include "fuse/base/result.hpp"
#include "fuse/model/coupling_graph.hpp"

namespace fuse {

inline constexpr std::uint32_t kGraphMagic   = 0x46555345u;  // "FUSE"
/// Bumped to 2 by Phase 7 Task 7.5, which added an iteration region's
/// `convergence` criterion and `acceleration` config to the record. A v1
/// buffer read as v2 would take the next region's name length out of the
/// measure count, so the version gate has to reject it rather than let a
/// plausible-looking graph through with no convergence test.
/// Version 3 is the first FUSE descriptor. Two changes from the inherited v2, both
/// forced by the merge and both invisible to a reader that guessed: the magic is
/// "FUSE", and a unit record carries its exact affine part (scale, offset) after
/// the seven dimension exponents. A v2 buffer read as v3 would take the next
/// field's bytes as a scale, so the version gate rejects it rather than admit a
/// plausible-looking graph with the wrong units.
/// 5: a component's ranks are an explicit list, not `{first, count}`. Strided
///    and overlapping placements are expressible; a v4 reader would read the
///    rank count as a first rank and produce a plausible, wrong placement,
///    which is the worst kind of incompatibility, so v4 is refused.
///
/// 4: adds the merge table (how several channels writing one field combine).
///
/// The bump is not optional. A v3 reader handed a v4 buffer would stop after
/// the iteration regions and never see the merge rules, so it would read a
/// graph that says "one writer per field" from bytes that say otherwise -- and
/// it would read it successfully. The version check turns that into a refusal.
inline constexpr std::uint32_t kGraphVersion = 5u;

namespace detail {

class Writer {
 public:
  template <class T>
  void pod(const T& v) {
    static_assert(std::is_trivially_copyable_v<T>);
    const auto* p = reinterpret_cast<const std::byte*>(&v);
    buf_.insert(buf_.end(), p, p + sizeof(T));
  }
  void str(const std::string& s) {
    pod(static_cast<std::uint32_t>(s.size()));
    const auto* p = reinterpret_cast<const std::byte*>(s.data());
    buf_.insert(buf_.end(), p, p + s.size());
  }
  void rational(const Rational& r) {
    pod(r.num());
    pod(r.den());
  }
  void units(const Units& u) {
    pod(u.length); pod(u.mass); pod(u.time); pod(u.current);
    pod(u.temperature); pod(u.amount); pod(u.luminosity);
    rational(u.scale); rational(u.offset);
  }
  void window(const Window& w) {
    pod(static_cast<std::uint8_t>(w.kind));
    pod(w.component.value());
    rational(w.length);
  }
  [[nodiscard]] std::vector<std::byte> take() { return std::move(buf_); }

 private:
  std::vector<std::byte> buf_;
};

class Reader {
 public:
  explicit Reader(std::span<const std::byte> b) : buf_(b) {}

  [[nodiscard]] bool ok() const noexcept { return ok_; }

  template <class T>
  T pod() {
    static_assert(std::is_trivially_copyable_v<T>);
    T v{};
    if (pos_ + sizeof(T) > buf_.size()) { ok_ = false; return v; }
    std::memcpy(&v, buf_.data() + pos_, sizeof(T));
    pos_ += sizeof(T);
    return v;
  }
  std::string str() {
    const auto n = pod<std::uint32_t>();
    if (!ok_ || pos_ + n > buf_.size()) { ok_ = false; return {}; }
    std::string s(reinterpret_cast<const char*>(buf_.data() + pos_), n);
    pos_ += n;
    return s;
  }
  Rational rational() {
    const auto n = pod<std::int64_t>();
    const auto d = pod<std::int64_t>();
    if (!ok_ || d == 0) { ok_ = false; return Rational{}; }
    return Rational{n, d};
  }
  Units units() {
    Units u;
    u.length = pod<std::int8_t>();      u.mass = pod<std::int8_t>();
    u.time = pod<std::int8_t>();        u.current = pod<std::int8_t>();
    u.temperature = pod<std::int8_t>(); u.amount = pod<std::int8_t>();
    u.luminosity = pod<std::int8_t>();
    u.scale = rational(); u.offset = rational();
    return u;
  }
  Window window() {
    Window w;
    w.kind = static_cast<Window::Kind>(pod<std::uint8_t>());
    w.component = ComponentId{pod<std::uint32_t>()};
    w.length = rational();
    return w;
  }
  [[nodiscard]] bool exhausted() const noexcept { return pos_ == buf_.size(); }

 private:
  std::span<const std::byte> buf_;
  std::size_t pos_ = 0;
  bool ok_ = true;
};

}  // namespace detail

/// Spec §3.2: the ABI narrow waist. C++, Fortran, Python, and YAML all
/// produce and consume this buffer.
[[nodiscard]] inline std::vector<std::byte> serialize(const CouplingGraph& g) {
  detail::Writer w;
  w.pod(kGraphMagic);
  w.pod(kGraphVersion);
  w.pod(static_cast<std::uint32_t>(g.mesh_count()));
  w.pod(static_cast<std::uint32_t>(g.region_count()));
  w.pod(static_cast<std::uint32_t>(g.component_count()));
  w.pod(static_cast<std::uint32_t>(g.field_count()));
  w.pod(static_cast<std::uint32_t>(g.channel_count()));
  w.pod(static_cast<std::uint32_t>(g.iteration_region_count()));
  w.pod(static_cast<std::uint32_t>(g.merge_count()));

  for (std::uint32_t i = 0; i < g.mesh_count(); ++i) {
    const auto& m = g.mesh(MeshId{i});
    w.str(m.name);
    w.str(m.source);
  }
  for (std::uint32_t i = 0; i < g.region_count(); ++i) {
    const auto& r = g.region(RegionId{i});
    w.str(r.name);
    w.pod(r.mesh.value());
    w.str(r.entity_set);
    w.pod(static_cast<std::int32_t>(r.dimension));
  }
  for (std::uint32_t i = 0; i < g.component_count(); ++i) {
    const auto& c = g.component(ComponentId{i});
    w.str(c.name);
    // Count then ranks, in the sorted order RankSet guarantees, so two ranks
    // serializing the same graph produce the same bytes.
    w.pod(static_cast<std::uint32_t>(c.ranks.size()));
    for (const int rank : c.ranks) w.pod(static_cast<std::int32_t>(rank));
    w.rational(c.interval);
  }
  for (std::uint32_t i = 0; i < g.field_count(); ++i) {
    const auto& f = g.field(FieldId{i});
    w.str(f.name);
    w.pod(g.owner_of(FieldId{i}).value());
    w.pod(f.region.value());
    w.pod(static_cast<std::uint8_t>(f.disc));
    w.pod(static_cast<std::int32_t>(f.order));
    w.pod(static_cast<std::int32_t>(f.components));
    w.units(f.units);
    w.pod(static_cast<std::uint8_t>(f.conservation));
    w.pod(static_cast<std::uint8_t>(f.intent));
  }
  for (std::uint32_t i = 0; i < g.channel_count(); ++i) {
    const auto& c = g.channel_at(ChannelId{i});
    w.str(c.name);
    w.pod(c.source.value());
    w.pod(c.target.value());
    w.pod(static_cast<std::uint8_t>(c.accumulate.op));
    w.window(c.accumulate.window);
    w.pod(static_cast<std::uint32_t>(c.remap.flags));
    w.str(c.remap.method_override);
    w.pod(static_cast<std::uint8_t>(c.spacetime.basis));
    w.pod(static_cast<std::int32_t>(c.spacetime.degree));
    w.window(c.spacetime.window);
    w.pod(static_cast<std::uint8_t>(c.relax.enabled));
    w.pod(c.relax.omega);
    w.pod(static_cast<std::uint8_t>(c.extrapolate.enabled));
    w.pod(static_cast<std::int32_t>(c.extrapolate.order));
    w.pod(static_cast<std::int32_t>(c.lag));
    w.str(c.time_level_mismatch_reason);
  }
  for (std::uint32_t i = 0; i < g.iteration_region_count(); ++i) {
    const auto& r = g.iteration_region(IterationRegionIdx{i});
    w.str(r.name);
    w.pod(static_cast<std::uint32_t>(r.channels.size()));
    for (const ChannelId c : r.channels) w.pod(c.value());
    w.pod(static_cast<std::int32_t>(r.max_iterations));
    // Spec §5.3, and the reason this record grew a version. Leaving the
    // criterion out here would silently drop it at the narrow waist: a graph
    // written by C++ and read back by Fortran, Python or the C ABI would
    // arrive with an EMPTY criterion, which reports satisfied and therefore
    // stops after one sweep. That is an explicit scheme reported as a
    // converged implicit one -- the wrong physics class, with no symptom.
    w.pod(static_cast<std::uint32_t>(r.convergence.size()));
    for (const ConvergenceMeasure& m : r.convergence.measures()) {
      w.pod(static_cast<std::uint8_t>(m.kind));
      w.pod(m.tolerance);
      w.str(m.field);
    }
    w.pod(static_cast<std::uint8_t>(r.acceleration.kind));
    w.pod(r.acceleration.initial_relaxation);
    w.pod(r.acceleration.omega);
    w.pod(static_cast<std::int32_t>(r.acceleration.reuse));
    w.pod(static_cast<std::uint8_t>(r.acceleration.filter));
    w.pod(r.acceleration.limit);
    w.pod(static_cast<std::uint8_t>(r.acceleration.scale_fields));
  }

  // The merge table. Sorted by field id in the graph, so the bytes are the
  // same whatever order the calls were made in -- the descriptor is compared
  // byte-for-byte across ranks, and an unordered container would make that
  // comparison depend on hash iteration order.
  for (std::size_t i = 0; i < g.merge_count(); ++i) {
    const auto& entry = g.merge_at(i);
    w.pod(entry.first);
    w.pod(static_cast<std::uint8_t>(entry.second.op));
  }

  return w.take();
}

[[nodiscard]] inline Result<CouplingGraph> deserialize(
    std::span<const std::byte> bytes) {
  detail::Reader r{bytes};
  if (r.pod<std::uint32_t>() != kGraphMagic || !r.ok()) {
    return Failure{"E-DESC-MAGIC", "not a FUSE graph descriptor"};
  }
  const auto version = r.pod<std::uint32_t>();
  if (version != kGraphVersion) {
    return Failure{"E-DESC-VERSION",
                 "descriptor version " + std::to_string(version) +
                     " but this build reads " + std::to_string(kGraphVersion)};
  }

  const auto n_mesh = r.pod<std::uint32_t>();
  const auto n_region = r.pod<std::uint32_t>();
  const auto n_comp = r.pod<std::uint32_t>();
  const auto n_field = r.pod<std::uint32_t>();
  const auto n_chan = r.pod<std::uint32_t>();
  const auto n_iter = r.pod<std::uint32_t>();
  const auto n_merge = r.pod<std::uint32_t>();
  if (!r.ok()) return Failure{"E-DESC-TRUNC", "truncated descriptor header"};

  CouplingGraph g;
  for (std::uint32_t i = 0; i < n_mesh; ++i) {
    MeshDecl m;
    m.name = r.str();
    m.source = r.str();
    g.add_mesh(std::move(m));
  }
  for (std::uint32_t i = 0; i < n_region; ++i) {
    Region reg;
    reg.name = r.str();
    reg.mesh = MeshId{r.pod<std::uint32_t>()};
    reg.entity_set = r.str();
    reg.dimension = r.pod<std::int32_t>();
    g.add_region(std::move(reg));
  }
  for (std::uint32_t i = 0; i < n_comp; ++i) {
    ComponentDecl c;
    c.name = r.str();
    {
      const auto n_ranks = r.pod<std::uint32_t>();
      if (!r.ok()) return Failure{"E-DESC-TRUNC", "truncated component ranks"};
      std::vector<int> ranks;
      ranks.reserve(n_ranks);
      for (std::uint32_t k = 0; k < n_ranks; ++k) {
        ranks.push_back(static_cast<int>(r.pod<std::int32_t>()));
      }
      if (!r.ok()) return Failure{"E-DESC-TRUNC", "truncated component ranks"};
      // Checked rather than trusted: RankSet's invariant is sorted and
      // distinct, and every consumer depends on it (CommGraph matches by
      // position, the agreement reduction indexes by position). A descriptor
      // that arrived unsorted would silently mis-place data.
      for (std::uint32_t k = 1; k < n_ranks; ++k) {
        if (ranks[k] <= ranks[k - 1]) {
          return Failure{"E-DESC-RANKS",
                         "component '" + c.name +
                             "' has ranks that are not strictly increasing; "
                             "refusing rather than sorting, because a "
                             "descriptor that disagrees with its writer about "
                             "rank order disagrees about which rank holds "
                             "which values"};
        }
      }
      c.ranks = RankSet{std::move(ranks)};
    }
    c.interval = r.rational();
    g.add_component(std::move(c));
  }
  for (std::uint32_t i = 0; i < n_field; ++i) {
    FieldSpec f;
    f.name = r.str();
    const ComponentId owner{r.pod<std::uint32_t>()};
    f.region = RegionId{r.pod<std::uint32_t>()};
    f.disc = static_cast<Discretization>(r.pod<std::uint8_t>());
    f.order = r.pod<std::int32_t>();
    f.components = r.pod<std::int32_t>();
    f.units = r.units();
    f.conservation = static_cast<Conservation>(r.pod<std::uint8_t>());
    f.intent = static_cast<Intent>(r.pod<std::uint8_t>());
    if (!r.ok() || owner.value() >= n_comp) {
      return Failure{"E-DESC-TRUNC", "truncated or invalid field table"};
    }
    g.add_field(owner, std::move(f));
  }
  for (std::uint32_t i = 0; i < n_chan; ++i) {
    const std::string name = r.str();
    const FieldId src{r.pod<std::uint32_t>()};
    const FieldId tgt{r.pod<std::uint32_t>()};
    if (!r.ok()) return Failure{"E-DESC-TRUNC", "truncated channel table"};
    const ChannelId cid = g.add_channel(name, src, tgt);
    auto b = g.channel(cid);
    const auto op = static_cast<AccumOp>(r.pod<std::uint8_t>());
    const Window aw = r.window();
    b.accumulate(op, aw);
    b.remap(static_cast<RemapFlags>(r.pod<std::uint32_t>()));
    b.remap_method(r.str());
    const auto basis = static_cast<TemporalBasis>(r.pod<std::uint8_t>());
    const auto degree = r.pod<std::int32_t>();
    const Window sw = r.window();
    b.spacetime(basis, degree, sw);
    const bool relax_on = r.pod<std::uint8_t>() != 0;
    const double omega = r.pod<double>();
    if (relax_on) b.relax(omega);
    const bool extrap_on = r.pod<std::uint8_t>() != 0;
    const auto extrap_order = r.pod<std::int32_t>();
    if (extrap_on) b.extrapolate(extrap_order);
    b.lag(r.pod<std::int32_t>());
    const std::string reason = r.str();
    if (!reason.empty()) b.allow_time_level_mismatch(reason);
  }
  for (std::uint32_t i = 0; i < n_iter; ++i) {
    IterationRegionDecl d;
    d.name = r.str();
    const auto n = r.pod<std::uint32_t>();
    if (!r.ok()) return Failure{"E-DESC-TRUNC", "truncated iteration region"};
    d.channels.reserve(n);
    for (std::uint32_t k = 0; k < n; ++k) {
      d.channels.push_back(ChannelId{r.pod<std::uint32_t>()});
    }
    d.max_iterations = r.pod<std::int32_t>();
    const auto n_measure = r.pod<std::uint32_t>();
    if (!r.ok()) return Failure{"E-DESC-TRUNC", "truncated iteration region"};
    for (std::uint32_t k = 0; k < n_measure; ++k) {
      ConvergenceMeasure m;
      m.kind = static_cast<MeasureKind>(r.pod<std::uint8_t>());
      m.tolerance = r.pod<double>();
      m.field = r.str();
      if (!r.ok()) {
        return Failure{"E-DESC-TRUNC", "truncated convergence measure"};
      }
      d.convergence.add(std::move(m));
    }
    d.acceleration.kind = static_cast<AccelKind>(r.pod<std::uint8_t>());
    d.acceleration.initial_relaxation = r.pod<double>();
    d.acceleration.omega = r.pod<double>();
    d.acceleration.reuse = r.pod<std::int32_t>();
    d.acceleration.filter = static_cast<FilterKind>(r.pod<std::uint8_t>());
    d.acceleration.limit = r.pod<double>();
    d.acceleration.scale_fields = r.pod<std::uint8_t>() != 0;
    if (!r.ok()) return Failure{"E-DESC-TRUNC", "truncated iteration region"};
    g.add_iteration_region(std::move(d));
  }

  for (std::uint32_t i = 0; i < n_merge; ++i) {
    const auto field_value = r.pod<std::uint32_t>();
    const auto op = r.pod<std::uint8_t>();
    if (!r.ok()) return Failure{"E-DESC-TRUNC", "truncated merge table"};
    if (field_value >= g.field_count()) {
      return Failure{"E-DESC-MERGE",
                     "the merge table names field id " + std::to_string(field_value) +
                         " but the descriptor carries " +
                         std::to_string(g.field_count()) + " fields"};
    }
    if (op > static_cast<std::uint8_t>(MergeOp::First)) {
      return Failure{"E-DESC-MERGE",
                     "unknown merge rule " + std::to_string(op) +
                         "; refusing rather than defaulting to Sum, which would "
                         "silently change the physics"};
    }
    g.merge(FieldId{field_value}, static_cast<MergeOp>(op));
  }

  if (!r.ok()) return Failure{"E-DESC-TRUNC", "truncated descriptor body"};
  if (!r.exhausted()) {
    return Failure{"E-DESC-TRAIL", "trailing bytes after descriptor"};
  }
  return g;
}

/// Compares every field the descriptor carries. Used by round-trip tests and
/// by the YAML round-trip in Task 1.7.
[[nodiscard]] inline bool structurally_equal(const CouplingGraph& a,
                                             const CouplingGraph& b) {
  return serialize(a) == serialize(b);
}

}  // namespace fuse
