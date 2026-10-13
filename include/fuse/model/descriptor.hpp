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
/// 14: the NewtonModel track (2026-10-12), measure-weighted Newton residual
///     scaling and the direct residual-evaluation path. `NewtonConfig`
///     (fuse/model/iteration.hpp) gains two more fields, `norm` (`fuse::
///     NewtonNorm`, Plain or MeasureWeighted) and `residual_path`
///     (`fuse::NewtonResidualPath`, Advance or Direct), each appended at
///     the END of the v13 NewtonConfig block in that order -- two more
///     enum bytes after `substrate`, following v13's own append-only
///     placement exactly (every earlier NewtonConfig field keeps its wire
///     offset; only the block's own trailing length grows, from 162 to
///     164 bytes including the two capability bytes). Neither field
///     changes `AccelKind` or widens its bound check -- both are
///     independent of which accelerator is selected, read unconditionally
///     for every region exactly like every other NewtonConfig field
///     already is. A v13 reader handed a v14 buffer would read the FIRST
///     region's `norm` byte as the start of whatever comes next (the next
///     region's own name length, or the merge table) -- everything after
///     it decodes as plausible garbage, the same class of silent misread
///     every prior bump was made to prevent, so v13 is refused.
///
/// 13: S7.2b (PetscSelect track), the per-region Newton substrate
///     selector. `NewtonConfig` (fuse/model/iteration.hpp) gains one more
///     field, `substrate` (`fuse::NewtonSubstrate`, Native or Petsc),
///     appended at the END of the v12 NewtonConfig block -- one more enum
///     byte after `use_block_jacobi_preconditioner`, following v12's own
///     append-only placement exactly (every earlier NewtonConfig field
///     keeps its wire offset; only the block's own trailing length
///     grows, from 161 to 162 bytes including the two capability bytes).
///     No `AccelKind` change and no widening of its own bound check --
///     `substrate` is independent of which accelerator is selected, read
///     unconditionally for every region exactly like every other
///     NewtonConfig field already is. A v12 reader handed a v13 buffer
///     would read the FIRST region's substrate byte as the start of
///     whatever comes next (the next region's own name length, or the
///     merge table) -- everything after it decodes as plausible garbage,
///     the same class of silent misread every prior bump was made to
///     prevent, so v12 is refused.
///
/// 12: S7.2, the native Jacobian-free Newton-Krylov coupler. Each iteration
///     region record gains, after its window-scope block: two capability
///     bytes (`IterationRegionDecl::requires_residual_capability`,
///     `::requires_linearization_capability`), and `NewtonConfig`'s own
///     25 fields in full (two enum bytes -- `line_search_kind`,
///     `forcing_choice` -- then 15 doubles, 5 signed words, 2 unsigned
///     64-bit counts, and a final presence byte for
///     `use_block_jacobi_preconditioner`), appended in the SAME declaration
///     order `fuse::NewtonConfig` (fuse/model/iteration.hpp) lists them.
///     Also widens the accelerator-kind bound check to admit
///     `AccelKind::Newton` (ordinal 7, appended after `IQN_IMVJ`). A v11
///     reader handed a v12 buffer would read the FIRST region's residual-
///     capability byte as the start of whatever comes next (the next
///     region's own name length, or the merge table) -- everything after
///     it decodes as plausible garbage, the same class of silent misread
///     every prior bump was made to prevent, so v11 is refused.
///
/// 11: the "LowFive" track, Unit 3. Each field record gains a presence byte
///     after `intent`, and when set, one more length-prefixed string:
///     `hdf5_path` (`FieldSpec::hdf5_path`, fuse/model/field_spec.hpp). Also
///     adds a FIFTH `LaunchMode` value, `LowFive` (fuse/model/launch.hpp) --
///     no new bytes of its own (it carries no LaunchSpec member beyond the
///     mode byte itself), but the enum-range check on that byte has to
///     widen to admit it, same as every prior LaunchMode addition. A v10
///     reader handed a v11 buffer would read the FIRST field's hdf5_path
///     presence byte as the start of the next field's name length, and
///     everything after it would decode as plausible garbage from there on
///     -- the same class of silent misread every prior bump was made to
///     prevent, so v10 is refused.
///
/// 10: the provider direction (S6 phase 4, "Adapter host" track). Each
///     component record's launch record (v9) gains a FOURTH `LaunchMode`
///     value, `Plugin`, and one more length-prefixed string after
///     `service_name`: `plugin_path`, the shared object
///     `fuse::abi_detail::Registry` dlopen()s for this component
///     (include/fuse/abi/provider.h). A v9 reader handed a v10 buffer would
///     read the FIRST component's `plugin_path` length as the start of the
///     next component's field count, and everything after it would decode
///     as plausible garbage from there on -- the same class of silent
///     misread every prior bump was made to prevent, so v9 is refused.
///
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
/// 9: the interfaces track (S6 phase 3). Each component record gains a
///    launch record after its `kind` byte: a mode byte (`LaunchMode`:
///    InProcess, Mpmd or Connect), a signed appnum word, an `is_acceptor`
///    byte, and two length-prefixed strings (`port_file`, `service_name`).
///    A v8 reader handed a v9 buffer would read the launch mode byte of the
///    FIRST component as the start of the next component's field count, and
///    everything after it would decode as plausible garbage from there on --
///    the same class of silent misread every prior bump was made to
///    prevent, so v8 is refused.
///
/// 8: the waveform-iteration option. A window scope's record (v7) gains one
///    more byte after the per-member `DtPolicy` list: `WaveformIteration`
///    (Jacobi or GaussSeidel), which classical scheme the window's own
///    sweep loop runs. A v7 reader handed a v8 buffer would read the FIRST
///    window scope's iteration byte as the start of whatever comes next
///    (the merge table, or the next region's own presence byte) --
///    everything after it decodes as plausible garbage, the same class of
///    silent misread every prior bump was made to prevent, so v7 is refused.
///
/// 7: the multirate-window-on-the-Runner track. Each iteration region record
///    gains an optional `CouplingWindowScope` after its acceleration block: a
///    presence byte, and when set, an optional window-defining component name
///    or explicit interval, the interpolation/extrapolation order, and the
///    per-member `DtPolicy` list. A v6 reader handed a v7 buffer would read
///    the window-scope presence byte of the FIRST region as the start of the
///    next region's name length, and everything after it would decode as
///    plausible garbage -- the same class of silent misread every prior bump
///    was made to prevent, so v6 is refused.
///
/// 6: the coupler-placement track. Each component record gains a `kind` byte
///    (`ComponentKind`: Physics or Coupler) after its interval, and each
///    channel record gains a `remap_on` component-id word after its remap
///    method string. A v5 reader handed a v6 buffer would read the coupler
///    kind byte of the FIRST component as the start of the next component's
///    name length, and every field after it would decode as plausible
///    garbage from there on -- the same class of silent misread v5 itself
///    was bumped to prevent, so v5 is refused.
///
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
inline constexpr std::uint32_t kGraphVersion = 14u;

namespace detail {

// BM-5 (review 2026-10-11): `pod()` below writes a value's native
// in-memory representation -- host-endian, NOT the little-endian rule
// fuse/base/bytes.hpp states for every other canonical encoding in this
// library. See bytes.hpp's own note and docs/dev/abi-design.md's "The
// graph descriptor's byte order" section for why this is recorded rather
// than changed immediately.
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
  /// BM-7: the wire contract for an exact-time record is "reduced, with a
  /// positive denominator" -- `Rational::from_record` enforces exactly
  /// that, with `E-BASE-TIME-DENOMINATOR`/`E-BASE-TIME-UNREDUCED`, for the
  /// SAME ABI record (`fuse::ExactTimeRecord`) crossing the SAME trust
  /// boundary through a different entry point. This reader used to accept
  /// and silently renormalize a non-positive denominator or an unreduced
  /// pair instead of refusing -- hiding, on one of the descriptor's own
  /// entry points, exactly the producer bug `from_record` exists to catch
  /// on the other. `ok_` is this class's one failure signal, the same one
  /// the truncation check already below used (now also covering a
  /// negative denominator, not only zero); every caller already checks
  /// `r.ok()` once after the record this value belongs to. No descriptor
  /// legitimately produced by this library's own `Rational` can trip this
  /// -- `Rational` maintains "reduced, positive denominator" as a class
  /// invariant -- so this only ever refuses a hand-patched or corrupted
  /// buffer, never the three-leg round trip's own golden bytes.
  Rational rational() {
    const auto n = pod<std::int64_t>();
    const auto d = pod<std::int64_t>();
    if (!ok_ || d <= 0) { ok_ = false; return Rational{}; }
    const Rational reduced{n, d};
    if (reduced.num() != n || reduced.den() != d) { ok_ = false; return Rational{}; }
    return reduced;
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
  /// BM-3: `kind` is read and range-checked by the caller (the same
  /// convention every other enumerator byte in `deserialize()` follows --
  /// `AccelKind`/`FilterKind`/`MeasureKind`/`ComponentKind`/`LaunchMode`
  /// are each checked at the call site, not inside a shared helper), so a
  /// corrupted byte is refused with `E-DESC-ENUM` before this ever
  /// constructs a `Window` with it. Before this change `Window::Kind` was
  /// the one enum this file's own R-04 sweep never reached: it read
  /// straight off the wire here with no check at all, and BOTH consumers
  /// (`validate.hpp`'s `channel_window()`, `yaml_io.cpp`'s `emit_window()`)
  /// silently folded an out-of-range value into `CouplingPeriod` -- fixed
  /// alongside this by removing those folds.
  Window window(Window::Kind kind) {
    Window w;
    w.kind = kind;
    w.component = ComponentId{pod<std::uint32_t>()};
    w.length = rational();
    return w;
  }
  [[nodiscard]] bool exhausted() const noexcept { return pos_ == buf_.size(); }

  /// BM-18: how many bytes are left to read. A count read off the wire
  /// (a ranks count, a channel-id count, a policy count) must be checked
  /// against this -- `count * min_record_size <= remaining()` -- BEFORE
  /// `reserve()`ing it: `std::vector::reserve` on an attacker-controlled
  /// `std::uint32_t` can ask for gigabytes (or throw `std::length_error`/
  /// `std::bad_alloc`, a non-`Result` escape from the one function whose
  /// whole contract is corrupted-input safety) long before the truncation
  /// that count implies would otherwise be noticed.
  [[nodiscard]] std::size_t remaining() const noexcept {
    return buf_.size() - pos_;
  }

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
    w.pod(static_cast<std::uint8_t>(c.kind));
    // v9: the launch record (descriptor.hpp's kGraphVersion history).
    w.pod(static_cast<std::uint8_t>(c.launch.mode));
    w.pod(static_cast<std::int32_t>(c.launch.appnum));
    w.pod(static_cast<std::uint8_t>(c.launch.is_acceptor ? 1 : 0));
    w.str(c.launch.port_file);
    w.str(c.launch.service_name);
    // v10: the provider direction (descriptor.hpp's kGraphVersion history).
    w.str(c.launch.plugin_path);
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
    // v11: the "LowFive" track, Unit 3 (descriptor.hpp's kGraphVersion
    // history).
    w.pod(static_cast<std::uint8_t>(f.hdf5_path.has_value() ? 1 : 0));
    if (f.hdf5_path.has_value()) w.str(*f.hdf5_path);
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
    w.pod(c.remap.remap_on.value());
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

    // v7: the multirate-window opt-in. `std::nullopt` writes a single zero
    // byte and nothing else, so a descriptor with no window-scoped region
    // (every region declared before this member existed) grows by exactly
    // one byte per region -- not by the whole payload below.
    w.pod(static_cast<std::uint8_t>(r.window_scope.has_value() ? 1 : 0));
    if (r.window_scope.has_value()) {
      const CouplingWindowScope& scope = *r.window_scope;
      w.pod(static_cast<std::uint8_t>(scope.window_component.has_value() ? 1 : 0));
      if (scope.window_component.has_value()) w.str(*scope.window_component);
      w.pod(static_cast<std::uint8_t>(scope.explicit_interval.has_value() ? 1 : 0));
      if (scope.explicit_interval.has_value()) w.rational(*scope.explicit_interval);
      w.pod(static_cast<std::int32_t>(scope.order));
      w.pod(static_cast<std::uint32_t>(scope.policies.size()));
      for (const WindowMemberPolicy& p : scope.policies) {
        w.str(p.component);
        w.pod(static_cast<std::uint8_t>(p.policy));
      }
      // v8: the waveform-iteration option.
      w.pod(static_cast<std::uint8_t>(scope.iteration));
    }

    // v12: S7.2's own Newton-Krylov coupler (descriptor.hpp's
    // kGraphVersion history). The two capability bytes first, then
    // NewtonConfig in full, in its own declaration order -- always
    // written, the same unconditional convention `acceleration` itself
    // already uses (not gated on `acceleration.kind == AccelKind::Newton`,
    // so a region's own capability flags survive even when set by
    // IterationBuilder::requires_residual() independently of Newton).
    w.pod(static_cast<std::uint8_t>(r.requires_residual_capability ? 1 : 0));
    w.pod(static_cast<std::uint8_t>(r.requires_linearization_capability ? 1 : 0));
    {
      const NewtonConfig& nc = r.newton_config;
      w.pod(static_cast<std::uint8_t>(nc.line_search_kind));
      w.pod(static_cast<std::uint8_t>(nc.forcing_choice));
      w.pod(nc.forcing_gamma);
      w.pod(nc.forcing_alpha);
      w.pod(nc.eta_initial);
      w.pod(nc.eta_max);
      w.pod(nc.eta_min);
      w.pod(nc.forcing_safeguard_threshold);
      w.pod(nc.forcing_eta_constant);
      w.pod(nc.atol);
      w.pod(nc.rtol);
      w.pod(nc.stol);
      w.pod(nc.divergence_tolerance);
      w.pod(static_cast<std::int32_t>(nc.max_nonlinear_iterations));
      w.pod(static_cast<std::uint64_t>(nc.max_function_evaluations));
      w.pod(nc.armijo_alpha);
      w.pod(nc.lambda_min);
      w.pod(static_cast<std::int32_t>(nc.max_backtracks));
      w.pod(nc.secant_tol);
      w.pod(static_cast<std::int32_t>(nc.max_secant_steps));
      w.pod(nc.damping);
      w.pod(static_cast<std::int32_t>(nc.gmres_restart));
      w.pod(static_cast<std::uint64_t>(nc.gmres_max_iterations));
      w.pod(static_cast<std::int32_t>(nc.preconditioner_lag));
      w.pod(static_cast<std::uint8_t>(nc.use_block_jacobi_preconditioner ? 1 : 0));
      // v13: S7.2b's own per-region substrate selector (descriptor.hpp's
      // kGraphVersion history), appended at the END of the block,
      // following v12's own append-only placement exactly.
      w.pod(static_cast<std::uint8_t>(nc.substrate));
      // v14: the NewtonModel track's own `norm`/`residual_path`
      // (descriptor.hpp's kGraphVersion history), appended in that order
      // at the END of the block, following v13's own append-only
      // placement exactly.
      w.pod(static_cast<std::uint8_t>(nc.norm));
      w.pod(static_cast<std::uint8_t>(nc.residual_path));
    }
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
      // BM-18: each rank is a 4-byte std::int32_t on the wire; a count
      // claiming more ranks than the remaining buffer could possibly hold
      // is truncated, not merely large, and must be refused before
      // `reserve()` acts on it.
      if (static_cast<std::uint64_t>(n_ranks) * sizeof(std::int32_t) > r.remaining()) {
        return Failure{"E-DESC-TRUNC",
                       "component '" + c.name + "' declares " +
                           std::to_string(n_ranks) +
                           " ranks, more than the remaining buffer can hold"};
      }
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
    // REVIEW.md R-04's pattern (MeasureKind/AccelKind/FilterKind below):
    // this byte comes straight off the wire, and an out-of-range value must
    // be refused here rather than admitted and later indexed unchecked by
    // to_yaml()'s enum-name table.
    const auto kind_byte = r.pod<std::uint8_t>();
    if (kind_byte > static_cast<std::uint8_t>(ComponentKind::Coupler)) {
      return Failure{"E-DESC-ENUM",
                     "unknown component kind " + std::to_string(kind_byte)};
    }
    c.kind = static_cast<ComponentKind>(kind_byte);
    // v9: the launch record (descriptor.hpp's kGraphVersion history). Same
    // REVIEW.md R-04 pattern as the kind byte just above: an out-of-range
    // mode must be refused here, not admitted and later indexed unchecked
    // by to_cstring()'s switch.
    const auto launch_mode_byte = r.pod<std::uint8_t>();
    // v11: the "LowFive" track, Unit 3 (descriptor.hpp's kGraphVersion
    // history) widens this to admit LaunchMode::LowFive.
    if (launch_mode_byte > static_cast<std::uint8_t>(LaunchMode::LowFive)) {
      return Failure{"E-DESC-ENUM",
                     "unknown launch mode " + std::to_string(launch_mode_byte)};
    }
    c.launch.mode = static_cast<LaunchMode>(launch_mode_byte);
    c.launch.appnum = r.pod<std::int32_t>();
    c.launch.is_acceptor = r.pod<std::uint8_t>() != 0;
    c.launch.port_file = r.str();
    c.launch.service_name = r.str();
    // v10: the provider direction (descriptor.hpp's kGraphVersion history).
    c.launch.plugin_path = r.str();
    if (!r.ok()) return Failure{"E-DESC-TRUNC", "truncated component launch record"};
    g.add_component(std::move(c));
  }
  for (std::uint32_t i = 0; i < n_field; ++i) {
    FieldSpec f;
    f.name = r.str();
    const ComponentId owner{r.pod<std::uint32_t>()};
    f.region = RegionId{r.pod<std::uint32_t>()};
    // BM-3: the six enum bytes named in the review's own table (Window::
    // Kind, AccumOp, TemporalBasis, Discretization, Conservation, Intent)
    // were the ones R-04's "validate every enumerator read from bytes
    // against its range" rule never reached. Discretization, Conservation
    // and Intent read straight off the wire here with no check at all --
    // the same class MeasureKind/AccelKind/FilterKind/ComponentKind/
    // LaunchMode already guard a few lines above and below this function.
    const auto disc_byte = r.pod<std::uint8_t>();
    if (disc_byte > static_cast<std::uint8_t>(Discretization::PointCloud)) {
      return Failure{"E-DESC-ENUM",
                     "unknown discretization " + std::to_string(disc_byte)};
    }
    f.disc = static_cast<Discretization>(disc_byte);
    f.order = r.pod<std::int32_t>();
    f.components = r.pod<std::int32_t>();
    f.units = r.units();
    const auto conservation_byte = r.pod<std::uint8_t>();
    if (conservation_byte > static_cast<std::uint8_t>(Conservation::Intensive)) {
      return Failure{"E-DESC-ENUM",
                     "unknown conservation class " + std::to_string(conservation_byte)};
    }
    f.conservation = static_cast<Conservation>(conservation_byte);
    const auto intent_byte = r.pod<std::uint8_t>();
    if (intent_byte > static_cast<std::uint8_t>(Intent::Bidirectional)) {
      return Failure{"E-DESC-ENUM",
                     "unknown intent " + std::to_string(intent_byte)};
    }
    f.intent = static_cast<Intent>(intent_byte);
    // v11: the "LowFive" track, Unit 3 (descriptor.hpp's kGraphVersion
    // history).
    if (r.pod<std::uint8_t>() != 0) f.hdf5_path = r.str();
    if (!r.ok() || owner.value() >= n_comp) {
      return Failure{"E-DESC-TRUNC", "truncated or invalid field table"};
    }
    // REVIEW.md R-19: Units has no constructor of its own to gate this at
    // (every call site aggregate-initializes it), so the descriptor reader
    // is the narrowest choke point available: a unit's scale relative to SI
    // base units is a conversion factor and is never non-positive (zero is
    // an annihilator, negative is a sign flip), and a corrupted or
    // adversarially constructed byte buffer must not be able to introduce
    // either through this, the one input in this scope that crosses a real
    // trust boundary.
    if (!(f.units.scale > Rational{0})) {
      return Failure{"E-DESC-UNITS-SCALE",
                     "field '" + f.name + "' has non-positive unit scale " +
                         f.units.scale.to_string()};
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
    // BM-3: AccumOp was another of the six unguarded enum bytes; refused
    // here with the same E-DESC-ENUM convention as every neighbouring
    // enumerator check in this function.
    const auto accum_op_byte = r.pod<std::uint8_t>();
    if (accum_op_byte > static_cast<std::uint8_t>(AccumOp::Integral)) {
      return Failure{"E-DESC-ENUM",
                     "unknown accumulate op " + std::to_string(accum_op_byte)};
    }
    const auto op = static_cast<AccumOp>(accum_op_byte);
    const auto accum_window_kind_byte = r.pod<std::uint8_t>();
    if (accum_window_kind_byte > static_cast<std::uint8_t>(Window::Kind::Explicit)) {
      return Failure{"E-DESC-ENUM",
                     "unknown window kind " + std::to_string(accum_window_kind_byte)};
    }
    const Window aw = r.window(static_cast<Window::Kind>(accum_window_kind_byte));
    b.accumulate(op, aw);
    b.remap(static_cast<RemapFlags>(r.pod<std::uint32_t>()));
    b.remap_method(r.str());
    b.remap_on(ComponentId{r.pod<std::uint32_t>()});
    // BM-3: TemporalBasis, the fourth of the six.
    const auto basis_byte = r.pod<std::uint8_t>();
    if (basis_byte > static_cast<std::uint8_t>(TemporalBasis::Spline)) {
      return Failure{"E-DESC-ENUM",
                     "unknown temporal basis " + std::to_string(basis_byte)};
    }
    const auto basis = static_cast<TemporalBasis>(basis_byte);
    const auto degree = r.pod<std::int32_t>();
    const auto spacetime_window_kind_byte = r.pod<std::uint8_t>();
    if (spacetime_window_kind_byte > static_cast<std::uint8_t>(Window::Kind::Explicit)) {
      return Failure{"E-DESC-ENUM",
                     "unknown window kind " + std::to_string(spacetime_window_kind_byte)};
    }
    const Window sw = r.window(static_cast<Window::Kind>(spacetime_window_kind_byte));
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
    // BM-18: each channel id is a 4-byte std::uint32_t on the wire.
    if (static_cast<std::uint64_t>(n) * sizeof(std::uint32_t) > r.remaining()) {
      return Failure{"E-DESC-TRUNC",
                     "iteration region '" + d.name + "' declares " +
                         std::to_string(n) +
                         " channels, more than the remaining buffer can hold"};
    }
    d.channels.reserve(n);
    for (std::uint32_t k = 0; k < n; ++k) {
      // BM-4: unlike the merge table's field ids (E-DESC-MERGE, below),
      // an iteration-region channel id was accepted here with no bounds
      // check at all; the first dereference is a raw `std::vector::at`
      // inside `build_iteration_regions()`/`has_unbroken_cycle()`
      // (src/ir/regions.cpp, src/schedule/window.cpp), so a corrupted
      // descriptor turned into an uncaught `std::out_of_range` -- a
      // per-rank throw ahead of later collectives, the I2 hang pattern.
      // `to_yaml()`'s emit path already guards the identical id
      // (`E-YAML-REF`, src/model/yaml_io.cpp); the read path now mirrors
      // it at the waist, with its own code.
      const auto channel_value = r.pod<std::uint32_t>();
      if (!r.ok()) return Failure{"E-DESC-TRUNC", "truncated iteration region"};
      if (channel_value >= g.channel_count()) {
        return Failure{"E-DESC-CHANNEL",
                       "iteration region '" + d.name +
                           "' references channel id " +
                           std::to_string(channel_value) + ", but the "
                           "descriptor carries " +
                           std::to_string(g.channel_count()) + " channels"};
      }
      d.channels.push_back(ChannelId{channel_value});
    }
    d.max_iterations = r.pod<std::int32_t>();
    const auto n_measure = r.pod<std::uint32_t>();
    if (!r.ok()) return Failure{"E-DESC-TRUNC", "truncated iteration region"};
    for (std::uint32_t k = 0; k < n_measure; ++k) {
      ConvergenceMeasure m;
      // REVIEW.md R-04: deserialize() is the ABI narrow waist every language
      // binding funnels through, and this byte comes straight off the wire.
      // Refusing an out-of-range enumerator here (matching the MergeOp check
      // above) is what stands between a corrupted descriptor and the raw,
      // unchecked array index in yaml_io.cpp's to_yaml() -- admitting the
      // value would be undefined behavior later, disconnected from its cause.
      const auto measure_kind = r.pod<std::uint8_t>();
      if (measure_kind > static_cast<std::uint8_t>(MeasureKind::ResidualRelative)) {
        return Failure{"E-DESC-ENUM",
                       "unknown convergence measure kind " +
                           std::to_string(measure_kind)};
      }
      m.kind = static_cast<MeasureKind>(measure_kind);
      m.tolerance = r.pod<double>();
      m.field = r.str();
      if (!r.ok()) {
        return Failure{"E-DESC-TRUNC", "truncated convergence measure"};
      }
      d.convergence.add(std::move(m));
    }
    // REVIEW.md R-04: same reasoning as the measure kind above -- both bytes
    // are read from the same untrusted buffer and indexed unchecked by
    // to_yaml()'s kAccelNames/kFilterNames tables.
    const auto accel_kind = r.pod<std::uint8_t>();
    // v12: widened to admit AccelKind::Newton (descriptor.hpp's
    // kGraphVersion history).
    if (accel_kind > static_cast<std::uint8_t>(AccelKind::Newton)) {
      return Failure{"E-DESC-ENUM",
                     "unknown acceleration kind " + std::to_string(accel_kind)};
    }
    d.acceleration.kind = static_cast<AccelKind>(accel_kind);
    d.acceleration.initial_relaxation = r.pod<double>();
    d.acceleration.omega = r.pod<double>();
    d.acceleration.reuse = r.pod<std::int32_t>();
    const auto accel_filter = r.pod<std::uint8_t>();
    if (accel_filter > static_cast<std::uint8_t>(FilterKind::QR3)) {
      return Failure{"E-DESC-ENUM",
                     "unknown acceleration filter " + std::to_string(accel_filter)};
    }
    d.acceleration.filter = static_cast<FilterKind>(accel_filter);
    d.acceleration.limit = r.pod<double>();
    d.acceleration.scale_fields = r.pod<std::uint8_t>() != 0;
    if (!r.ok()) return Failure{"E-DESC-TRUNC", "truncated iteration region"};

    // v7: the multirate-window opt-in.
    const bool has_scope = r.pod<std::uint8_t>() != 0;
    if (!r.ok()) return Failure{"E-DESC-TRUNC", "truncated iteration region"};
    if (has_scope) {
      CouplingWindowScope scope;
      const bool has_component = r.pod<std::uint8_t>() != 0;
      if (!r.ok()) return Failure{"E-DESC-TRUNC", "truncated window scope"};
      if (has_component) scope.window_component = r.str();
      const bool has_interval = r.pod<std::uint8_t>() != 0;
      if (!r.ok()) return Failure{"E-DESC-TRUNC", "truncated window scope"};
      if (has_interval) scope.explicit_interval = r.rational();
      scope.order = r.pod<std::int32_t>();
      const auto n_policies = r.pod<std::uint32_t>();
      if (!r.ok()) return Failure{"E-DESC-TRUNC", "truncated window scope"};
      // BM-18: each policy record is at minimum a 4-byte empty-string
      // length prefix plus a 1-byte policy enum.
      if (static_cast<std::uint64_t>(n_policies) * 5 > r.remaining()) {
        return Failure{"E-DESC-TRUNC",
                       "window scope declares " + std::to_string(n_policies) +
                           " policies, more than the remaining buffer can "
                           "hold"};
      }
      scope.policies.reserve(n_policies);
      for (std::uint32_t k = 0; k < n_policies; ++k) {
        WindowMemberPolicy p;
        p.component = r.str();
        const auto policy_byte = r.pod<std::uint8_t>();
        // REVIEW.md R-04's pattern (MeasureKind/AccelKind/FilterKind above):
        // this byte comes straight off the wire and is indexed unchecked by
        // to_yaml()'s enum-name table.
        if (policy_byte > static_cast<std::uint8_t>(DtPolicy::Adjustable)) {
          return Failure{"E-DESC-ENUM",
                        "unknown DtPolicy " + std::to_string(policy_byte) +
                            " in window scope policy for component '" +
                            p.component + "'"};
        }
        p.policy = static_cast<DtPolicy>(policy_byte);
        if (!r.ok()) return Failure{"E-DESC-TRUNC", "truncated window scope"};
        scope.policies.push_back(std::move(p));
      }
      // v8: the waveform-iteration option.
      const auto iteration_byte = r.pod<std::uint8_t>();
      if (!r.ok()) return Failure{"E-DESC-TRUNC", "truncated window scope"};
      if (iteration_byte >
          static_cast<std::uint8_t>(WaveformIteration::GaussSeidel)) {
        return Failure{"E-DESC-ENUM",
                       "unknown WaveformIteration " +
                           std::to_string(iteration_byte) + " in window scope"};
      }
      scope.iteration = static_cast<WaveformIteration>(iteration_byte);
      d.window_scope = std::move(scope);
    }

    // v12: S7.2's own Newton-Krylov coupler (descriptor.hpp's
    // kGraphVersion history). Mirrors the write side's own order exactly:
    // the two capability bytes, then NewtonConfig in full.
    d.requires_residual_capability = r.pod<std::uint8_t>() != 0;
    d.requires_linearization_capability = r.pod<std::uint8_t>() != 0;
    if (!r.ok()) return Failure{"E-DESC-TRUNC", "truncated iteration region capability bytes"};
    {
      NewtonConfig& nc = d.newton_config;
      const auto ls_kind = r.pod<std::uint8_t>();
      if (ls_kind > static_cast<std::uint8_t>(LineSearchKind::Basic)) {
        return Failure{"E-DESC-ENUM",
                       "unknown newton_config.line_search_kind " + std::to_string(ls_kind)};
      }
      nc.line_search_kind = static_cast<LineSearchKind>(ls_kind);
      const auto ew_choice = r.pod<std::uint8_t>();
      if (ew_choice > static_cast<std::uint8_t>(EisenstatWalkerChoice::Choice2)) {
        return Failure{"E-DESC-ENUM",
                       "unknown newton_config.forcing_choice " + std::to_string(ew_choice)};
      }
      nc.forcing_choice = static_cast<EisenstatWalkerChoice>(ew_choice);
      nc.forcing_gamma = r.pod<double>();
      nc.forcing_alpha = r.pod<double>();
      nc.eta_initial = r.pod<double>();
      nc.eta_max = r.pod<double>();
      nc.eta_min = r.pod<double>();
      nc.forcing_safeguard_threshold = r.pod<double>();
      nc.forcing_eta_constant = r.pod<double>();
      nc.atol = r.pod<double>();
      nc.rtol = r.pod<double>();
      nc.stol = r.pod<double>();
      nc.divergence_tolerance = r.pod<double>();
      nc.max_nonlinear_iterations = r.pod<std::int32_t>();
      nc.max_function_evaluations = r.pod<std::uint64_t>();
      nc.armijo_alpha = r.pod<double>();
      nc.lambda_min = r.pod<double>();
      nc.max_backtracks = r.pod<std::int32_t>();
      nc.secant_tol = r.pod<double>();
      nc.max_secant_steps = r.pod<std::int32_t>();
      nc.damping = r.pod<double>();
      nc.gmres_restart = r.pod<std::int32_t>();
      nc.gmres_max_iterations = r.pod<std::uint64_t>();
      nc.preconditioner_lag = r.pod<std::int32_t>();
      nc.use_block_jacobi_preconditioner = r.pod<std::uint8_t>() != 0;
      // v13: S7.2b's own per-region substrate selector (descriptor.hpp's
      // kGraphVersion history). Mirrors the write side's own order:
      // appended at the END of the block, read last.
      const auto substrate_byte = r.pod<std::uint8_t>();
      if (substrate_byte > static_cast<std::uint8_t>(NewtonSubstrate::Petsc)) {
        return Failure{"E-DESC-ENUM",
                       "unknown newton_config.substrate " + std::to_string(substrate_byte)};
      }
      nc.substrate = static_cast<NewtonSubstrate>(substrate_byte);
      // v14: the NewtonModel track's own `norm`/`residual_path`
      // (descriptor.hpp's kGraphVersion history). Mirrors the write
      // side's own order: appended at the END of the block, read last.
      const auto norm_byte = r.pod<std::uint8_t>();
      if (norm_byte > static_cast<std::uint8_t>(NewtonNorm::MeasureWeighted)) {
        return Failure{"E-DESC-ENUM",
                       "unknown newton_config.norm " + std::to_string(norm_byte)};
      }
      nc.norm = static_cast<NewtonNorm>(norm_byte);
      const auto residual_path_byte = r.pod<std::uint8_t>();
      if (residual_path_byte > static_cast<std::uint8_t>(NewtonResidualPath::Direct)) {
        return Failure{"E-DESC-ENUM",
                       "unknown newton_config.residual_path " + std::to_string(residual_path_byte)};
      }
      nc.residual_path = static_cast<NewtonResidualPath>(residual_path_byte);
    }
    if (!r.ok()) return Failure{"E-DESC-TRUNC", "truncated newton_config"};
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
