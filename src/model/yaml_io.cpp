#include "fuse/model/yaml_io.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>
#include <optional>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "fuse/model/descriptor.hpp"
#include "fuse/model/semantics.hpp"
#include "fuse/model/validate.hpp"

namespace fuse {
namespace {

constexpr int kSchemaVersion = 1;

// --- symbol tables ---------------------------------------------------------

struct UnitEntry { const char* symbol; Units value; };

const UnitEntry kUnitTable[] = {
    {"1", units::kDimensionless}, {"m", units::kMetre},
    {"kg", units::kKilogram},     {"s", units::kSecond},
    {"K", units::kKelvin},        {"N", units::kNewton},
    {"J", units::kJoule},         {"W", units::kWatt},
    {"Pa", units::kPascal},       {"W/m^2", units::kHeatFluxDensity},
};

bool parse_rational(const std::string& s, Rational& out) {
  const auto slash = s.find('/');
  try {
    if (slash == std::string::npos) {
      out = Rational{std::stoll(s)};
    } else {
      const auto n = std::stoll(s.substr(0, slash));
      const auto d = std::stoll(s.substr(slash + 1));
      if (d == 0) return false;
      out = Rational{n, d};
    }
  } catch (...) {
    return false;
  }
  return true;
}

/// A unit that has no table symbol is emitted STRUCTURALLY, as a map of the
/// exponents it actually has plus its exact affine part, rather than as a
/// composed string.
///
/// The parent emitted `u.to_string()` here -- a human-readable form like
/// `kg*m^-1*s^-2` that its own parser could not read back. The three-leg
/// round-trip caught it on the first run: a graph with a kilopascal field
/// emitted fine and then failed to parse. An emitter that writes what its parser
/// cannot read is not a front-end, so the fallback is now a form that parses.
YAML::Node units_node(const Units& u) {
  for (const auto& e : kUnitTable) {
    if (u == e.value) return YAML::Node(std::string(e.symbol));  // the readable case
  }
  YAML::Node n(YAML::NodeType::Map);
  const auto put = [&n](const char* key, int value) {
    if (value != 0) n[key] = value;
  };
  put("length", u.length);
  put("mass", u.mass);
  put("time", u.time);
  put("current", u.current);
  put("temperature", u.temperature);
  put("amount", u.amount);
  put("luminosity", u.luminosity);
  if (u.scale != Rational{1}) n["scale"] = u.scale.to_string();
  if (u.offset != Rational{0}) n["offset"] = u.offset.to_string();
  return n;
}

/// Reads either form: a unit expression, or the structural map `units_node`
/// emits.
///
/// BM-12 (review 2026-10-11): the scalar branch used to look symbols up in
/// `kUnitTable` above -- a 10-entry table private to this file -- while
/// semantics.hpp documents `parse_unit_expression()` as "how a YAML or
/// C-ABI caller states units at all". The two vocabularies disagreed:
/// `"m/s"`, `"kg/m^3"`, `"Hz"` and eight more of the parser's 21 symbols
/// parsed everywhere else in the library and were refused here with
/// `E-YAML-UNITS`. Routing through the real parser closes the gap for
/// every symbol it supports (a strict superset of `kUnitTable`'s ten,
/// checked by hand against `unit_symbols`, src/model/semantics.cpp);
/// `units_node()`'s own emission table above is unchanged, so the round
/// trip for every symbol it ever writes stays byte-identical.
///
/// BM-15 (review 2026-10-11): an exponent read here used to narrow with a
/// bare `static_cast<std::int8_t>`, which wraps silently (`length: 300`
/// read back as 44) instead of refusing -- the in-process path refuses the
/// identical case (`detail::narrow_exponent`, fuse/base/units.hpp,
/// `E-BASE-UNITS-EXPONENT`), so a hand-written YAML document should not be
/// able to declare a different physical quantity than the one it wrote.
/// `exponent_in_range` is threaded through rather than thrown: this
/// function's whole contract is "fail by returning false", and the two
/// scale/offset checks below already follow that shape.
bool parse_units(const YAML::Node& n, Units& out) {
  if (n.IsScalar()) {
    if (const auto parsed = parse_unit_expression(n.as<std::string>())) {
      out = *parsed;
      return true;
    }
    return false;
  }
  if (!n.IsMap()) return false;
  Units u{};
  bool exponent_in_range = true;
  const auto get = [&n, &exponent_in_range](const char* key, std::int8_t& into) {
    if (!n[key]) return;
    const int value = n[key].as<int>();
    if (value < -128 || value > 127) { exponent_in_range = false; return; }
    into = static_cast<std::int8_t>(value);
  };
  get("length", u.length);
  get("mass", u.mass);
  get("time", u.time);
  get("current", u.current);
  get("temperature", u.temperature);
  get("amount", u.amount);
  get("luminosity", u.luminosity);
  if (!exponent_in_range) return false;
  if (n["scale"] && !parse_rational(n["scale"].as<std::string>(), u.scale)) return false;
  if (n["offset"] && !parse_rational(n["offset"].as<std::string>(), u.offset)) return false;
  out = u;
  return true;
}

template <class E, std::size_t N>
bool lookup_enum(const std::string& s, const char* const (&names)[N], E& out) {
  for (std::size_t i = 0; i < N; ++i) {
    if (s == names[i]) { out = static_cast<E>(i); return true; }
  }
  return false;
}

// REVIEW.md R-04: deserialize() now refuses an out-of-range AccelKind/
// FilterKind before it can reach a CouplingGraph (descriptor.hpp), but the
// raw add_iteration_region() API has no such gate, so a value indexed here
// can still originate from direct API misuse. Bounds-checking the index
// before it reaches the array is defense in depth against a second unchecked
// source, matching the switch-with-fallback style ids.hpp and iteration.hpp's
// own to_cstring(MeasureKind) already use for the same purpose.
template <class E, std::size_t N>
const char* enum_name(E value, const char* const (&names)[N]) {
  const auto idx = static_cast<std::size_t>(value);
  return idx < N ? names[idx] : "?";
}

const char* const kDiscNames[]    = {"FV", "cGLL", "dGLL", "PointCloud"};
const char* const kConservNames[] = {"None", "Extensive", "Intensive"};
const char* const kIntentNames[]  = {"Provides", "Requires", "Bidirectional"};
const char* const kAccumNames[]   = {"Instant", "Sum", "Mean", "Min", "Max",
                                     "Integral"};
const char* const kBasisNames[]   = {"None", "Constant", "Linear", "DG",
                                     "Spline"};
// Spec Section 5.3. These three must stay in the declaration order of MeasureKind
// (num/convergence.hpp), AccelKind and FilterKind (num/accelerator.hpp):
// lookup_enum() maps a name to its index and casts. A reordered enum with an
// unreordered table here reads "Relative" back as "Absolute", which is a
// different stopping rule with no diagnostic, so each table is pinned by a
// round-trip test over every enumerator in test/unit/yaml_io_test.cpp.
const char* const kMeasureNames[] = {"Relative", "Absolute",
                                     "AbsoluteOrRelative", "ResidualRelative"};
const char* const kAccelNames[]   = {"None",       "Constant", "Aitken",
                                     "Secant",     "Steffensen",
                                     "IQN_ILS",    "IQN_IMVJ",
                                     // S7.2: APPENDED, never inserted -- this
                                     // array is indexed positionally by
                                     // AccelKind's own declaration order
                                     // (this file's own comment above), and
                                     // Newton was itself appended at the end
                                     // of that enum for the identical
                                     // reason.
                                     "Newton"};
const char* const kFilterNames[]  = {"None", "QR1", "QR2", "QR3"};
// S7.2: fuse::LineSearchKind / fuse::EisenstatWalkerChoice
// (fuse/model/iteration.hpp), same positional-index convention as every
// table above.
const char* const kLineSearchNames[] = {"BT", "CP", "Basic"};
const char* const kEwChoiceNames[]   = {"Constant", "Choice1", "Choice2"};
// fuse::NewtonSubstrate (fuse/model/iteration.hpp), same positional-index
// convention and the same spelling rule as every other enum table on this
// page: the C++ enumerator name verbatim. (A first draft wrote these
// lowercase; one rule for all enum keys beats a special case.) The
// examples' command-line flag `--substrate native|petsc` is a separate,
// user-facing spelling and stays lowercase there.
const char* const kNewtonSubstrateNames[] = {"Native", "Petsc"};
// NewtonModel track (2026-10-12): fuse::NewtonNorm / fuse::
// NewtonResidualPath (fuse/model/iteration.hpp), same positional-index
// convention and the same C-enumerator-name-verbatim spelling rule as
// every other enum table on this page.
const char* const kNewtonNormNames[] = {"Plain", "MeasureWeighted"};
const char* const kNewtonResidualPathNames[] = {"Advance", "Direct"};
// Time windows track: must stay in DtPolicy's declaration order
// (fuse/model/iteration.hpp), same reasoning as the three tables above.
const char* const kDtPolicyNames[] = {"Fixed", "Adjustable"};
// Time windows track: must stay in WaveformIteration's declaration order
// (fuse/model/iteration.hpp), same reasoning as the tables above.
const char* const kWaveformIterationNames[] = {"Jacobi", "GaussSeidel"};

// --- rational parsing ------------------------------------------------------


// --- unknown-key enforcement -------------------------------------------
//
// Brief 1.7 states "the reader rejects unknown keys". yaml-cpp only reports
// on keys the caller actually looks up, so this must be an explicit
// allow-list check per map node, called once for every object the schema
// defines a fixed key set for.
std::optional<Failure> check_unknown_keys(const YAML::Node& n,
                                        const std::vector<std::string>& allowed,
                                        const std::string& context) {
  if (!n || !n.IsMap()) return std::nullopt;
  for (const auto& kv : n) {
    const auto key = kv.first.as<std::string>();
    if (std::find(allowed.begin(), allowed.end(), key) == allowed.end()) {
      return Failure{"E-YAML-UNKNOWN-KEY",
                   "unknown key '" + key + "' in " + context};
    }
  }
  return std::nullopt;
}

// BM-2: the single guard every required-key site below calls. Returns a
// Failure naming the key and the context when the key is absent; the
// caller still does its own `.as<T>()` afterwards (the type itself is
// handled by `from_yaml`'s outer try/catch, which turns a wrong-typed
// scalar on this now-confirmed-present key into the same E-YAML-PARSE
// code rather than a raw yaml-cpp exception).
std::optional<Failure> require_key(const YAML::Node& n, const char* key,
                                   const std::string& context) {
  if (n[key]) return std::nullopt;
  return Failure{"E-YAML-PARSE",
               context + " is missing required key '" + key + "'"};
}

using ComponentByName = std::unordered_map<std::string, ComponentId>;

// --- window (de)serialization ------------------------------------------
//
// Windows reference components by name, matching the convention fields and
// channels already use ("fluid.heat_flux"), not by index.
//
// `coupling_period` -- the default, and the only kind Brief 1.7's schema
// example shows -- is emitted as the bare scalar "coupling_period" so the
// checked-in fixture and the brief's exact schema text keep working
// unchanged. `from_component` and `explicit` carry a payload (a component
// name, or a rational length) that a bare scalar cannot hold, so those two
// kinds are emitted as a small map instead. The parser accepts both shapes
// for "coupling_period" (scalar, or {kind: coupling_period}) so a
// hand-written map form is also legal.

// N-base-model-1 (docs/dev/reviews/2026-09-19/pass2/re-base-model.md): a
// FromComponent window's ComponentId is accepted by ChannelBuilder::
// accumulate()/spacetime() with no bounds check, same as the four sites
// R-18 already guards -- window_component_ok() (validate.hpp) is the same
// helper validate() uses for this exact id, reused here rather than
// duplicated so the two call sites cannot drift out of agreement on what
// "in range" means.
std::optional<Failure> emit_window(YAML::Emitter& e, const CouplingGraph& g,
                                    const Window& w) {
  if (!detail::window_component_ok(g, w)) {
    return Failure{"E-YAML-REF", "window references an invalid component"};
  }
  e << YAML::Key << "window" << YAML::Value;
  switch (w.kind) {
    case Window::Kind::FromComponent:
      e << YAML::Flow << YAML::BeginMap
        << YAML::Key << "kind" << YAML::Value << "from_component"
        << YAML::Key << "component" << YAML::Value
        << g.component(w.component).name
        << YAML::EndMap;
      return std::nullopt;
    case Window::Kind::Explicit:
      e << YAML::Flow << YAML::BeginMap
        << YAML::Key << "kind" << YAML::Value << "explicit"
        << YAML::Key << "length" << YAML::Value << w.length.to_string()
        << YAML::EndMap;
      return std::nullopt;
    case Window::Kind::CouplingPeriod:
      e << "coupling_period";
      return std::nullopt;
  }
  // BM-3: there used to be a `default:` here that folded an out-of-range
  // `Window::Kind` into the `CouplingPeriod` case above -- the exact
  // silent-reinterpretation shape R-04 was fixed for, except this was the
  // one enum R-04's own sweep missed. `deserialize()` now range-checks
  // the byte a `Window::Kind` is built from (fuse/model/descriptor.hpp),
  // so every graph reaching this emitter has one of the three
  // enumerators above; reaching here means a `Window` was constructed
  // directly with a `static_cast` outside both the deserializer and the
  // builder factories, a programmer error FUSE_ASSERT is the project's
  // own spelling for.
  FUSE_ASSERT(false, "emit_window: Window::Kind {} is out of range",
             static_cast<int>(w.kind));
  return Failure{"E-YAML-PARSE", "unreachable"};
}

Result<Window> parse_window(const YAML::Node& wn,
                            const ComponentByName& comp_by_name,
                            const std::string& context) {
  if (!wn) return Window::coupling_period();
  if (wn.IsScalar()) {
    if (wn.as<std::string>() == "coupling_period") {
      return Window::coupling_period();
    }
    return Failure{"E-YAML-PARSE",
                 "unknown window scalar '" + wn.as<std::string>() + "' in " +
                     context};
  }
  if (auto e = check_unknown_keys(wn, {"kind", "component", "length"},
                                  context)) {
    return *e;
  }
  if (!wn["kind"]) {
    return Failure{"E-YAML-PARSE", "window missing 'kind' in " + context};
  }
  const auto kind = wn["kind"].as<std::string>();
  if (kind == "coupling_period") return Window::coupling_period();
  if (kind == "from_component") {
    if (!wn["component"]) {
      return Failure{"E-YAML-PARSE",
                   "from_component window missing 'component' in " + context};
    }
    const auto cname = wn["component"].as<std::string>();
    const auto it = comp_by_name.find(cname);
    if (it == comp_by_name.end()) {
      return Failure{"E-YAML-REF",
                   "window references unknown component '" + cname + "'"};
    }
    return Window::from(it->second);
  }
  if (kind == "explicit") {
    if (!wn["length"]) {
      return Failure{"E-YAML-PARSE",
                   "explicit window missing 'length' in " + context};
    }
    Rational len;
    if (!parse_rational(wn["length"].as<std::string>(), len)) {
      return Failure{"E-YAML-PARSE", "bad window length in " + context};
    }
    return Window::explicit_length(len);
  }
  return Failure{"E-YAML-PARSE", "unknown window kind '" + kind + "' in " + context};
}

}  // namespace


/// The `launch:` block, written identically for a component and for a
/// coupler. A coupler is a component-array entry whose kind is Coupler, so
/// its launch record is the same record; it is written from the couplers
/// section only because that is where the entry lives in the file.
void write_launch(YAML::Emitter& e, const LaunchSpec& launch,
                  const char* mode_name) {
  e << YAML::Key << "launch" << YAML::Value << YAML::BeginMap
    << YAML::Key << "mode" << YAML::Value << mode_name;
  if (launch.mode == LaunchMode::Mpmd) {
    e << YAML::Key << "appnum" << YAML::Value << launch.appnum;
  }
  if (launch.mode == LaunchMode::Connect) {
    e << YAML::Key << "is_acceptor" << YAML::Value << launch.is_acceptor;
    if (!launch.port_file.empty()) {
      e << YAML::Key << "port_file" << YAML::Value << launch.port_file;
    }
    if (!launch.service_name.empty()) {
      e << YAML::Key << "service_name" << YAML::Value << launch.service_name;
    }
  }
  // S6 phase 4, "Adapter host" track.
  if (launch.mode == LaunchMode::Plugin) {
    e << YAML::Key << "plugin_path" << YAML::Value << launch.plugin_path;
  }
  e << YAML::EndMap;
}

/// `mode_name` for an emitter, or nullptr when there is nothing to write
/// (an in-process entry carries no `launch:` block at all, which is what
/// keeps every existing file byte-identical).
const char* launch_mode_name(const LaunchSpec& launch) {
  switch (launch.mode) {
    case LaunchMode::InProcess: return nullptr;
    case LaunchMode::Mpmd:      return "mpmd";
    case LaunchMode::Connect:   return "connect";
    case LaunchMode::Plugin:    return "plugin";
    case LaunchMode::LowFive:   return "lowfive";
  }
  return nullptr;
}

/// The reader's half. `who` names the entry in any diagnostic.
[[nodiscard]] std::optional<Failure> read_launch(const YAML::Node& ln,
                                                 const std::string& who,
                                                 LaunchSpec& launch) {
  if (auto e = check_unknown_keys(
          ln, {"mode", "appnum", "is_acceptor", "port_file", "service_name", "plugin_path"},
          who + " launch")) {
    return *e;
  }
  if (!ln["mode"]) {
    return Failure{"E-YAML-PARSE", who + " launch missing 'mode'"};
  }
  const auto mode = ln["mode"].as<std::string>();
  if (mode == "mpmd") {
    launch.mode = LaunchMode::Mpmd;
    launch.appnum = ln["appnum"] ? ln["appnum"].as<int>() : -1;
  } else if (mode == "connect") {
    launch.mode = LaunchMode::Connect;
    launch.is_acceptor = ln["is_acceptor"] && ln["is_acceptor"].as<bool>();
    if (ln["port_file"])    launch.port_file = ln["port_file"].as<std::string>();
    if (ln["service_name"]) launch.service_name = ln["service_name"].as<std::string>();
  } else if (mode == "plugin") {
    // S6 phase 4, "Adapter host" track.
    launch.mode = LaunchMode::Plugin;
    if (ln["plugin_path"]) launch.plugin_path = ln["plugin_path"].as<std::string>();
  } else if (mode == "lowfive") {
    // S6 phase 3, "LowFive" track, Unit 3. No extra LaunchSpec member of
    // its own -- see launch.hpp's own doc comment on this value.
    launch.mode = LaunchMode::LowFive;
  } else {
    return Failure{"E-YAML-PARSE",
                   who + " has unknown launch mode '" + mode + "'"};
  }
  return std::nullopt;
}

Result<std::string> to_yaml(const CouplingGraph& g) {
  YAML::Emitter e;
  e << YAML::BeginMap;
  e << YAML::Key << "version" << YAML::Value << kSchemaVersion;

  e << YAML::Key << "meshes" << YAML::Value << YAML::BeginSeq;
  for (std::uint32_t i = 0; i < g.mesh_count(); ++i) {
    const auto& m = g.mesh(MeshId{i});
    e << YAML::BeginMap << YAML::Key << "name" << YAML::Value << m.name
      << YAML::Key << "source" << YAML::Value << m.source << YAML::EndMap;
  }
  e << YAML::EndSeq;

  e << YAML::Key << "regions" << YAML::Value << YAML::BeginSeq;
  for (std::uint32_t i = 0; i < g.region_count(); ++i) {
    const auto& r = g.region(RegionId{i});
    // REVIEW.md R-18: unlike serialize() (descriptor.hpp), which never
    // dereferences a foreign id through an .at()-based accessor, to_yaml()
    // needs the mesh's NAME, not its raw id, for readable YAML -- and
    // add_region() accepts any MeshId with no bounds check at construction
    // time. Refusing here is what routes a bad reference through the
    // Result<std::string> this function promises, instead of letting
    // CouplingGraph::mesh()'s std::vector::at() throw a raw
    // std::out_of_range past the boundary.
    if (!r.mesh.valid() || r.mesh.value() >= g.mesh_count()) {
      return Failure{"E-YAML-REF",
                     "region '" + r.name + "' references an invalid mesh"};
    }
    e << YAML::BeginMap
      << YAML::Key << "name" << YAML::Value << r.name
      << YAML::Key << "mesh" << YAML::Value << g.mesh(r.mesh).name
      << YAML::Key << "entity_set" << YAML::Value << r.entity_set
      << YAML::Key << "dimension" << YAML::Value << r.dimension
      << YAML::EndMap;
  }
  e << YAML::EndSeq;

  // Couplers are ComponentKind::Coupler entries of the SAME components_
  // array (component.hpp's CouplerDecl doc comment), but the text form
  // gives them their own top-level list: a coupler owns no fields and no
  // interval, so nesting it under "components:" would either force those
  // two keys to mean nothing on some entries or need a "kind:" discriminant
  // on every entry just to tell the two apart. A separate list needs
  // neither, and reads as what a coupler is -- a placement, not a physics
  // component.
  e << YAML::Key << "components" << YAML::Value << YAML::BeginSeq;
  for (std::uint32_t i = 0; i < g.component_count(); ++i) {
    const ComponentId cid{i};
    const auto& c = g.component(cid);
    if (c.kind == ComponentKind::Coupler) continue;
    e << YAML::BeginMap
      << YAML::Key << "name" << YAML::Value << c.name
      << YAML::Key << "ranks" << YAML::Value;
    // A contiguous placement keeps the {first, count} form, which is what
    // almost every setup is and is far more readable than [0, 1, 2, ..., 63].
    // A strided or shared placement emits the explicit list. Both parse, and
    // the round-trip test covers each.
    if (c.ranks.is_contiguous() && !c.ranks.empty()) {
      e << YAML::Flow << YAML::BeginMap
        << YAML::Key << "first" << YAML::Value << c.ranks.min_rank()
        << YAML::Key << "count" << YAML::Value << static_cast<int>(c.ranks.size())
        << YAML::EndMap;
    } else {
      e << YAML::Flow << YAML::BeginSeq;
      for (const int rank : c.ranks) e << rank;
      e << YAML::EndSeq;
    }
    e
      << YAML::Key << "interval" << YAML::Value << c.interval.to_string();
    // Emitted only when non-default, like ComponentKind above: the common
    // case (today's only case, before this track) is InProcess, and a file
    // written before S6 phase 3 round-trips with no "launch:" key at all.
    if (c.launch.mode != LaunchMode::InProcess) {
      const char* mode_name = "connect";
      if (c.launch.mode == LaunchMode::Mpmd) mode_name = "mpmd";
      else if (c.launch.mode == LaunchMode::Plugin) mode_name = "plugin";
      else if (c.launch.mode == LaunchMode::LowFive) mode_name = "lowfive";
      write_launch(e, c.launch, mode_name);
    }
    e << YAML::Key << "fields" << YAML::Value << YAML::BeginSeq;
    for (const FieldId fid : c.fields) {
      const auto& f = g.field(fid);
      // N-base-model-1: f.region is a raw RegionId add_field() accepts with
      // no bounds check (add_field() only asserts the owning component is
      // in range); g.region() below is .at()-based, same class of gap as
      // the region->mesh check above and the four sites R-18 already
      // guards.
      if (!f.region.valid() || f.region.value() >= g.region_count()) {
        return Failure{"E-YAML-REF",
                       "field '" + f.name + "' references an invalid region"};
      }
      e << YAML::BeginMap
        << YAML::Key << "name" << YAML::Value << f.name
        // The order fields were declared in, which is NOT the order they
        // appear here: the text nests fields under their component, so a graph
        // that interleaved two components' fields would come back regrouped
        // and every field id would shift. Ids are what channels and merge
        // rules refer to, and what two ranks compare byte-for-byte, so the
        // order is part of the graph and has to survive the text.
        << YAML::Key << "index" << YAML::Value << fid.value()
        << YAML::Key << "region" << YAML::Value << g.region(f.region).name
        << YAML::Key << "disc" << YAML::Value << to_cstring(f.disc)
        << YAML::Key << "order" << YAML::Value << f.order
        << YAML::Key << "components" << YAML::Value << f.components
        << YAML::Key << "units" << YAML::Value << units_node(f.units)
        << YAML::Key << "conservation" << YAML::Value << to_cstring(f.conservation)
        << YAML::Key << "intent" << YAML::Value << to_cstring(f.intent);
      // S6 phase 3 ("LowFive" track), Unit 3. Emitted only when set, like
      // launch above: every field before this track round-trips with no
      // "hdf5_path" key at all.
      if (f.hdf5_path.has_value()) {
        e << YAML::Key << "hdf5_path" << YAML::Value << *f.hdf5_path;
      }
      e << YAML::EndMap;
    }
    e << YAML::EndSeq << YAML::EndMap;
  }
  e << YAML::EndSeq;

  e << YAML::Key << "couplers" << YAML::Value << YAML::BeginSeq;
  for (std::uint32_t i = 0; i < g.component_count(); ++i) {
    const auto& c = g.component(ComponentId{i});
    if (c.kind != ComponentKind::Coupler) continue;
    // `index` is the coupler's position in the component array. Component
    // index is the placement space (every placed_view caller, the IR's
    // placements, the canonical bytes), so a coupler authored between two
    // physics components must come back between them, not after all of
    // them; the two-section layout alone loses that.
    e << YAML::BeginMap << YAML::Key << "name" << YAML::Value << c.name
      << YAML::Key << "index" << YAML::Value << i
      << YAML::Key << "ranks" << YAML::Value;
    if (c.ranks.is_contiguous() && !c.ranks.empty()) {
      e << YAML::Flow << YAML::BeginMap
        << YAML::Key << "first" << YAML::Value << c.ranks.min_rank()
        << YAML::Key << "count" << YAML::Value << static_cast<int>(c.ranks.size())
        << YAML::EndMap;
    } else {
      e << YAML::Flow << YAML::BeginSeq;
      for (const int rank : c.ranks) e << rank;
      e << YAML::EndSeq;
    }
    // A coupler launched as its own program says so, exactly as a
    // component does. Omitted for an in-process coupler, so every file
    // written before couplers could carry this is byte-identical still.
    if (const char* mode_name = launch_mode_name(c.launch)) {
      write_launch(e, c.launch, mode_name);
    }
    e << YAML::EndMap;
  }
  e << YAML::EndSeq;

  e << YAML::Key << "channels" << YAML::Value << YAML::BeginSeq;
  for (std::uint32_t i = 0; i < g.channel_count(); ++i) {
    const auto& c = g.channel_at(ChannelId{i});
    // REVIEW.md R-18: c.source/c.target are raw FieldIds add_channel()
    // accepts with no bounds check; qualified() below dereferences both
    // through g.field()/g.owner_of(), each an .at()-based accessor. Once a
    // field id is confirmed in range, owner_of() and component() on its
    // owner are always safe (add_field() populates field_owner_ 1:1 with
    // fields_ and asserts the owner is in range at that time).
    if (!c.source.valid() || c.source.value() >= g.field_count() ||
        !c.target.valid() || c.target.value() >= g.field_count()) {
      return Failure{"E-YAML-REF",
                     "channel '" + c.name + "' references an invalid field"};
    }
    const auto qualified = [&](FieldId f) {
      return g.component(g.owner_of(f)).name + "." + g.field(f).name;
    };
    e << YAML::BeginMap
      << YAML::Key << "name" << YAML::Value << c.name
      << YAML::Key << "source" << YAML::Value << qualified(c.source)
      << YAML::Key << "target" << YAML::Value << qualified(c.target)
      << YAML::Key << "accumulate" << YAML::Value << YAML::Flow << YAML::BeginMap
      << YAML::Key << "op" << YAML::Value << to_cstring(c.accumulate.op);
    if (auto fail = emit_window(e, g, c.accumulate.window)) return *fail;
    e << YAML::EndMap;

    if (c.remap.flags != RemapFlags::None || !c.remap.method_override.empty() ||
        c.remap.remap_on.valid()) {
      e << YAML::Key << "remap" << YAML::Value << YAML::BeginMap;
      if (c.remap.flags != RemapFlags::None) {
        e << YAML::Key << "flags" << YAML::Value << YAML::Flow << YAML::BeginSeq;
        if (has(c.remap.flags, RemapFlags::Conservative)) e << "Conservative";
        if (has(c.remap.flags, RemapFlags::Monotone))     e << "Monotone";
        if (has(c.remap.flags, RemapFlags::HighOrder))    e << "HighOrder";
        e << YAML::EndSeq;
      }
      if (!c.remap.method_override.empty()) {
        e << YAML::Key << "method" << YAML::Value << c.remap.method_override;
      }
      if (c.remap.remap_on.valid()) {
        // REVIEW.md R-18's pattern: remap_on() accepts any ComponentId with
        // no bounds check, so the emitter -- which needs the NAME, not the
        // raw id -- must refuse a foreign one through Result<std::string>
        // rather than let g.component() throw past this function's caller.
        if (c.remap.remap_on.value() >= g.component_count()) {
          return Failure{"E-YAML-REF",
                         "channel '" + c.name +
                             "' has a remap_on that references an invalid "
                             "component"};
        }
        e << YAML::Key << "remap_on" << YAML::Value
          << g.component(c.remap.remap_on).name;
      }
      e << YAML::EndMap;
    }
    if (c.spacetime.basis != TemporalBasis::None) {
      e << YAML::Key << "spacetime" << YAML::Value << YAML::Flow << YAML::BeginMap
        << YAML::Key << "basis" << YAML::Value << to_cstring(c.spacetime.basis)
        << YAML::Key << "degree" << YAML::Value << c.spacetime.degree;
      if (auto fail = emit_window(e, g, c.spacetime.window)) return *fail;
      e << YAML::EndMap;
    }
    if (c.relax.enabled) e << YAML::Key << "relax" << YAML::Value << c.relax.omega;
    if (c.extrapolate.enabled) {
      e << YAML::Key << "extrapolate" << YAML::Value << c.extrapolate.order;
    }
    if (c.lag != 0) e << YAML::Key << "lag" << YAML::Value << c.lag;
    if (!c.time_level_mismatch_reason.empty()) {
      e << YAML::Key << "allow_time_level_mismatch" << YAML::Value
        << c.time_level_mismatch_reason;
    }
    e << YAML::EndMap;
  }
  e << YAML::EndSeq;

  // The merge table. Emitted even though it is usually empty, because the
  // three-leg round trip compares canonical bytes: a field the text form drops
  // is a field a YAML-authored graph silently loses.
  if (g.merge_count() > 0) {
    e << YAML::Key << "merges" << YAML::Value << YAML::BeginSeq;
    for (std::size_t i = 0; i < g.merge_count(); ++i) {
      const auto& entry = g.merge_at(i);
      // REVIEW.md R-18: entry.first is a raw field id merge() accepts with
      // no bounds check; g.field()/g.owner_of() below are .at()-based.
      if (entry.first >= g.field_count()) {
        return Failure{"E-YAML-REF",
                       "a merge entry references invalid field id " +
                           std::to_string(entry.first)};
      }
      e << YAML::BeginMap
        << YAML::Key << "field" << YAML::Value << g.field(FieldId{entry.first}).name
        << YAML::Key << "component" << YAML::Value
        << g.component(g.owner_of(FieldId{entry.first})).name
        << YAML::Key << "rule" << YAML::Value << std::string(merge_op_name(entry.second.op))
        << YAML::EndMap;
    }
    e << YAML::EndSeq;
  }

  e << YAML::Key << "iteration_regions" << YAML::Value << YAML::BeginSeq;
  for (std::uint32_t i = 0; i < g.iteration_region_count(); ++i) {
    const auto& r = g.iteration_region(IterationRegionIdx{i});
    e << YAML::BeginMap
      << YAML::Key << "name" << YAML::Value << r.name
      << YAML::Key << "channels" << YAML::Value << YAML::Flow << YAML::BeginSeq;
    // REVIEW.md R-18: r.channels is a raw ChannelId list add_iteration_region()
    // accepts with no bounds check; g.channel_at() is .at()-based.
    for (const ChannelId c : r.channels) {
      if (!c.valid() || c.value() >= g.channel_count()) {
        return Failure{"E-YAML-REF",
                       "iteration region '" + r.name +
                           "' references an invalid channel"};
      }
      e << g.channel_at(c).name;
    }
    e << YAML::EndSeq
      << YAML::Key << "max_iterations" << YAML::Value << r.max_iterations;
    // Emitted unconditionally, empty sequence and all. An omitted
    // `convergence` reads back as an empty criterion, which is exactly the
    // state E-ITER-NOCONVERGE exists to refuse; making the emitter silent
    // about it would make a dropped criterion indistinguishable from an
    // authored-empty one in the artifact a user diffs.
    e << YAML::Key << "convergence" << YAML::Value << YAML::BeginSeq;
    for (const ConvergenceMeasure& m : r.convergence.measures()) {
      e << YAML::Flow << YAML::BeginMap
        << YAML::Key << "kind" << YAML::Value
        << enum_name(m.kind, kMeasureNames)
        << YAML::Key << "tolerance" << YAML::Value << m.tolerance
        << YAML::Key << "field" << YAML::Value << m.field
        << YAML::EndMap;
    }
    e << YAML::EndSeq;
    e << YAML::Key << "acceleration" << YAML::Value << YAML::BeginMap
      << YAML::Key << "kind" << YAML::Value
      << enum_name(r.acceleration.kind, kAccelNames)
      << YAML::Key << "initial_relaxation" << YAML::Value
      << r.acceleration.initial_relaxation
      << YAML::Key << "omega" << YAML::Value << r.acceleration.omega
      << YAML::Key << "reuse" << YAML::Value << r.acceleration.reuse
      << YAML::Key << "filter" << YAML::Value
      << enum_name(r.acceleration.filter, kFilterNames)
      << YAML::Key << "limit" << YAML::Value << r.acceleration.limit
      << YAML::Key << "scale_fields" << YAML::Value
      << r.acceleration.scale_fields
      << YAML::EndMap;
    // S7.1/S7.2's own capability seam. Emitted ONLY when set (unlike
    // `convergence` above): `false` is every pre-S7.1 region's own
    // implicit state, including every non-Newton region a concurrent
    // track's own committed YAML already carries (examples/brusselator/
    // participants/reference_mpmd_3.yaml, found drifting the moment this
    // was emitted unconditionally) -- an omitted key here reads back as
    // exactly that default, so there is nothing for a human to lose by it
    // staying silent. Still enough to round-trip a Newton region that
    // sets either flag: the read side below applies the SAME auto-trigger
    // `IterationBuilder::newton()`/`::accelerate()` already guarantee in
    // C++, so `acceleration.kind: Newton` alone (with neither key present)
    // still comes back requiring the residual capability.
    if (r.requires_residual_capability) {
      e << YAML::Key << "requires_residual_capability" << YAML::Value << true;
    }
    if (r.requires_linearization_capability) {
      e << YAML::Key << "requires_linearization_capability" << YAML::Value << true;
    }
    // S7.2's own Newton options. Conditional on the kind actually being
    // Newton (unlike the two booleans above): twenty-odd fields that mean
    // nothing for every OTHER accelerator would otherwise clutter every
    // region's YAML, the same reasoning `window_scope`'s own conditional
    // emission below already follows for an opt-in block.
    if (r.acceleration.kind == AccelKind::Newton) {
      const NewtonConfig& nc = r.newton_config;
      e << YAML::Key << "newton_config" << YAML::Value << YAML::BeginMap
        << YAML::Key << "line_search_kind" << YAML::Value
        << enum_name(nc.line_search_kind, kLineSearchNames)
        << YAML::Key << "forcing_choice" << YAML::Value
        << enum_name(nc.forcing_choice, kEwChoiceNames)
        << YAML::Key << "forcing_gamma" << YAML::Value << nc.forcing_gamma
        << YAML::Key << "forcing_alpha" << YAML::Value << nc.forcing_alpha
        << YAML::Key << "eta_initial" << YAML::Value << nc.eta_initial
        << YAML::Key << "eta_max" << YAML::Value << nc.eta_max
        << YAML::Key << "eta_min" << YAML::Value << nc.eta_min
        << YAML::Key << "forcing_safeguard_threshold" << YAML::Value
        << nc.forcing_safeguard_threshold
        << YAML::Key << "forcing_eta_constant" << YAML::Value
        << nc.forcing_eta_constant
        << YAML::Key << "atol" << YAML::Value << nc.atol
        << YAML::Key << "rtol" << YAML::Value << nc.rtol
        << YAML::Key << "stol" << YAML::Value << nc.stol
        << YAML::Key << "divergence_tolerance" << YAML::Value
        << nc.divergence_tolerance
        << YAML::Key << "max_nonlinear_iterations" << YAML::Value
        << nc.max_nonlinear_iterations
        << YAML::Key << "max_function_evaluations" << YAML::Value
        << static_cast<std::uint64_t>(nc.max_function_evaluations)
        << YAML::Key << "armijo_alpha" << YAML::Value << nc.armijo_alpha
        << YAML::Key << "lambda_min" << YAML::Value << nc.lambda_min
        << YAML::Key << "max_backtracks" << YAML::Value << nc.max_backtracks
        << YAML::Key << "secant_tol" << YAML::Value << nc.secant_tol
        << YAML::Key << "max_secant_steps" << YAML::Value
        << nc.max_secant_steps
        << YAML::Key << "damping" << YAML::Value << nc.damping
        << YAML::Key << "gmres_restart" << YAML::Value << nc.gmres_restart
        << YAML::Key << "gmres_max_iterations" << YAML::Value
        << static_cast<std::uint64_t>(nc.gmres_max_iterations)
        << YAML::Key << "preconditioner_lag" << YAML::Value
        << nc.preconditioner_lag
        << YAML::Key << "use_block_jacobi_preconditioner" << YAML::Value
        << nc.use_block_jacobi_preconditioner
        << YAML::Key << "substrate" << YAML::Value
        << enum_name(nc.substrate, kNewtonSubstrateNames)
        << YAML::Key << "norm" << YAML::Value
        << enum_name(nc.norm, kNewtonNormNames)
        << YAML::Key << "residual_path" << YAML::Value
        << enum_name(nc.residual_path, kNewtonResidualPathNames)
        << YAML::EndMap;
    }
    if (r.window_scope.has_value()) {
      const CouplingWindowScope& scope = *r.window_scope;
      e << YAML::Key << "window_scope" << YAML::Value << YAML::BeginMap;
      if (scope.window_component.has_value()) {
        e << YAML::Key << "window_component" << YAML::Value
          << *scope.window_component;
      }
      if (scope.explicit_interval.has_value()) {
        e << YAML::Key << "explicit_interval" << YAML::Value
          << scope.explicit_interval->to_string();
      }
      e << YAML::Key << "order" << YAML::Value << scope.order;
      e << YAML::Key << "policies" << YAML::Value << YAML::BeginSeq;
      for (const WindowMemberPolicy& p : scope.policies) {
        e << YAML::Flow << YAML::BeginMap
          << YAML::Key << "component" << YAML::Value << p.component
          << YAML::Key << "policy" << YAML::Value << to_cstring(p.policy)
          << YAML::EndMap;
      }
      e << YAML::EndSeq;
      e << YAML::Key << "iteration" << YAML::Value
        << to_cstring(scope.iteration);
      e << YAML::EndMap;
    }
    e << YAML::EndMap;
  }
  e << YAML::EndSeq << YAML::EndMap;

  if (!e.good()) return Failure{"E-YAML-EMIT", e.GetLastError()};
  return std::string{e.c_str()};
}

/// BM-2 (review 2026-10-11): the parse body below used to read every
/// required key with a bare `node["key"].as<T>()` -- no presence guard, no
/// surrounding `try`. yaml-cpp throws `YAML::BadConversion`/`InvalidNode`
/// for a missing, misspelled or wrong-typed scalar, and that exception
/// used to escape a function whose declared contract is `Result<
/// CouplingGraph>`, past every caller that checks `.ok()` instead of
/// wrapping the call in a `try`. Each required key below is now guarded
/// explicitly, by name and by the entry it belongs to, so the common
/// mistake (a typo'd key, a map where a scalar was expected) reports
/// `E-YAML-PARSE` naming the exact key; `from_yaml_body` is additionally
/// wrapped in a `try` that translates ANY exception still reaching it --
/// a wrong-typed scalar on an OPTIONAL key, or a `fuse::Error` escaping a
/// helper such as `RankSet`'s own range check (BM-14) -- into the same
/// `E-YAML-PARSE` failure rather than letting it propagate as a non-FUSE,
/// undeclared exception.
namespace {
Result<CouplingGraph> from_yaml_body(const YAML::Node& root) {
  if (auto e = check_unknown_keys(
          root,
          {"version", "meshes", "regions", "components", "couplers",
           "channels", "iteration_regions", "merges"},
          "top-level document")) {
    return *e;
  }

  if (!root["version"] || root["version"].as<int>() != kSchemaVersion) {
    return Failure{"E-YAML-VERSION",
                 "expected schema version " + std::to_string(kSchemaVersion)};
  }

  CouplingGraph g;
  std::unordered_map<std::string, MeshId>     mesh_by_name;
  std::unordered_map<std::string, RegionId>   region_by_name;
  ComponentByName                             component_by_name;
  std::unordered_map<std::string, FieldId>    field_by_qualified;
  std::unordered_map<std::string, ChannelId>  channel_by_name;

  for (const auto& n : root["meshes"]) {
    if (auto e = check_unknown_keys(n, {"name", "source"}, "mesh entry")) {
      return *e;
    }
    if (auto e = require_key(n, "name", "mesh entry")) return *e;
    const auto name = n["name"].as<std::string>();
    // BM-16: name-keyed maps collapse a duplicate silently (last wins),
    // which is how a region naming this mesh by name could resolve to a
    // DIFFERENT mesh than the one the author thought they had declared.
    // Refused here, at the point the duplicate first becomes ambiguous,
    // rather than left to silently rewrite the graph on a round trip.
    if (mesh_by_name.count(name)) {
      return Failure{"E-YAML-DUP", "duplicate mesh name '" + name + "'"};
    }
    mesh_by_name[name] = g.add_mesh(
        MeshDecl{.name = name,
                 .source = n["source"] ? n["source"].as<std::string>() : ""});
  }

  for (const auto& n : root["regions"]) {
    if (auto e = check_unknown_keys(
            n, {"name", "mesh", "entity_set", "dimension"}, "region entry")) {
      return *e;
    }
    if (auto e = require_key(n, "mesh", "region entry")) return *e;
    const auto mesh_name = n["mesh"].as<std::string>();
    const auto it = mesh_by_name.find(mesh_name);
    if (it == mesh_by_name.end()) {
      return Failure{"E-YAML-REF", "unknown mesh '" + mesh_name + "'"};
    }
    if (auto e = require_key(n, "name", "region entry")) return *e;
    const auto name = n["name"].as<std::string>();
    // BM-16: same reasoning as the mesh table above -- a channel or field
    // resolving this region's name must not be able to silently land on
    // whichever of two identically-named regions happened to be declared
    // last.
    if (region_by_name.count(name)) {
      return Failure{"E-YAML-DUP", "duplicate region name '" + name + "'"};
    }
    region_by_name[name] = g.add_region(Region{
        .name = name,
        .mesh = it->second,
        .entity_set = n["entity_set"] ? n["entity_set"].as<std::string>() : "",
        .dimension = n["dimension"] ? n["dimension"].as<int>() : -1});
  }

  struct PendingField {
    std::uint32_t index;
    ComponentId owner;
    std::string component_name;
    FieldSpec spec;
  };
  std::vector<PendingField> pending_fields;

  struct PendingCoupler {
    std::uint32_t index;
    std::string   name;
    CouplerDecl   decl;
  };
  std::vector<PendingCoupler> pending_couplers;
  for (const auto& n : root["couplers"]) {
    if (auto e = check_unknown_keys(n, {"name", "index", "ranks", "launch"}, "coupler entry")) {
      return *e;
    }
    if (auto e = require_key(n, "name", "coupler entry")) return *e;
    CouplerDecl c;
    c.name = n["name"].as<std::string>();
    if (n["ranks"]) {
      if (n["ranks"].IsSequence()) {
        std::vector<int> ranks;
        for (const auto& r : n["ranks"]) ranks.push_back(r.as<int>());
        c.ranks = RankSet{std::move(ranks)};
      } else {
        const std::string ranks_ctx = "coupler '" + c.name + "' ranks";
        if (auto e = check_unknown_keys(n["ranks"], {"first", "count"},
                                        ranks_ctx)) {
          return *e;
        }
        if (auto e = require_key(n["ranks"], "first", ranks_ctx)) return *e;
        if (auto e = require_key(n["ranks"], "count", ranks_ctx)) return *e;
        c.ranks = RankSet{n["ranks"]["first"].as<int>(),
                          n["ranks"]["count"].as<int>()};
      }
    }
    if (n["launch"]) {
      if (auto e = read_launch(n["launch"], "coupler '" + c.name + "'", c.launch)) {
        return *e;
      }
    }
    // Couplers share the SAME name -> id table channels resolve `source`,
    // `target` and (below) `remap_on` through, so a channel can name a
    // coupler exactly as it names a physics component.
    //
    // The name is read into a local BEFORE the std::move(c) below: since
    // C++17, `a[k] = f(std::move(c))` sequences the right operand first, so
    // `c.name` used directly as the map key would read a moved-from string.
    const std::string coupler_name = c.name;
    // A text without `index` (hand-written, most likely) appends its
    // couplers after every physics component, which is what the writer
    // produced before `index` existed.
    const std::uint32_t at = n["index"] ? n["index"].as<std::uint32_t>()
                                        : std::numeric_limits<std::uint32_t>::max();
    pending_couplers.push_back({at, coupler_name, std::move(c)});
  }
  std::stable_sort(pending_couplers.begin(), pending_couplers.end(),
                   [](const PendingCoupler& a, const PendingCoupler& b) { return a.index < b.index; });
  std::size_t next_coupler = 0;
  // Adds every pending coupler whose declared position is the next slot of
  // the component array, so interleaving with add_component() below
  // reproduces the authored order exactly.
  const auto flush_couplers = [&]() {
    while (next_coupler < pending_couplers.size() &&
           pending_couplers[next_coupler].index <= g.component_count()) {
      PendingCoupler& pc = pending_couplers[next_coupler++];
      component_by_name[pc.name] = g.add_coupler(std::move(pc.decl));
    }
  };

  for (const auto& n : root["components"]) {
    if (auto e = check_unknown_keys(
            n, {"name", "ranks", "interval", "fields", "launch"}, "component entry")) {
      return *e;
    }
    if (auto e = require_key(n, "name", "component entry")) return *e;
    ComponentDecl c;
    c.name = n["name"].as<std::string>();
    if (n["ranks"]) {
      if (n["ranks"].IsSequence()) {
        std::vector<int> ranks;
        for (const auto& r : n["ranks"]) ranks.push_back(r.as<int>());
        // RankSet sorts and de-duplicates, so a hand-written file may list
        // ranks in launcher order without that changing the graph.
        c.ranks = RankSet{std::move(ranks)};
      } else {
        const std::string ranks_ctx = "component '" + c.name + "' ranks";
        if (auto e = check_unknown_keys(n["ranks"], {"first", "count"},
                                        ranks_ctx)) {
          return *e;
        }
        if (auto e = require_key(n["ranks"], "first", ranks_ctx)) return *e;
        if (auto e = require_key(n["ranks"], "count", ranks_ctx)) return *e;
        c.ranks = RankSet{n["ranks"]["first"].as<int>(),
                          n["ranks"]["count"].as<int>()};
      }
    }
    if (n["interval"]) {
      if (!parse_rational(n["interval"].as<std::string>(), c.interval)) {
        return Failure{"E-YAML-PARSE",
                     "bad interval for component '" + c.name + "'"};
      }
    }
    if (n["launch"]) {
      if (auto e = read_launch(n["launch"], "component '" + c.name + "'", c.launch)) {
        return *e;
      }
    }
    flush_couplers();
    const ComponentId cid = g.add_component(c);
    component_by_name[c.name] = cid;

    for (const auto& fn : n["fields"]) {
      if (auto e = check_unknown_keys(
              fn,
              {"name", "region", "disc", "order", "components", "units",
               "conservation", "intent", "index", "hdf5_path"},
              "field entry in component '" + c.name + "'")) {
        return *e;
      }
      if (auto e = require_key(fn, "name", "field entry in component '" + c.name + "'")) {
        return *e;
      }
      FieldSpec f;
      f.name = fn["name"].as<std::string>();
      if (auto e = require_key(fn, "region", "field '" + f.name + "' in component '" + c.name + "'")) {
        return *e;
      }
      const auto rname = fn["region"].as<std::string>();
      const auto rit = region_by_name.find(rname);
      if (rit == region_by_name.end()) {
        return Failure{"E-YAML-REF", "unknown region '" + rname + "'"};
      }
      f.region = rit->second;
      if (fn["disc"] &&
          !lookup_enum(fn["disc"].as<std::string>(), kDiscNames, f.disc)) {
        return Failure{"E-YAML-PARSE",
                     "unknown discretization '" + fn["disc"].as<std::string>() + "'"};
      }
      if (fn["order"])      f.order = fn["order"].as<int>();
      if (fn["components"]) f.components = fn["components"].as<int>();
      if (fn["units"]) {
        if (!parse_units(fn["units"], f.units)) {
          // BM-15: `parse_units()` fails for two different node shapes --
          // an unrecognised SCALAR symbol, and a structural MAP with an
          // out-of-range exponent (this function's own new check) -- and
          // only the first is safely describable with `.as<std::string>()`;
          // calling that on the second (a map) throws, which used to turn
          // this E-YAML-UNITS failure into an unrelated E-YAML-PARSE one
          // via from_yaml()'s outer catch.
          const std::string detail = fn["units"].IsScalar()
              ? "unknown unit symbol '" + fn["units"].as<std::string>() + "'"
              : "unparseable unit specification for field '" + f.name + "'";
          return Failure{"E-YAML-UNITS", detail};
        }
        // N-base-model-2: R-19 gated a non-positive Units::scale at the
        // binary descriptor boundary (descriptor.hpp) and at
        // check_units()/validate_field_semantics(), but from_yaml() -- an
        // equally real trust boundary per testing.md's three-leg
        // round-trip and porting.md's front-end list -- parsed "scale"
        // with no sign or zero check at all. Same rule, same message
        // shape, matching E-DESC-UNITS-SCALE's code family so the two
        // front ends report the identical defect the same way.
        if (!(f.units.scale > Rational{0})) {
          return Failure{"E-YAML-UNITS-SCALE",
                       "field '" + f.name + "' has non-positive unit scale " +
                           f.units.scale.to_string()};
        }
      }
      if (fn["conservation"] &&
          !lookup_enum(fn["conservation"].as<std::string>(), kConservNames,
                       f.conservation)) {
        return Failure{"E-YAML-PARSE", "unknown conservation class"};
      }
      if (fn["intent"] &&
          !lookup_enum(fn["intent"].as<std::string>(), kIntentNames, f.intent)) {
        return Failure{"E-YAML-PARSE", "unknown intent"};
      }
      // S6 phase 3 ("LowFive" track), Unit 3. validate() (fuse/model/
      // validate.hpp), not here, enforces the two graph-wide rules this
      // can set up (a LowFive-launched component must exist; no two
      // fields may share a path) -- from_yaml() only has one field's own
      // text in view at this point, same reason "conservation"/"intent"
      // above are not cross-checked here either.
      if (fn["hdf5_path"]) f.hdf5_path = fn["hdf5_path"].as<std::string>();
      // Collected, not added: the fields are added below in declaration
      // order so the ids match the graph this text came from.
      pending_fields.push_back(PendingField{
          fn["index"] ? fn["index"].as<std::uint32_t>()
                      : static_cast<std::uint32_t>(pending_fields.size()),
          cid, c.name, f});
    }
  }


  // Couplers declared after the last physics component, or without `index`.
  while (next_coupler < pending_couplers.size()) {
    PendingCoupler& pc = pending_couplers[next_coupler++];
    component_by_name[pc.name] = g.add_coupler(std::move(pc.decl));
  }

  // Declaration order. A text without `index` (hand-written, most likely)
  // keeps the order it was read in, which is the best information available.
  std::stable_sort(pending_fields.begin(), pending_fields.end(),
                   [](const PendingField& a, const PendingField& b) {
                     return a.index < b.index;
                   });
  for (const PendingField& pf : pending_fields) {
    field_by_qualified[pf.component_name + "." + pf.spec.name] =
        g.add_field(pf.owner, pf.spec);
  }

  for (const auto& n : root["channels"]) {
    if (auto e = check_unknown_keys(
            n,
            {"name", "source", "target", "accumulate", "remap", "spacetime",
             "relax", "extrapolate", "lag", "allow_time_level_mismatch"},
            "channel entry")) {
      return *e;
    }
    if (auto e = require_key(n, "name", "channel entry")) return *e;
    const auto name = n["name"].as<std::string>();
    if (auto e = require_key(n, "source", "channel '" + name + "'")) return *e;
    if (auto e = require_key(n, "target", "channel '" + name + "'")) return *e;
    const auto src_name = n["source"].as<std::string>();
    const auto tgt_name = n["target"].as<std::string>();
    const auto sit = field_by_qualified.find(src_name);
    const auto tit = field_by_qualified.find(tgt_name);
    if (sit == field_by_qualified.end()) {
      return Failure{"E-YAML-REF", "unknown source field '" + src_name + "'"};
    }
    if (tit == field_by_qualified.end()) {
      return Failure{"E-YAML-REF", "unknown target field '" + tgt_name + "'"};
    }
    // BM-16: same reasoning as the mesh/region tables above -- an
    // iteration region naming this channel by name (below) must not be
    // able to silently bind to whichever of two identically-named
    // channels happened to be declared last.
    if (channel_by_name.count(name)) {
      return Failure{"E-YAML-DUP", "duplicate channel name '" + name + "'"};
    }
    const ChannelId cid = g.add_channel(name, sit->second, tit->second);
    channel_by_name[name] = cid;
    auto b = g.channel(cid);

    if (n["accumulate"]) {
      if (auto e = check_unknown_keys(n["accumulate"], {"op", "window"},
                                      "channel '" + name + "' accumulate")) {
        return *e;
      }
      AccumOp op = AccumOp::Instant;
      if (n["accumulate"]["op"] &&
          !lookup_enum(n["accumulate"]["op"].as<std::string>(), kAccumNames, op)) {
        return Failure{"E-YAML-PARSE", "unknown accumulate op"};
      }
      auto win = parse_window(n["accumulate"]["window"], component_by_name,
                              "channel '" + name + "' accumulate.window");
      if (!win.ok()) return win.error();
      b.accumulate(op, win.value());
    }
    if (n["remap"]) {
      if (auto e = check_unknown_keys(n["remap"], {"flags", "method", "remap_on"},
                                      "channel '" + name + "' remap")) {
        return *e;
      }
      if (n["remap"]["flags"]) {
        RemapFlags flags = RemapFlags::None;
        for (const auto& f : n["remap"]["flags"]) {
          const auto s = f.as<std::string>();
          if (s == "Conservative")   flags = flags | RemapFlags::Conservative;
          else if (s == "Monotone")  flags = flags | RemapFlags::Monotone;
          else if (s == "HighOrder") flags = flags | RemapFlags::HighOrder;
          else return Failure{"E-YAML-PARSE", "unknown remap flag '" + s + "'"};
        }
        b.remap(flags);
      }
      if (n["remap"]["method"]) {
        b.remap_method(n["remap"]["method"].as<std::string>());
      }
      if (n["remap"]["remap_on"]) {
        // The coupler-placement track: names either a coupler or a physics
        // component by name, resolved through the SAME `component_by_name`
        // table `source`/`target` use. Whether it names a valid coupler or
        // one of this channel's own endpoints is a validate()-time question
        // (E-CHANNEL-REMAP-ON): a reference to an unknown NAME, in contrast,
        // is a parse-time reference error, same class as an unknown source
        // or target field above.
        const auto ro_name = n["remap"]["remap_on"].as<std::string>();
        const auto ro_it = component_by_name.find(ro_name);
        if (ro_it == component_by_name.end()) {
          return Failure{"E-YAML-REF",
                       "channel '" + name + "' remap_on references unknown "
                       "component '" + ro_name + "'"};
        }
        b.remap_on(ro_it->second);
      }
    }
    if (n["spacetime"]) {
      if (auto e = check_unknown_keys(n["spacetime"],
                                      {"basis", "degree", "window"},
                                      "channel '" + name + "' spacetime")) {
        return *e;
      }
      if (auto e = require_key(n["spacetime"], "basis",
                               "channel '" + name + "' spacetime")) {
        return *e;
      }
      TemporalBasis basis = TemporalBasis::None;
      if (!lookup_enum(n["spacetime"]["basis"].as<std::string>(), kBasisNames,
                       basis)) {
        return Failure{"E-YAML-PARSE", "unknown temporal basis"};
      }
      const int degree =
          n["spacetime"]["degree"] ? n["spacetime"]["degree"].as<int>() : 0;
      auto win = parse_window(n["spacetime"]["window"], component_by_name,
                              "channel '" + name + "' spacetime.window");
      if (!win.ok()) return win.error();
      b.spacetime(basis, degree, win.value());
    }
    if (n["relax"])       b.relax(n["relax"].as<double>());
    if (n["extrapolate"]) b.extrapolate(n["extrapolate"].as<int>());
    if (n["lag"])         b.lag(n["lag"].as<int>());
    if (n["allow_time_level_mismatch"]) {
      b.allow_time_level_mismatch(
          n["allow_time_level_mismatch"].as<std::string>());
    }
  }

  for (const auto& n : root["merges"]) {
    if (!n["field"] || !n["component"] || !n["rule"]) {
      return Failure{"E-YAML-MERGE",
                     "a merge entry needs field, component and rule"};
    }
    const std::string comp = n["component"].as<std::string>();
    const std::string field = n["field"].as<std::string>();
    FieldId target;
    for (std::uint32_t i = 0; i < g.field_count(); ++i) {
      if (g.field(FieldId{i}).name == field &&
          g.component(g.owner_of(FieldId{i})).name == comp) {
        target = FieldId{i};
        break;
      }
    }
    if (!target.valid()) {
      return Failure{"E-YAML-MERGE",
                     "merge names field '" + comp + "." + field + "', which is not declared"};
    }
    const std::string rule = n["rule"].as<std::string>();
    MergeOp op{};
    if (rule == "Sum") op = MergeOp::Sum;
    else if (rule == "Mean") op = MergeOp::Mean;
    else if (rule == "First") op = MergeOp::First;
    else {
      return Failure{"E-YAML-MERGE",
                     "unknown merge rule '" + rule +
                         "'; expected Sum, Mean or First. Refusing rather than "
                         "defaulting, which would silently change the physics"};
    }
    g.merge(target, op);
  }

  for (const auto& n : root["iteration_regions"]) {
    if (auto e = check_unknown_keys(
            n, {"name", "channels", "max_iterations", "convergence",
                "acceleration", "window_scope",
                // S7.1/S7.2's own capability seam.
                "requires_residual_capability",
                "requires_linearization_capability", "newton_config"},
            "iteration region entry")) {
      return *e;
    }
    if (auto e = require_key(n, "name", "iteration region entry")) return *e;
    IterationRegionDecl d;
    d.name = n["name"].as<std::string>();
    for (const auto& c : n["channels"]) {
      const auto cname = c.as<std::string>();
      const auto it = channel_by_name.find(cname);
      if (it == channel_by_name.end()) {
        return Failure{"E-YAML-REF", "unknown channel '" + cname + "'"};
      }
      d.channels.push_back(it->second);
    }
    if (n["max_iterations"]) {
      d.max_iterations = n["max_iterations"].as<int>();
      // BM-17: an authoring-time gate for a defect that otherwise surfaces
      // only mid-run, inside the sweep loop (`Runner`'s own non-positive-
      // count check, several collectives deep) instead of here, where the
      // whole document's problems can be reported together.
      if (d.max_iterations < 1) {
        return Failure{"E-YAML-PARSE",
                     "iteration region '" + d.name +
                         "': max_iterations must be at least 1, got " +
                         std::to_string(d.max_iterations)};
      }
    }
    for (const auto& mn : n["convergence"]) {
      if (auto e = check_unknown_keys(mn, {"kind", "tolerance", "field"},
                                      "convergence measure")) {
        return *e;
      }
      ConvergenceMeasure m;
      if (!mn["kind"] ||
          !lookup_enum(mn["kind"].as<std::string>(), kMeasureNames, m.kind)) {
        return Failure{"E-YAML-PARSE",
                     "iteration region '" + d.name +
                         "': unknown or missing convergence measure kind"};
      }
      if (!mn["tolerance"] || !mn["field"]) {
        return Failure{"E-YAML-PARSE",
                     "iteration region '" + d.name +
                         "': a convergence measure needs both a tolerance and "
                         "a field"};
      }
      m.tolerance = mn["tolerance"].as<double>();
      // BM-17: `ConvergenceMeasure::satisfied()` (fuse/model/iteration.hpp)
      // refuses a non-finite or non-positive tolerance with FUSE_THROW,
      // mid-sweep, several collectives into a run -- the same condition is
      // checkable, and reportable alongside every other document problem,
      // at the authoring gate instead.
      if (!(m.tolerance > 0.0) || !std::isfinite(m.tolerance)) {
        return Failure{"E-YAML-PARSE",
                     "iteration region '" + d.name + "': convergence measure "
                         "for field '" + mn["field"].as<std::string>() +
                         "' has a tolerance that is not finite and positive"};
      }
      m.field = mn["field"].as<std::string>();
      d.convergence.add(std::move(m));
    }
    if (const auto& an = n["acceleration"]) {
      if (auto e = check_unknown_keys(
              an, {"kind", "initial_relaxation", "omega", "reuse", "filter",
                   "limit", "scale_fields"},
              "acceleration block")) {
        return *e;
      }
      if (an["kind"] &&
          !lookup_enum(an["kind"].as<std::string>(), kAccelNames,
                       d.acceleration.kind)) {
        return Failure{"E-YAML-PARSE", "unknown accelerator kind '" +
                                         an["kind"].as<std::string>() + "'"};
      }
      if (an["filter"] &&
          !lookup_enum(an["filter"].as<std::string>(), kFilterNames,
                       d.acceleration.filter)) {
        return Failure{"E-YAML-PARSE", "unknown accelerator filter '" +
                                         an["filter"].as<std::string>() + "'"};
      }
      if (an["initial_relaxation"]) {
        d.acceleration.initial_relaxation =
            an["initial_relaxation"].as<double>();
      }
      if (an["omega"]) d.acceleration.omega = an["omega"].as<double>();
      if (an["reuse"]) d.acceleration.reuse = an["reuse"].as<int>();
      if (an["limit"]) d.acceleration.limit = an["limit"].as<double>();
      if (an["scale_fields"]) {
        d.acceleration.scale_fields = an["scale_fields"].as<bool>();
      }
    }
    if (n["requires_residual_capability"]) {
      d.requires_residual_capability = n["requires_residual_capability"].as<bool>();
    }
    if (n["requires_linearization_capability"]) {
      d.requires_linearization_capability =
          n["requires_linearization_capability"].as<bool>();
    }
    if (const auto& nn = n["newton_config"]) {
      if (auto e = check_unknown_keys(
              nn,
              {"line_search_kind", "forcing_choice", "forcing_gamma",
               "forcing_alpha", "eta_initial", "eta_max", "eta_min",
               "forcing_safeguard_threshold", "forcing_eta_constant", "atol",
               "rtol", "stol", "divergence_tolerance",
               "max_nonlinear_iterations", "max_function_evaluations",
               "armijo_alpha", "lambda_min", "max_backtracks", "secant_tol",
               "max_secant_steps", "damping", "gmres_restart",
               "gmres_max_iterations", "preconditioner_lag",
               "use_block_jacobi_preconditioner", "substrate", "norm",
               "residual_path"},
              "newton_config block")) {
        return *e;
      }
      NewtonConfig& nc = d.newton_config;
      if (nn["line_search_kind"] &&
          !lookup_enum(nn["line_search_kind"].as<std::string>(), kLineSearchNames,
                       nc.line_search_kind)) {
        return Failure{"E-YAML-PARSE", "unknown newton_config.line_search_kind '" +
                                         nn["line_search_kind"].as<std::string>() + "'"};
      }
      if (nn["forcing_choice"] &&
          !lookup_enum(nn["forcing_choice"].as<std::string>(), kEwChoiceNames,
                       nc.forcing_choice)) {
        return Failure{"E-YAML-PARSE", "unknown newton_config.forcing_choice '" +
                                         nn["forcing_choice"].as<std::string>() + "'"};
      }
      if (nn["forcing_gamma"]) nc.forcing_gamma = nn["forcing_gamma"].as<double>();
      if (nn["forcing_alpha"]) nc.forcing_alpha = nn["forcing_alpha"].as<double>();
      if (nn["eta_initial"]) nc.eta_initial = nn["eta_initial"].as<double>();
      if (nn["eta_max"]) nc.eta_max = nn["eta_max"].as<double>();
      if (nn["eta_min"]) nc.eta_min = nn["eta_min"].as<double>();
      if (nn["forcing_safeguard_threshold"]) {
        nc.forcing_safeguard_threshold = nn["forcing_safeguard_threshold"].as<double>();
      }
      if (nn["forcing_eta_constant"]) {
        nc.forcing_eta_constant = nn["forcing_eta_constant"].as<double>();
      }
      if (nn["atol"]) nc.atol = nn["atol"].as<double>();
      if (nn["rtol"]) nc.rtol = nn["rtol"].as<double>();
      if (nn["stol"]) nc.stol = nn["stol"].as<double>();
      if (nn["divergence_tolerance"]) {
        nc.divergence_tolerance = nn["divergence_tolerance"].as<double>();
      }
      if (nn["max_nonlinear_iterations"]) {
        nc.max_nonlinear_iterations = nn["max_nonlinear_iterations"].as<int>();
      }
      if (nn["max_function_evaluations"]) {
        nc.max_function_evaluations =
            nn["max_function_evaluations"].as<std::size_t>();
      }
      if (nn["armijo_alpha"]) nc.armijo_alpha = nn["armijo_alpha"].as<double>();
      if (nn["lambda_min"]) nc.lambda_min = nn["lambda_min"].as<double>();
      if (nn["max_backtracks"]) nc.max_backtracks = nn["max_backtracks"].as<int>();
      if (nn["secant_tol"]) nc.secant_tol = nn["secant_tol"].as<double>();
      if (nn["max_secant_steps"]) {
        nc.max_secant_steps = nn["max_secant_steps"].as<int>();
      }
      if (nn["damping"]) nc.damping = nn["damping"].as<double>();
      if (nn["gmres_restart"]) nc.gmres_restart = nn["gmres_restart"].as<int>();
      if (nn["gmres_max_iterations"]) {
        nc.gmres_max_iterations = nn["gmres_max_iterations"].as<std::size_t>();
      }
      if (nn["preconditioner_lag"]) {
        nc.preconditioner_lag = nn["preconditioner_lag"].as<int>();
      }
      if (nn["use_block_jacobi_preconditioner"]) {
        nc.use_block_jacobi_preconditioner =
            nn["use_block_jacobi_preconditioner"].as<bool>();
      }
      if (nn["substrate"] &&
          !lookup_enum(nn["substrate"].as<std::string>(), kNewtonSubstrateNames,
                       nc.substrate)) {
        return Failure{"E-YAML-PARSE", "unknown newton_config.substrate '" +
                                         nn["substrate"].as<std::string>() + "'"};
      }
      if (nn["norm"] &&
          !lookup_enum(nn["norm"].as<std::string>(), kNewtonNormNames,
                       nc.norm)) {
        return Failure{"E-YAML-PARSE", "unknown newton_config.norm '" +
                                         nn["norm"].as<std::string>() + "'"};
      }
      if (nn["residual_path"] &&
          !lookup_enum(nn["residual_path"].as<std::string>(), kNewtonResidualPathNames,
                       nc.residual_path)) {
        return Failure{"E-YAML-PARSE", "unknown newton_config.residual_path '" +
                                         nn["residual_path"].as<std::string>() + "'"};
      }
    }
    // The SAME auto-trigger `IterationBuilder::accelerate()`/`::newton()`
    // apply in C++ (fuse/model/coupling_graph.hpp): a YAML-authored region
    // that selects Newton must not be able to omit the capability bit by
    // omission any more than a C++-authored one can. Applied AFTER both
    // blocks above are read, so an explicit `requires_residual_capability:
    // false` paired with `acceleration.kind: Newton` is overridden the
    // same way the builder overrides it -- never silently honoured as "no
    // capability needed", which would defeat the whole seam.
    if (d.acceleration.kind == AccelKind::Newton) {
      d.requires_residual_capability = true;
      if (d.newton_config.use_block_jacobi_preconditioner) {
        d.requires_linearization_capability = true;
      }
    }
    if (const auto& wn = n["window_scope"]) {
      if (auto e = check_unknown_keys(
              wn, {"window_component", "explicit_interval", "order",
                   "policies", "iteration"},
              "window_scope block")) {
        return *e;
      }
      CouplingWindowScope scope;
      if (wn["window_component"]) {
        scope.window_component = wn["window_component"].as<std::string>();
      }
      if (wn["explicit_interval"]) {
        Rational len;
        if (!parse_rational(wn["explicit_interval"].as<std::string>(), len)) {
          return Failure{"E-YAML-PARSE",
                       "iteration region '" + d.name +
                           "': window_scope.explicit_interval is not a "
                           "valid rational"};
        }
        scope.explicit_interval = len;
      }
      if (wn["order"]) scope.order = wn["order"].as<int>();
      for (const auto& pn : wn["policies"]) {
        if (auto e = check_unknown_keys(pn, {"component", "policy"},
                                        "window_scope policy entry")) {
          return *e;
        }
        WindowMemberPolicy p;
        if (!pn["component"]) {
          return Failure{"E-YAML-PARSE",
                       "iteration region '" + d.name +
                           "': window_scope policy entry has no component"};
        }
        p.component = pn["component"].as<std::string>();
        if (!pn["policy"] ||
            !lookup_enum(pn["policy"].as<std::string>(), kDtPolicyNames,
                        p.policy)) {
          return Failure{"E-YAML-PARSE",
                       "iteration region '" + d.name +
                           "': window_scope policy for component '" +
                           p.component + "' has an unknown or missing "
                           "DtPolicy"};
        }
        scope.policies.push_back(std::move(p));
      }
      if (wn["iteration"]) {
        if (!lookup_enum(wn["iteration"].as<std::string>(),
                         kWaveformIterationNames, scope.iteration)) {
          return Failure{"E-YAML-PARSE",
                       "iteration region '" + d.name +
                           "': window_scope.iteration is not a known "
                           "WaveformIteration"};
        }
      }
      d.window_scope = std::move(scope);
    }
    g.add_iteration_region(std::move(d));
  }

  return g;
}
}  // namespace

Result<CouplingGraph> from_yaml(std::string_view text) {
  YAML::Node root;
  try {
    root = YAML::Load(std::string{text});
  } catch (const YAML::Exception& ex) {
    return Failure{"E-YAML-PARSE", ex.what()};
  }
  // BM-2: the body above this point can only fail via YAML::Load itself
  // (caught just above); everything from here on -- required-key access,
  // RankSet construction (BM-14), and every other `.as<T>()` call on an
  // OPTIONAL key that still turns out to be the wrong YAML type -- is
  // wrapped in one catch-all so no exception, FUSE-shaped or not, can
  // escape a function whose contract is `Result<CouplingGraph>`.
  try {
    return from_yaml_body(root);
  } catch (const std::exception& ex) {
    return Failure{"E-YAML-PARSE", ex.what()};
  }
}

Result<CouplingGraph> load_yaml_file(const std::string& path) {
  // Read with iterators rather than `ss << in.rdbuf()`: the invariant gate
  // rejects <iostream>/<sstream> in library code (locale-dependent formatting is
  // how a diagnostic's text changes with the environment), and a whole-file read
  // needs neither. `std::ifstream` itself is fine -- it is file IO, not
  // formatted output.
  std::ifstream in{path, std::ios::binary};
  if (!in) return Failure{"E-YAML-IO", "cannot open '" + path + "'"};
  const std::string text{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
  if (in.bad()) return Failure{"E-YAML-IO", "error while reading '" + path + "'"};
  return from_yaml(text);
}

}  // namespace fuse
