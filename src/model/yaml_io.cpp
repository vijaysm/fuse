#include "fuse/model/yaml_io.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <fstream>
#include <iterator>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "fuse/model/descriptor.hpp"

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

bool lookup_units(const std::string& sym, Units& out) {
  for (const auto& e : kUnitTable) {
    if (sym == e.symbol) { out = e.value; return true; }
  }
  return false;
}

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

/// Reads either form: a table symbol, or the structural map `units_node` emits.
bool parse_units(const YAML::Node& n, Units& out) {
  if (n.IsScalar()) return lookup_units(n.as<std::string>(), out);
  if (!n.IsMap()) return false;
  Units u{};
  const auto get = [&n](const char* key, std::int8_t& into) {
    if (n[key]) into = static_cast<std::int8_t>(n[key].as<int>());
  };
  get("length", u.length);
  get("mass", u.mass);
  get("time", u.time);
  get("current", u.current);
  get("temperature", u.temperature);
  get("amount", u.amount);
  get("luminosity", u.luminosity);
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

const char* const kDiscNames[]    = {"FV", "cGLL", "dGLL", "PointCloud"};
const char* const kConservNames[] = {"None", "Extensive", "Intensive"};
const char* const kIntentNames[]  = {"Provides", "Requires", "Bidirectional"};
const char* const kAccumNames[]   = {"Instant", "Sum", "Mean", "Min", "Max",
                                     "Integral"};
const char* const kBasisNames[]   = {"None", "Constant", "Linear", "DG",
                                     "Spline"};
// Spec §5.3. These three must stay in the declaration order of MeasureKind
// (num/convergence.hpp), AccelKind and FilterKind (num/accelerator.hpp):
// lookup_enum() maps a name to its index and casts. A reordered enum with an
// unreordered table here reads "Relative" back as "Absolute", which is a
// different stopping rule with no diagnostic, so each table is pinned by a
// round-trip test over every enumerator in test/unit/yaml_io_test.cpp.
const char* const kMeasureNames[] = {"Relative", "Absolute",
                                     "AbsoluteOrRelative", "ResidualRelative"};
const char* const kAccelNames[]   = {"None",       "Constant", "Aitken",
                                     "Secant",     "Steffensen",
                                     "IQN_ILS",    "IQN_IMVJ"};
const char* const kFilterNames[]  = {"None", "QR1", "QR2", "QR3"};

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

using ComponentByName = std::unordered_map<std::string, ComponentId>;

// --- window (de)serialization ------------------------------------------
//
// Windows reference components by name, matching the convention fields and
// channels already use ("fluid.heat_flux"), not by index.
//
// `coupling_period` — the default, and the only kind Brief 1.7's schema
// example shows — is emitted as the bare scalar "coupling_period" so the
// checked-in fixture and the brief's exact schema text keep working
// unchanged. `from_component` and `explicit` carry a payload (a component
// name, or a rational length) that a bare scalar cannot hold, so those two
// kinds are emitted as a small map instead. The parser accepts both shapes
// for "coupling_period" (scalar, or {kind: coupling_period}) so a
// hand-written map form is also legal.

void emit_window(YAML::Emitter& e, const CouplingGraph& g, const Window& w) {
  e << YAML::Key << "window" << YAML::Value;
  switch (w.kind) {
    case Window::Kind::FromComponent:
      e << YAML::Flow << YAML::BeginMap
        << YAML::Key << "kind" << YAML::Value << "from_component"
        << YAML::Key << "component" << YAML::Value
        << g.component(w.component).name
        << YAML::EndMap;
      break;
    case Window::Kind::Explicit:
      e << YAML::Flow << YAML::BeginMap
        << YAML::Key << "kind" << YAML::Value << "explicit"
        << YAML::Key << "length" << YAML::Value << w.length.to_string()
        << YAML::EndMap;
      break;
    case Window::Kind::CouplingPeriod:
    default:
      e << "coupling_period";
      break;
  }
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
    e << YAML::BeginMap
      << YAML::Key << "name" << YAML::Value << r.name
      << YAML::Key << "mesh" << YAML::Value << g.mesh(r.mesh).name
      << YAML::Key << "entity_set" << YAML::Value << r.entity_set
      << YAML::Key << "dimension" << YAML::Value << r.dimension
      << YAML::EndMap;
  }
  e << YAML::EndSeq;

  e << YAML::Key << "components" << YAML::Value << YAML::BeginSeq;
  for (std::uint32_t i = 0; i < g.component_count(); ++i) {
    const ComponentId cid{i};
    const auto& c = g.component(cid);
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
      << YAML::Key << "interval" << YAML::Value << c.interval.to_string()
      << YAML::Key << "fields" << YAML::Value << YAML::BeginSeq;
    for (const FieldId fid : c.fields) {
      const auto& f = g.field(fid);
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
        << YAML::Key << "intent" << YAML::Value << to_cstring(f.intent)
        << YAML::EndMap;
    }
    e << YAML::EndSeq << YAML::EndMap;
  }
  e << YAML::EndSeq;

  e << YAML::Key << "channels" << YAML::Value << YAML::BeginSeq;
  for (std::uint32_t i = 0; i < g.channel_count(); ++i) {
    const auto& c = g.channel_at(ChannelId{i});
    const auto qualified = [&](FieldId f) {
      return g.component(g.owner_of(f)).name + "." + g.field(f).name;
    };
    e << YAML::BeginMap
      << YAML::Key << "name" << YAML::Value << c.name
      << YAML::Key << "source" << YAML::Value << qualified(c.source)
      << YAML::Key << "target" << YAML::Value << qualified(c.target)
      << YAML::Key << "accumulate" << YAML::Value << YAML::Flow << YAML::BeginMap
      << YAML::Key << "op" << YAML::Value << to_cstring(c.accumulate.op);
    emit_window(e, g, c.accumulate.window);
    e << YAML::EndMap;

    if (c.remap.flags != RemapFlags::None || !c.remap.method_override.empty()) {
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
      e << YAML::EndMap;
    }
    if (c.spacetime.basis != TemporalBasis::None) {
      e << YAML::Key << "spacetime" << YAML::Value << YAML::Flow << YAML::BeginMap
        << YAML::Key << "basis" << YAML::Value << to_cstring(c.spacetime.basis)
        << YAML::Key << "degree" << YAML::Value << c.spacetime.degree;
      emit_window(e, g, c.spacetime.window);
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
    for (const ChannelId c : r.channels) e << g.channel_at(c).name;
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
        << kMeasureNames[static_cast<std::size_t>(m.kind)]
        << YAML::Key << "tolerance" << YAML::Value << m.tolerance
        << YAML::Key << "field" << YAML::Value << m.field
        << YAML::EndMap;
    }
    e << YAML::EndSeq;
    e << YAML::Key << "acceleration" << YAML::Value << YAML::BeginMap
      << YAML::Key << "kind" << YAML::Value
      << kAccelNames[static_cast<std::size_t>(r.acceleration.kind)]
      << YAML::Key << "initial_relaxation" << YAML::Value
      << r.acceleration.initial_relaxation
      << YAML::Key << "omega" << YAML::Value << r.acceleration.omega
      << YAML::Key << "reuse" << YAML::Value << r.acceleration.reuse
      << YAML::Key << "filter" << YAML::Value
      << kFilterNames[static_cast<std::size_t>(r.acceleration.filter)]
      << YAML::Key << "limit" << YAML::Value << r.acceleration.limit
      << YAML::Key << "scale_fields" << YAML::Value
      << r.acceleration.scale_fields
      << YAML::EndMap;
    e << YAML::EndMap;
  }
  e << YAML::EndSeq << YAML::EndMap;

  if (!e.good()) return Failure{"E-YAML-EMIT", e.GetLastError()};
  return std::string{e.c_str()};
}

Result<CouplingGraph> from_yaml(std::string_view text) {
  YAML::Node root;
  try {
    root = YAML::Load(std::string{text});
  } catch (const YAML::Exception& ex) {
    return Failure{"E-YAML-PARSE", ex.what()};
  }

  if (auto e = check_unknown_keys(
          root,
          {"version", "meshes", "regions", "components", "channels",
           "iteration_regions", "merges"},
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
    const auto name = n["name"].as<std::string>();
    mesh_by_name[name] = g.add_mesh(
        MeshDecl{.name = name,
                 .source = n["source"] ? n["source"].as<std::string>() : ""});
  }

  for (const auto& n : root["regions"]) {
    if (auto e = check_unknown_keys(
            n, {"name", "mesh", "entity_set", "dimension"}, "region entry")) {
      return *e;
    }
    const auto mesh_name = n["mesh"].as<std::string>();
    const auto it = mesh_by_name.find(mesh_name);
    if (it == mesh_by_name.end()) {
      return Failure{"E-YAML-REF", "unknown mesh '" + mesh_name + "'"};
    }
    const auto name = n["name"].as<std::string>();
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

  for (const auto& n : root["components"]) {
    if (auto e = check_unknown_keys(
            n, {"name", "ranks", "interval", "fields"}, "component entry")) {
      return *e;
    }
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
        if (auto e = check_unknown_keys(n["ranks"], {"first", "count"},
                                        "component '" + c.name + "' ranks")) {
          return *e;
        }
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
    const ComponentId cid = g.add_component(c);
    component_by_name[c.name] = cid;

    for (const auto& fn : n["fields"]) {
      if (auto e = check_unknown_keys(
              fn,
              {"name", "region", "disc", "order", "components", "units",
               "conservation", "intent", "index"},
              "field entry in component '" + c.name + "'")) {
        return *e;
      }
      FieldSpec f;
      f.name = fn["name"].as<std::string>();
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
      if (fn["units"] && !parse_units(fn["units"], f.units)) {
        return Failure{"E-YAML-UNITS",
                     "unknown unit symbol '" + fn["units"].as<std::string>() + "'"};
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
      // Collected, not added: the fields are added below in declaration
      // order so the ids match the graph this text came from.
      pending_fields.push_back(PendingField{
          fn["index"] ? fn["index"].as<std::uint32_t>()
                      : static_cast<std::uint32_t>(pending_fields.size()),
          cid, c.name, f});
    }
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
    const auto name = n["name"].as<std::string>();
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
      if (auto e = check_unknown_keys(n["remap"], {"flags", "method"},
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
    }
    if (n["spacetime"]) {
      if (auto e = check_unknown_keys(n["spacetime"],
                                      {"basis", "degree", "window"},
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
                "acceleration"},
            "iteration region entry")) {
      return *e;
    }
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
    if (n["max_iterations"]) d.max_iterations = n["max_iterations"].as<int>();
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
    g.add_iteration_region(std::move(d));
  }

  return g;
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
