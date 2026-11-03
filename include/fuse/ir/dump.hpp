#pragma once

#include <cstdint>
#include <locale>
#include <sstream>
#include <string>

#include "fuse/ir/dag.hpp"

namespace fuse {
namespace detail {

/// std::to_string(double) formats through the global locale, so its output
/// varies with LC_NUMERIC (e.g. "0,7" under de_DE) even though the dump is a
/// byte-for-byte golden-file artifact. Route every floating-point
/// value in the dump through a stream imbued with the classic "C" locale
/// instead, so the format is fixed regardless of the process locale.
inline std::string format_double(double v) {
  std::ostringstream os;
  os.imbue(std::locale::classic());
  os << v;
  return os.str();
}

/// Only non-default attributes, in a fixed order, so dumps stay diff-friendly.
inline std::string attrs_string(OpKind kind, const OpAttrs& a) {
  const OpAttrs def{};
  std::string out;
  const auto add = [&out](const std::string& kv) {
    if (!out.empty()) out += ", ";
    out += kv;
  };
  if (a.accum != def.accum) add(std::string{"accum="} + to_cstring(a.accum));
  if (a.basis != def.basis) add(std::string{"basis="} + to_cstring(a.basis));
  if (a.degree != def.degree) add("degree=" + std::to_string(a.degree));
  if (a.omega != def.omega) add("omega=" + format_double(a.omega));
  if (a.order != def.order) add("order=" + std::to_string(a.order));
  if (a.lag != def.lag) add("lag=" + std::to_string(a.lag));
  // Exact, so it prints as a rational -- "scale=1/1000", never 0.001. The
  // whole point of carrying it as a Rational is that the dump and the plan
  // show the factor the user's units implied, not a rounding of it.
  if (a.scale != def.scale) add("scale=" + a.scale.to_string());
  if (a.reason != def.reason) add("reason=" + a.reason);
  if (a.remap_flags != def.remap_flags) {
    std::string f;
    if (has(a.remap_flags, RemapFlags::Conservative)) f += "C";
    if (has(a.remap_flags, RemapFlags::Monotone))     f += "M";
    if (has(a.remap_flags, RemapFlags::HighOrder))    f += "H";
    add("remap=" + f);
  }
  if (!a.method.empty()) add("method=" + a.method);
  if (a.window_import != def.window_import) {
    add("window_import=%" + std::to_string(a.window_import.value()));
  }
  if (kind == OpKind::Transport) {
    add(std::to_string(a.src_rank_set) + "->" + std::to_string(a.dst_rank_set));
  }
  return out;
}

inline std::string pad(std::string s, std::size_t width) {
  while (s.size() < width) s += ' ';
  return s;
}

}  // namespace detail

/// Spec Section 10 level 2: this text is checked into test/golden/ and compared
/// byte-for-byte, so a lowering change cannot silently alter the numerics.
[[nodiscard]] inline std::string dump_text(const OperatorDAG& d) {
  std::ostringstream os;
  os.imbue(std::locale::classic());
  os << "# fuse-ir v1\n";

  for (const NodeId nid : d.topological_order()) {
    const Node& n = d.node(nid);
    os << "node " << detail::pad(std::to_string(nid.value()), 3) << " "
       << detail::pad(to_cstring(n.kind), 12) << " ";

    std::string ins;
    for (const ArcId a : n.inputs) {
      if (!ins.empty()) ins += ", ";
      ins += "%" + std::to_string(a.value());
    }
    os << detail::pad(ins, 12) << " -> ";

    if (n.output.valid()) {
      os << "%" << n.output.value() << " : "
         << d.arc(n.output).type.to_string();
    } else {
      os << "(none)";
    }

    if (!n.label.empty()) os << "  \"" << n.label << "\"";
    const std::string attrs = detail::attrs_string(n.kind, n.attrs);
    if (!attrs.empty()) os << " {" << attrs << "}";
    os << "\n";
  }

  for (std::size_t i = 0; i < d.region_count(); ++i) {
    const auto& r = d.region_at(RegionIdx{static_cast<std::uint32_t>(i)});
    os << "region " << i << " \"" << r.name << "\" body=[";
    for (std::size_t k = 0; k < r.body.size(); ++k) {
      if (k) os << ",";
      os << r.body[k].value();
    }
    os << "] residuals=[";
    for (std::size_t k = 0; k < r.residuals.size(); ++k) {
      if (k) os << ",";
      os << "%" << r.residuals[k].value();
    }
    os << "] carried=[";
    for (std::size_t k = 0; k < r.carried.size(); ++k) {
      if (k) os << ",";
      os << "%" << r.carried[k].value();
    }
    os << "] max_iter=" << r.max_iterations << "\n";
  }

  return os.str();
}

[[nodiscard]] inline std::string dump_graphviz(const OperatorDAG& d) {
  std::ostringstream os;
  os.imbue(std::locale::classic());
  // The graph name is "fuse_ir", not a soma-prefixed one (conventions.md
  // bans soma-prefixed names in public output artifacts).
  os << "digraph fuse_ir {\n  rankdir=LR;\n  node [shape=box];\n";
  for (const NodeId nid : d.topological_order()) {
    const Node& n = d.node(nid);
    os << "  n" << nid.value() << " [label=\"" << to_cstring(n.kind);
    if (!n.label.empty()) os << "\\n" << n.label;
    const std::string attrs = detail::attrs_string(n.kind, n.attrs);
    if (!attrs.empty()) os << "\\n" << attrs;
    os << "\"];\n";
  }
  for (const NodeId nid : d.topological_order()) {
    const Node& n = d.node(nid);
    for (const ArcId in : n.inputs) {
      const NodeId prod = d.arc(in).producer;
      if (!prod.valid()) continue;
      os << "  n" << prod.value() << " -> n" << nid.value()
         << " [label=\"%" << in.value() << "\"];\n";
    }
  }
  os << "}\n";
  return os.str();
}

}  // namespace fuse
