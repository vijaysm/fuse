#pragma once

#include <string>
#include <vector>

#include "soma/base/ids.hpp"
#include "soma/base/time_level.hpp"
#include "soma/base/units.hpp"
#include "soma/graph/channel.hpp"  // RemapFlags

namespace soma {

/// Spec §5.1's twelve primitives, plus Source/Sink boundary nodes so that
/// every arc has a producer and every value a consumer.
enum class OpKind {
  Sample, Accumulate, Interpolate, Extrapolate, Remap, Merge, Relax,
  Transport, Residual, Accelerate, Constrain, Diagnose,
  Source, Sink,
};

[[nodiscard]] constexpr const char* to_cstring(OpKind k) noexcept {
  switch (k) {
    case OpKind::Sample:      return "Sample";
    case OpKind::Accumulate:  return "Accumulate";
    case OpKind::Interpolate: return "Interpolate";
    case OpKind::Extrapolate: return "Extrapolate";
    case OpKind::Remap:       return "Remap";
    case OpKind::Merge:       return "Merge";
    case OpKind::Relax:       return "Relax";
    case OpKind::Transport:   return "Transport";
    case OpKind::Residual:    return "Residual";
    case OpKind::Accelerate:  return "Accelerate";
    case OpKind::Constrain:   return "Constrain";
    case OpKind::Diagnose:    return "Diagnose";
    case OpKind::Source:      return "Source";
    case OpKind::Sink:        return "Sink";
  }
  return "Unknown";
}

enum class ValueKind { Instant, Waveform };

[[nodiscard]] constexpr const char* to_cstring(ValueKind k) noexcept {
  return k == ValueKind::Waveform ? "Waveform" : "Instant";
}

/// The type of an SSA value. Spec §2: every arc carries its time level, so a
/// staggering error is a type error found at lowering.
struct ValueType {
  ValueKind    kind = ValueKind::Instant;
  RegionId     region;
  int          components = 1;
  Units        units{};
  Conservation conservation = Conservation::None;
  TimeLevel    time{};
  int          rank_set = -1;  // component index the value lives on; -1 = unplaced

  friend bool operator==(const ValueType&, const ValueType&) = default;

  [[nodiscard]] std::string to_string() const {
    std::string s = to_cstring(kind);
    s += "<r";
    s += region.valid() ? std::to_string(region.value()) : std::string{"?"};
    if (components != 1) {
      s += "x";
      s += std::to_string(components);
    }
    s += ", ";
    s += units.to_string();
    if (conservation != Conservation::None) {
      s += ", ";
      s += to_cstring(conservation);
    }
    s += ", ";
    s += time.to_string();
    if (rank_set >= 0) {
      s += ", ranks=";
      s += std::to_string(rank_set);
    }
    s += ">";
    return s;
  }
};

/// Can `consumer` accept a value of type `producer`?
/// allow_time_mismatch comes from Channel::time_level_mismatch_reason and
/// relaxes ONLY the time check (spec §4.3).
[[nodiscard]] inline bool assignable_from(const ValueType& producer,
                                          const ValueType& consumer,
                                          bool allow_time_mismatch) noexcept {
  if (producer.kind != consumer.kind) return false;
  if (producer.region != consumer.region) return false;
  if (producer.components != consumer.components) return false;
  if (!(producer.units == consumer.units)) return false;
  if (producer.conservation != consumer.conservation) return false;
  if (!allow_time_mismatch && !compatible(producer.time, consumer.time)) {
    return false;
  }
  return true;
}

/// Flat attribute bag. Each OpKind reads only what it needs; keeping it flat
/// keeps Node cheap to copy and the dump format stable.
struct OpAttrs {
  AccumOp       accum = AccumOp::Instant;
  TemporalBasis basis = TemporalBasis::None;
  int           degree = 0;
  double        omega = 1.0;
  int           order = 0;
  int           lag = 0;
  RemapFlags    remap_flags = RemapFlags::None;
  std::string   method;
  int           src_rank_set = -1;
  int           dst_rank_set = -1;

  friend bool operator==(const OpAttrs&, const OpAttrs&) = default;
};

struct Node {
  NodeId               id;
  OpKind               kind = OpKind::Source;
  std::vector<ArcId>   inputs;
  ArcId                output;
  OpAttrs              attrs;
  ChannelId            origin;  // which channel emitted this node
  std::string          label;
};

struct Arc {
  ArcId       id;
  NodeId      producer;
  ValueType   type;
  std::string name;
};

}  // namespace soma
