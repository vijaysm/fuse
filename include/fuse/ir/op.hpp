#pragma once

#include <string>
#include <vector>

#include "fuse/base/ids.hpp"
#include "fuse/base/rational.hpp"
#include "fuse/base/time_level.hpp"
#include "fuse/base/units.hpp"
#include "fuse/model/channel.hpp"  // RemapFlags
#include "fuse/model/merge.hpp"

namespace fuse {

/// Spec §5.1's twelve primitives, plus Source/Sink boundary nodes so that
/// every arc has a producer and every value a consumer.
enum class OpKind {
  Sample, Accumulate, Interpolate, Extrapolate, Remap, Merge, Relax,
  Transport, Residual, Accelerate, Constrain, Diagnose,
  Source, Sink,

  /// Unit conversion by an exact rational factor. Separate from `Relax` (which
  /// scales by a double) because a unit factor must be exact: 1/1000 applied as
  /// a double is not 1/1000, and a conservation check downstream would then be
  /// comparing sums that differ in the last bits for no physical reason.
  ///
  /// Emitted by the compiler on the connection checker's authority, never on
  /// the user's -- which is why the dump prints it and the plan names it.
  Scale,

  /// Rank agreement point. Every rank in the coupling reaches this instruction
  /// in the same order, and each one learns the same outcome (all-succeeded, or
  /// which rank failed) before any of them proceeds. It is the only place a
  /// failure on one rank can be turned into a decision on all of them, and it
  /// is what stops a local error from becoming a hang at the next collective.
  Epoch,

  /// Reserved for multiscale. Declared so the enumerator ordinals do not move
  /// when they land, and REJECTED by the scheduler today -- see
  /// include/fuse/model/multiscale.hpp for the contracts, and the same
  /// reasoning: a reserved opcode that silently schedules as a no-op would let
  /// a micro-macro coupling compile and run as though the closure had been
  /// applied.
  Map, Reduce, Closure,
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
    case OpKind::Scale:       return "Scale";
    case OpKind::Epoch:       return "Epoch";
    case OpKind::Map:         return "Map";
    case OpKind::Reduce:      return "Reduce";
    case OpKind::Closure:     return "Closure";
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

  /// `Scale`'s exact factor, and `Epoch`'s reason. Both default to values the
  /// dump omits, so no existing golden file moves.
  Rational      scale{1};
  std::string   reason;

  /// `Merge`'s rule. Carried as the enumerator rather than parsed back out of
  /// `reason`, so the interpreter cannot disagree with the graph about which
  /// combination it is performing.
  MergeOp       merge = MergeOp::Sum;

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

}  // namespace fuse
