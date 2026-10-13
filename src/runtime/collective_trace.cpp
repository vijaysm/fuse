#include "fuse/runtime/collective_trace.hpp"

#include <fmt/format.h>

namespace fuse {

std::string format_collective_trace_line(int rank, const std::string& region,
                                         int period, int sweep,
                                         std::uint32_t instruction_id,
                                         const std::string& kind,
                                         const std::string& communicator,
                                         std::int64_t ordinal) {
  return fmt::format(
      "FUSE_COLLECTIVE_TRACE rank={} region={} period={} sweep={} instr={} "
      "kind={} comm={} ordinal={}",
      rank, region, period, sweep, instruction_id, kind, communicator, ordinal);
}

}  // namespace fuse
