#pragma once

// FUSE_COLLECTIVE_TRACE=1 is an off-by-default, per-rank diagnostic for the
// class of defect where a rank enters a base-communicator collective (OpKind::
// Constrain's reduction, fuse::is_collective(), include/fuse/ir/op.hpp) that
// another rank's own schedule does not also reach. Printed from the one
// place that has both the region's own name and the current sweep index
// (Runner::Impl::run_region()'s per-instruction dispatch loop, src/runtime/
// iteration.cpp) -- never from inside execute() itself (src/runtime/
// runner.cpp), which has neither.
//
// This header holds only the PURE formatter (no getenv, no MPI, no Runner
// state) so it is tier-0 testable with no coroutine, no communicator, and
// no graph at all -- see test/runtime/collective_trace_test.cpp.

#include <cstdint>
#include <string>

namespace fuse {

// One line of trace, for one collective instruction's own execution on
// `rank` (-1 when no base communicator is configured, matching the
// existing FUSE_CONSERVATION_DEBUG convention's own `me` variable,
// src/runtime/runner.cpp). `ordinal` is the CALLER's own monotonically
// increasing count of collectives this rank has entered so far in the
// whole run (Runner::Impl::collective_trace_ordinal, incremented once per
// call before this function is invoked) -- comparing two ranks' own
// ordinal-N lines is what tells the host whether they agree on what their
// Nth collective instruction is diffing `u.collective_trace.log` against
// `v.collective_trace.log` against `coupler.collective_trace.log`,
// docs/dev/collective-participation.md's own worked example).
[[nodiscard]] std::string format_collective_trace_line(
    int rank, const std::string& region, int period, int sweep,
    std::uint32_t instruction_id, const std::string& kind,
    const std::string& communicator, std::int64_t ordinal);

}  // namespace fuse
