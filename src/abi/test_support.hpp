#pragma once

/// A TEST-ONLY seam, never installed and not part of the frozen ABI
/// (include/fuse/abi/{entries.def,types.h,participant.h}): the plain C
/// participant ABI has no entry that registers an in-process advance
/// callback (deliberately -- see docs/dev/abi-design.md and the `run` entry's
/// own doc string in entries.def), so there is no way for a C caller, or an
/// ABI-level test using only the public table, to put a component whose
/// `advance()` throws into a running graph. This header is how
/// test/abi/participant_test.cpp's exception-containment case
/// (AThrowingComponentBecomesRuntimeErrorWithTheMessage) and
/// test/abi/negotiation_test.cpp's fault-injection case construct one
/// anyway: by reaching past the C surface, in C++, exactly the way those
/// test files already reach past fuse_coupler_abi() itself to build a
/// deliberately wrong table.
///
/// Only declared here; test/abi/*.cpp link src/abi/test_support.cpp
/// directly (not through the installed library), which is why this is a
/// free function and not a table entry.

#include <memory>
#include <string>

#include "fuse/runtime/component.hpp"

struct fuse_ctx;

namespace fuse::abi_test {

/// Replaces the fuse::abi_detail::ExternalStepComponent that fuse_initialize()
/// would otherwise build for the graph component named `component_name` with
/// `c`, so that component is driven IN-PROCESS (through fuse_run(), never
/// through fuse_begin_step()/fuse_end_step()) instead.
///
/// Must be called after a successful fuse_load_graph() and before
/// fuse_initialize(); `component_name` must name a component the loaded
/// graph declares. Returns false (and leaves `ctx` unchanged) when `ctx` is
/// null, the context is not in the Loaded state, `component_name` is not a
/// component of the loaded graph, or a component has already been injected
/// under that name -- in every one of those cases `c` is destroyed.
bool inject_component(fuse_ctx* ctx, const std::string& component_name,
                      std::unique_ptr<fuse::Component> c);

}  // namespace fuse::abi_test
