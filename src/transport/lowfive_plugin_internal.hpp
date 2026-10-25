#pragma once

// PRIVATE to
// fuse-lowfive-vol's own two translation units (lowfive_plugin_vol.cpp,
// lowfive_plugin_graph.cpp) -- not installed, not a public fuse/ header.
//
// Plain data ONLY: no `fmt`, no `lowfive/...`, no `fuse/model/...`,
// nothing from `fuse/base/result.hpp`. This is the whole reason this
// header exists. Measured from a FRESH configure with
// `/opt/lowfive;/opt/hdf5` FIRST on `CMAKE_PREFIX_PATH`: a translation
// unit that includes BOTH a LowFive header (whose own include directory
// carries a bundled, incompatible `fmt` 8.1.1) AND a FUSE header that
// needs `fmt` 12 (`fuse/base/result.hpp`, pulled in by
// `fuse/model/yaml_io.hpp` and by `fuse/transport/transport.hpp` alike)
// fails to compile outright --
// `fmt/format-inl.h:2534: call to consteval function ... is not a
// constant expression` -- regardless of which `fmt` a `target_link_
// libraries` ORDER happens to put first on the search path (relying on
// that order breaks under a different, still-supported
// `CMAKE_PREFIX_PATH` ordering). The only configure-order-independent fix is to never
// let one translation unit see both `fmt` variants' own headers at all --
// enforced here at the CMake level (lowfive_plugin_graph.cpp's own
// object library links `fuse::model`/`fuse::base` and NEVER `fuse::
// lowfive`; lowfive_plugin_vol.cpp's links `fuse::lowfive` and NEVER
// `fuse::model` -- src/CMakeLists.txt), not merely by a source-level
// `#include` discipline a future edit could quietly violate.
//
// lowfive_plugin_graph.cpp: the ONLY place Config is ever built, from
// `FUSE_LOWFIVE_BINDINGS` and (for a multi-rank LowFive-bound channel
// naming this process) `fuse::plan_lowfive_rendezvous()`.
// lowfive_plugin_vol.cpp: the ONLY place Config is ever consumed --
// `set_memory()` per binding, and the actual `MPI_Comm_split`/
// `MPI_Intercomm_create`/`DistMetadataVOL` construction when
// `distributed` is set.

#include <string>
#include <vector>

namespace fuse::lowfive_plugin_internal {

/// One `LowFive::MetadataVOL::set_memory(filename, pattern)` call's own
/// two arguments, already split (lowfive_plugin_graph.cpp's own job --
/// `LowFiveFieldBindingTable`'s "<file>.h5/<dataset>" port syntax is a
/// `fuse::transport::lowfive` concept this header must not name).
struct FieldBinding {
  std::string filename_pattern;
  std::string path_pattern;
};

/// Plain mirror of `fuse::LowFiveRendezvousPlan` (fuse/transport/
/// lowfive.hpp) -- every field copied by value, nothing from that header
/// (or the RankSet type it is built from) named here.
struct RendezvousPlan {
  bool is_writer = false;
  int split_color = 0;
  int split_key = 0;
  int local_leader = 0;
  int remote_leader = 0;
  int tag = 0;
};

struct Config {
  std::vector<FieldBinding> bindings;
  /// True iff a LowFive-bound channel naming this rank was found in a
  /// multi-rank MPI world -- `rendezvous` is meaningful only then.
  bool distributed = false;
  RendezvousPlan rendezvous;
};

/// Reads `FUSE_LOWFIVE_BINDINGS` (a graph YAML path) and populates
/// `*out`. Returns false (an empty `*out`) on a missing env var or any
/// load/parse failure -- lowfive_plugin_graph.cpp's own stderr carries
/// the diagnostic (that file, unlike this header, may use `fmt`); this
/// function's own boolean is the only signal this header's callers get.
[[nodiscard]] bool LoadConfig(Config* out);

}  // namespace fuse::lowfive_plugin_internal
