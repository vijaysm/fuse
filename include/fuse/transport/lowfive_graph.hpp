#pragma once

// Build a LowFiveFieldBindingTable
// FROM a fuse::CouplingGraph's own fields, instead of a caller hand-listing
// bindings one at a time (fuse/transport/lowfive.hpp's own standalone
// `LowFiveFieldBindingTable::add()` constructor, which stays -- it is what
// test/transport/lowfive_test.cpp already exercises, and what a caller with
// no graph at all, e.g. a plain HDF5-only Demo B driver, still needs).
//
// A SEPARATE header from lowfive.hpp, not an addition to it, on purpose:
// lowfive.hpp's own opening comment documents why it was built with no
// dependency on the model layer -- "read by examples/demo_b's own drivers
// only, and by test/transport/lowfive_test.cpp" -- and every existing
// caller of that header still wants that property (no CouplingGraph pulled
// in just to parse a port string). `fuse::FieldSpec::hdf5_path`
// (fuse/model/field_spec.hpp) makes a graph-aware builder
// useful, but it belongs beside lowfive.hpp, not inside it:
// `transport` sits above `model` in the layer order (tools/check-
// layering.py), so this file may include fuse/model/coupling_graph.hpp;
// lowfive.hpp may not, and keeps not needing to.
//
// LowFiveFieldBindingTable itself -- the TYPE -- is the standalone
// implementation type; this file only adds one more way to populate one.

#include <cstdint>
#include <string>

#include "fuse/base/result.hpp"
#include "fuse/model/coupling_graph.hpp"
#include "fuse/transport/lowfive.hpp"

namespace fuse {

/// One binding per field that sets `FieldSpec::hdf5_path`, qualified as
/// `"<component>.<field>"` (yaml_io.cpp's own qualification scheme for a
/// channel's source/target) rather than the field's bare name: two
/// different components may legally use the same local field name (a
/// coupling graph scopes field names per component, same as validate()'s
/// own `E-DUP-FIELD` check), and `LowFiveFieldBindingTable::add` refuses a
/// second binding under a `field_name` it has already seen
/// (`E-LOWFIVE-DUPFIELD`) -- which a plain, unqualified name could trip
/// over two fields that never meant to collide at all.
///
/// Errors identical to `LowFiveFieldBindingTable::add`'s own
/// (`E-LOWFIVE-PORT`, `E-LOWFIVE-DUPFIELD`, `E-LOWFIVE-DUPPATH`) --
/// `validate()` (fuse/model/validate.hpp) is expected to have already
/// refused a graph whose `hdf5_path`s collide or whose `*.h5` wildcards
/// are malformed before this is ever called on it, but this does not
/// trust that it was: a caller skipping validate() still gets a refusal,
/// never a silently wrong table.
[[nodiscard]] inline Result<LowFiveFieldBindingTable> build_lowfive_bindings(
    const CouplingGraph& g) {
  LowFiveFieldBindingTable table;
  for (std::uint32_t i = 0; i < g.field_count(); ++i) {
    const FieldId    fid{i};
    const FieldSpec& f = g.field(fid);
    if (!f.hdf5_path.has_value()) continue;
    const ComponentId owner = g.owner_of(fid);
    std::string qualified = f.name;
    if (owner.valid() && owner.value() < g.component_count()) {
      qualified = g.component(owner).name + "." + f.name;
    }
    auto r = table.add(LowFiveFieldBinding{.field_name = std::move(qualified),
                                           .hdf5_path = *f.hdf5_path});
    if (!r.ok()) return r.error();
  }
  return table;
}

}  // namespace fuse
