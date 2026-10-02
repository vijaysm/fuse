#pragma once

/// Private to src/abi/. See context.hpp's header comment for the resumable
/// scheme this class is the Runner-side half of: the component this stands
/// in for is driven by an external caller through fuse_begin_step()/
/// fuse_end_step(), so its advance() is a no-op that trusts the registered
/// storage already holds the right values -- by the time
/// Runner::Impl::advance() (src/runtime/runner.cpp) reaches it, every
/// exchange feeding this component has already run (KNOWN_ISSUES.md D12).

#include <cstddef>
#include <string>
#include <vector>

#include "fuse/field/registry.hpp"
#include "fuse/model/capabilities.hpp"
#include "fuse/model/coupling_graph.hpp"
#include "fuse/model/field_spec.hpp"
#include "fuse/runtime/component.hpp"

namespace fuse::abi_detail {

class ExternalStepComponent final : public fuse::Component {
 public:
  explicit ExternalStepComponent(std::string name) : name_(std::move(name)) {}

  [[nodiscard]] std::string name() const override { return name_; }

  /// KNOWN_ISSUES.md D12: the one thing this override does is collect, not
  /// bind -- every field this component owns is already bound directly by
  /// fuse_register_field() (src/abi/participant.cpp), before
  /// fuse_initialize() ever constructs a fuse::Runner. What this records
  /// is which of those ALREADY-bound views are this component's own
  /// `Intent::Provides` outputs, so save_state()/restore_state() below
  /// have something real to snapshot: entering an iteration region at all
  /// requires `supports_rollback()` (component.hpp), and a component whose
  /// save/restore were still the base class's no-op default could not
  /// honestly claim it -- Runner::Impl's own checkpoint verification
  /// (`R.verify_arcs`, src/runtime/iteration.cpp) would then find the
  /// registered storage NOT restored and refuse with E-ITER-ROLLBACK on
  /// the very first re-sweep.
  fuse::Result<void> register_fields(fuse::FieldRegistry& registry,
                                     const fuse::CouplingGraph& graph) override {
    const auto cid = graph.find_component(name_);
    if (!cid.has_value()) {
      return fuse::Failure{
          "E-ABI-EXTERNAL-COMPONENT",
          "ExternalStepComponent '" + name_ +
              "' does not name a component in the graph being initialized"};
    }
    provides_.clear();
    for (const fuse::FieldId fid : graph.component(*cid).fields) {
      if (graph.field(fid).intent != fuse::Intent::Provides) {
        // A Requires field is written by FUSE itself (Remap/Sink), never
        // by this component's own (externally performed) advance(), so
        // it needs no save/restore here -- and is not one of the arcs
        // R.verify_arcs checks against this component either.
        continue;
      }
      const fuse::Result<fuse::FieldRef> ref = registry.get(fid);
      if (!ref.ok()) continue;  // not bound; nothing to checkpoint
      provides_.push_back(ref.value());
    }
    return fuse::Result<void>::success();
  }

  /// A no-op: the external caller already wrote this component's Provides
  /// fields, through the pointer it registered, directly into the
  /// registered storage, before resuming past the suspension point this
  /// component's own `is_externally_driven()` causes (see
  /// src/runtime/runner.cpp / src/runtime/iteration.cpp).
  fuse::Result<void> advance(const fuse::StepContext&) override {
    return fuse::Result<void>::success();
  }

  /// Byte-exact snapshot of every Provides field `register_fields()`
  /// found, element by element through `FieldRef::at()` so an arbitrary
  /// (non-contiguous) layout is handled the same way the rest of the
  /// runtime handles one. At most one snapshot is ever live -- spec
  /// §5.3's own rule, mirrored in component.hpp's doc comment on this
  /// override -- so `saved_` is simply overwritten, never appended to.
  fuse::Result<void> save_state() override {
    saved_.resize(provides_.size());
    for (std::size_t f = 0; f < provides_.size(); ++f) {
      const fuse::FieldRef& ref = provides_[f];
      const auto count = ref.layout().count;
      const auto components = static_cast<std::size_t>(ref.layout().components);
      std::vector<double>& buf = saved_[f];
      buf.resize(count * components);
      std::size_t k = 0;
      for (std::size_t i = 0; i < count; ++i) {
        for (int c = 0; c < ref.layout().components; ++c) {
          buf[k++] = ref.at(i, c);
        }
      }
    }
    return fuse::Result<void>::success();
  }

  /// The inverse of save_state(): copies the snapshot back into the SAME
  /// registered storage, element by element, so Runner::Impl's own
  /// post-restore verification (`R.verify_arcs` memcmp) finds exactly the
  /// bytes save_state() captured -- true semantically, not just by the
  /// check's own bookkeeping, since nothing between this call and the
  /// region's next `Sink` delivery reads this component's Provides arrays.
  fuse::Result<void> restore_state() override {
    if (saved_.size() != provides_.size()) {
      return fuse::Failure{
          "E-ABI-EXTERNAL-RESTORE",
          "ExternalStepComponent '" + name_ +
              "': restore_state() called with no matching save_state()"};
    }
    for (std::size_t f = 0; f < provides_.size(); ++f) {
      fuse::FieldRef& ref = provides_[f];
      const std::vector<double>& buf = saved_[f];
      std::size_t k = 0;
      for (std::size_t i = 0; i < ref.layout().count; ++i) {
        for (int c = 0; c < ref.layout().components; ++c) {
          ref.at(i, c) = buf[k++];
        }
      }
    }
    return fuse::Result<void>::success();
  }

  /// Rollback is genuine (save_state()/restore_state() above are a real
  /// pair, not the base class's no-op) -- determinism and hidden-state
  /// are deliberately left UNDECLARED (the default, 0): this component's
  /// real physics runs entirely in the external caller's own process,
  /// which this header has no way to characterise on its behalf. An
  /// iteration region warns on an undeclared determinism level rather
  /// than refusing, which is the honest answer here.
  [[nodiscard]] std::uint64_t capabilities() const override {
    return fuse::kCapabilityRollback;
  }

  /// KNOWN_ISSUES.md D12's fix: the one override that makes
  /// `Runner::Impl::advance()`/`advance_window()` (src/runtime/runner.cpp,
  /// src/runtime/iteration.cpp) suspend at this component's own advance()
  /// point instead of calling the (still here, still a no-op) override
  /// above directly -- see component.hpp's own doc comment on this
  /// virtual.
  [[nodiscard]] bool is_externally_driven() const noexcept override {
    return true;
  }

 private:
  std::string name_;
  /// This component's own `Intent::Provides` fields, resolved once in
  /// register_fields(). Each is a copy of the FieldRef the registry
  /// already holds (FieldRef is a non-owning view; copying it copies the
  /// pointer, not the data -- field_ref.hpp's own trivially-copyable
  /// contract), so writes through `provides_[f].at(...)` in
  /// save_state()/restore_state() land in the SAME storage the external
  /// caller registered and FUSE's own Remap/Sink instructions read.
  std::vector<fuse::FieldRef> provides_;
  /// Parallel to `provides_`: the bytes save_state() last captured.
  std::vector<std::vector<double>> saved_;
};

}  // namespace fuse::abi_detail
