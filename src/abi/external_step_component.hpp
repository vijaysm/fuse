#pragma once

/// Private to src/abi/. See context.hpp's header comment for why this class
/// is a no-op: the component it stands in for is driven by an external
/// caller through fuse_begin_step()/fuse_end_step(), entirely outside
/// fuse::Runner's own call stack, so by the time Runner::run_period() calls
/// advance() here the registered storage already holds the right values.

#include <string>

#include "fuse/field/registry.hpp"
#include "fuse/model/coupling_graph.hpp"
#include "fuse/runtime/component.hpp"

namespace fuse::abi_detail {

class ExternalStepComponent final : public fuse::Component {
 public:
  explicit ExternalStepComponent(std::string name) : name_(std::move(name)) {}

  [[nodiscard]] std::string name() const override { return name_; }

  /// A no-op: this component's storage is bound directly by
  /// fuse_register_field() (src/abi/participant.cpp), before
  /// fuse_initialize() ever constructs a fuse::Runner, so there is nothing
  /// left for Runner::initialize() to ask this override to do.
  fuse::Result<void> register_fields(fuse::FieldRegistry&,
                                     const fuse::CouplingGraph&) override {
    return fuse::Result<void>::success();
  }

  /// A no-op: the external caller already wrote this component's Provides
  /// fields, through the pointer it registered, during the window between
  /// its own fuse_begin_step() and fuse_end_step() calls -- which is also
  /// where it read this component's Requires fields, themselves made
  /// current by the PRIOR real run_period() (see context.hpp's header
  /// comment on the lag-1 requirement this relies on).
  fuse::Result<void> advance(const fuse::StepContext&) override {
    return fuse::Result<void>::success();
  }

 private:
  std::string name_;
};

}  // namespace fuse::abi_detail
