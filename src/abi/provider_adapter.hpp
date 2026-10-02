#pragma once

/// Wraps a loaded `fuse_provider_abi_v1` table (include/fuse/abi/provider.h)
/// as an ordinary `fuse::Component`, so `fuse::Runner` can drive a plugin
/// through the same in-process path it drives any native component through,
/// with no knowledge that a C ABI boundary sits underneath
/// (docs/dev/abi-design.md, "Provider direction").
///
/// STORAGE. `ProviderAdapter` -- not the plugin -- owns the field storage
/// (`fuse::Component::register_fields()`'s own contract: the implementer
/// owns the address and it must never move for the run's whole lifetime).
/// `register_fields()` allocates one buffer per field this component
/// declares, binds it into FUSE's own `FieldRegistry` the way any native
/// component would, and ALSO hands the plugin the same pointer through
/// `fuse_provider_abi_v1::bind` -- so FUSE's remap and the plugin's own
/// `advance()` read and write the identical bytes, never a copy.
///
/// Entity counts come from a private `fuse::SerialMeshBackend` this adapter
/// loads itself: a `Component`'s `register_fields()` is handed no backend
/// of its own (every example component either hardcodes its size or is
/// handed one at construction), and `ProviderAdapter` is generic over any
/// graph, so it resolves each field's region the same way the participant
/// ABI's own `fuse_ctx::backend` does.

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "fuse/abi/provider.h"
#include "fuse/backend/serial_backend.hpp"
#include "fuse/base/result.hpp"
#include "fuse/model/coupling_graph.hpp"
#include "fuse/runtime/component.hpp"

namespace fuse::abi_detail {

class ProviderAdapter final : public fuse::Component {
 public:
  /// `table` is borrowed and must outlive this object; `component_name`
  /// must equal a `ComponentDecl::name` of the graph `register_fields()`
  /// will be called with (the same contract every `fuse::Component`
  /// carries -- component.hpp's own doc on `name()`).
  ProviderAdapter(std::string component_name, const fuse_provider_abi_v1* table);
  ~ProviderAdapter() override;

  ProviderAdapter(const ProviderAdapter&) = delete;
  ProviderAdapter& operator=(const ProviderAdapter&) = delete;
  ProviderAdapter(ProviderAdapter&&) = delete;
  ProviderAdapter& operator=(ProviderAdapter&&) = delete;

  [[nodiscard]] std::string name() const override { return name_; }

  fuse::Result<void> register_fields(fuse::FieldRegistry& registry,
                                     const fuse::CouplingGraph& graph) override;
  fuse::Result<void> advance(const fuse::StepContext& ctx) override;
  fuse::Result<void> save_state() override;
  fuse::Result<void> restore_state() override;
  [[nodiscard]] std::uint64_t capabilities() const override { return capabilities_; }

 private:
  std::string                 name_;
  const fuse_provider_abi_v1* table_;
  fuse_provider_ctx*           self_ = nullptr;
  std::uint64_t                capabilities_ = 0;
  fuse::SerialMeshBackend      backend_;
  /// Owns the storage every field this component declares reads and writes;
  /// indexed the same order as `graph.component(id).fields` at
  /// `register_fields()` time, which is the order every later call
  /// (`advance()`, `save_state()`) relies on being fixed. Never resized
  /// after `register_fields()` returns successfully (the same "address
  /// must not move" rule `t_`/`scratch_` observe in
  /// examples/mms_pair/heat_component.hpp).
  std::vector<std::vector<double>> storage_;
};

}  // namespace fuse::abi_detail
