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
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <Kokkos_Core.hpp>

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

  /// KNOWN_ISSUES.md D15 round 3: this process's own base-communicator
  /// rank, so register_fields() (below) can tell whether THIS process
  /// actually hosts the component `table` was mounted for. `std::nullopt`
  /// (the default -- every existing, single-process caller never calls
  /// this) means "assume hosted everywhere", which is what register_
  /// fields() always did before this round and is exactly right when
  /// there is only one rank. Set once, before register_fields() runs
  /// (participant.cpp's impl_initialize_local(), right after extracting
  /// this adapter from the plugin registry) -- never again after.
  void set_hosting_rank(std::optional<int> rank) { hosting_rank_ = rank; }

  [[nodiscard]] std::string name() const override { return name_; }

  fuse::Result<void> register_fields(fuse::FieldRegistry& registry,
                                     const fuse::CouplingGraph& graph) override;
  fuse::Result<void> advance(const fuse::StepContext& ctx) override;
  fuse::Result<void> save_state() override;
  fuse::Result<void> restore_state() override;
  /// Restart track round 2, item 4: forwards to the plugin's own
  /// save_restart_blob/load_restart_blob table entries (Option B,
  /// docs/dev/restart.md Section 3) when FUSE_CAP_RESTART_FORM_B was
  /// set at create() time; the inherited Component defaults (empty
  /// blob, version 0) apply otherwise, exactly as save_state()/
  /// restore_state() already fall back to a no-op when the table entry
  /// itself is null.
  fuse::Result<std::vector<std::byte>> save_restart_blob() override;
  fuse::Result<void> load_restart_blob(std::span<const std::byte> bytes,
                                       std::uint32_t written_by_version) override;
  [[nodiscard]] std::uint32_t restart_blob_version() const override {
    return last_blob_version_;
  }
  [[nodiscard]] std::uint64_t capabilities() const override { return capabilities_; }

 private:
  std::string                 name_;
  const fuse_provider_abi_v1* table_;
  fuse_provider_ctx*           self_ = nullptr;
  std::uint64_t                capabilities_ = 0;
  /// Raw FUSE_CAP_DEVICE_FIELDS bit from this plugin's own capabilities()
  /// call, kept separately from `capabilities_` -- see that member's own
  /// assignment site (provider_adapter.cpp) for why this has no
  /// fuse::Component-side bit of its own. Consulted only by
  /// register_fields()'s own per-field memspace gate.
  bool                          device_fields_capability_ = false;
  /// Set by save_restart_blob() from the plugin's own `out->blob_version`
  /// (the single source of truth for this run's own blob format), read
  /// back by restart_blob_version() -- the Runner calls these two methods
  /// in that order (src/runtime/runner.cpp's write_restart_fields()), so
  /// caching is correct, not a race.
  std::uint32_t                last_blob_version_ = 0;
  fuse::SerialMeshBackend      backend_;
  /// See set_hosting_rank()'s own comment above.
  std::optional<int>           hosting_rank_;
  /// Owns the storage every field this component declares reads and writes;
  /// indexed the same order as `graph.component(id).fields` at
  /// `register_fields()` time, which is the order every later call
  /// (`advance()`, `save_state()`) relies on being fixed. Never resized
  /// after `register_fields()` returns successfully (the same "address
  /// must not move" rule `t_`/`scratch_` observe in
  /// examples/mms_pair/heat_component.hpp). A field whose describe() row
  /// requested FUSE_MEMSPACE_DEVICE keeps its entry here empty and is
  /// allocated into `device_storage_` below instead -- the two vectors
  /// are parallel-indexed over the SAME field order, and exactly one of
  /// the two holds a given field's real storage.
  std::vector<std::vector<double>> storage_;

  /// KNOWN_ISSUES.md D15 item 1 ("Provider parity" track, Stage B-2,
  /// Option A): the device-resident counterpart of `storage_` above, one
  /// `Kokkos::View` per field this component declares, parallel-indexed
  /// the same way. Only a field whose describe() row requests
  /// FUSE_MEMSPACE_DEVICE -- and whose provider also sets
  /// FUSE_CAP_DEVICE_FIELDS in capabilities(), and whose BUILD compiles
  /// in a real device backend -- gets a non-empty View here; every other
  /// field's own entry is left default-constructed (an empty View costs
  /// nothing to keep around). `DeviceMemSpace` is
  /// `Kokkos::DefaultExecutionSpace::memory_space` -- on a build with no
  /// CUDA/HIP/SYCL this IS `Kokkos::HostSpace` (device-resident-arcs.md's
  /// own documented "Device mode on a CPU-only build" mechanism), but
  /// this adapter never reaches this branch on such a build at all: the
  /// host-only refusal (register_fields(), below) fires first, by the
  /// owner's own explicit instruction that a build with no device
  /// backend must refuse a device-field request by name rather than
  /// silently handing back host-addressable memory under the device
  /// label.
  using DeviceMemSpace = Kokkos::DefaultExecutionSpace::memory_space;
  std::vector<Kokkos::View<double*, DeviceMemSpace>> device_storage_;

  /// Set once, at the end of a successful register_fields(): true iff at
  /// least one field of this component is device-resident. advance()
  /// reads this to decide whether the post-advance Kokkos::fence() below
  /// is needed at all -- unconditional FOR A DEVICE-TOUCHING PROVIDER
  /// (the owner's own instruction: no escape hatch, not even for a
  /// provider that claims to fence itself), but never paid by a
  /// host-only provider, which is every provider in this tree before
  /// this round.
  bool has_device_fields_ = false;
};

}  // namespace fuse::abi_detail
