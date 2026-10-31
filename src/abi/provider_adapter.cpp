#include "provider_adapter.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <unordered_set>

#include "fuse/abi/rank_share.hpp"
#include "fuse/base/rational.hpp"
#include "fuse/field/field_ref.hpp"
#include "fuse/model/capabilities.hpp"

namespace fuse::abi_detail {

namespace {

// abi-design.md rule 2. Correct for fuse_provider_instance and
// fuse_capability_set, which (c-abi.md's own growth table) have each
// shipped exactly one frozen layout so far -- "smaller than the oldest
// known layout" and "smaller than sizeof(T)" agree for those two. NOT
// correct for fuse_step_report (see step_report_struct_size_ok below):
// that struct HAS grown, and this template's
// ">= sizeof(T) [current]" bound refuses a plugin built against an
// older header's smaller-but-still-valid struct_size, the same
// backward-compatibility deviation participant.cpp's own
// FirstFrozenStructSize<T> avoids in the opposite direction.
template <class T>
[[nodiscard]] bool struct_size_ok(const T* s) noexcept {
  return s != nullptr && s->struct_size >= sizeof(T) &&
         s->struct_size <= static_cast<std::uint32_t>(FUSE_ABI_MAX_STRUCT_SIZE);
}

// The read-side mirror of
// FirstFrozenStructSize<fuse_step_report> (src/abi/participant.cpp) --
// this is a DIFFERENT translation unit from a plugin's own `advance()`,
// writing INTO this host's `rep`, so the question is "does the
// plugin's struct_size cover the members THIS HOST reads" (t_begin,
// t_end only -- see advance() below), not "is struct_size at least
// this host's current sizeof(fuse_step_report)". A plugin built
// against the pre-growth 64-byte layout (c-abi.md's growth table,
// commit da33572's predecessor) never had next_component/step_opened/
// period/delivery_pending/delivery_field to write, and this host never
// reads any of them from a plugin's own report -- only t_begin/t_end,
// both inside that original 64 bytes.
[[nodiscard]] bool step_report_struct_size_ok(const fuse_step_report* s) noexcept {
  constexpr std::size_t kFirstFrozenSize =
      offsetof(fuse_step_report, t_end) + sizeof(fuse_exact_time);
  return s != nullptr && s->struct_size >= kFirstFrozenSize &&
         s->struct_size <= static_cast<std::uint32_t>(FUSE_ABI_MAX_STRUCT_SIZE);
}

// fuse_exact_time and fuse::ExactTimeRecord are the same two-int64 layout
// (participant.cpp asserts this once; repeated here because this
// translation unit does not include that one).
static_assert(sizeof(fuse_exact_time) == sizeof(fuse::ExactTimeRecord));

[[nodiscard]] fuse_exact_time to_abi_time(const fuse::Rational& r) noexcept {
  const fuse::ExactTimeRecord rec = r.to_record();
  return fuse_exact_time{rec.numerator, rec.denominator};
}

[[nodiscard]] bool same_time(fuse_exact_time a, fuse_exact_time b) noexcept {
  return a.numerator == b.numerator && a.denominator == b.denominator;
}

// Member-count cross-check (abi-design.md point 5 / the same device
// src/abi/participant.cpp's own kEntryCountFromDef uses on the participant
// table): counts FUSE_ABI_PROVIDER_ENTRY expansions in entries.def
// independently of provider.h's own member list, so a hand-edited
// provider.h that silently drops or duplicates an entry is caught here at
// compile time rather than only by the golden-header test.
constexpr int kProviderEntryCountFromDef = 0
#define FUSE_ABI_PROVIDER_ENTRY(name, in, out, doc) +1
#include "fuse/abi/entries.def"
#undef FUSE_ABI_PROVIDER_ENTRY
    ;

static_assert(
    (sizeof(fuse_provider_abi_v1) - offsetof(fuse_provider_abi_v1, version)) /
        sizeof(void (*)()) ==
    static_cast<std::size_t>(kProviderEntryCountFromDef),
    "fuse_provider_abi_v1's member count disagrees with entries.def's "
    "FUSE_ABI_PROVIDER_ENTRY count; regenerate provider.h (tools/gen-abi.py --write)");

}  // namespace

ProviderAdapter::ProviderAdapter(std::string component_name,
                                 const fuse_provider_abi_v1* table)
    : name_(std::move(component_name)), table_(table) {}

ProviderAdapter::ProviderAdapter(std::string component_name)
    : name_(std::move(component_name)), table_(nullptr), unhosted_shell_(true) {}

ProviderAdapter::~ProviderAdapter() {
  if (self_ != nullptr && table_ != nullptr && table_->destroy != nullptr) {
    fuse_null in{sizeof(fuse_null)};
    fuse_null out{sizeof(fuse_null)};
    (void)table_->destroy(self_, &in, &out);
  }
}

fuse::Result<void> ProviderAdapter::register_fields(fuse::FieldRegistry& registry,
                                                     const fuse::CouplingGraph& graph) {
  if (table_ == nullptr) {
    if (!unhosted_shell_) {
      return fuse::Failure{"E-ABI-PROVIDER-TABLE",
                           "provider '" + name_ + "': no table (dlopen or version negotiation failed earlier)"};
    }
    // This shell was
    // built for a rank that does NOT host the component (src/abi/
    // participant.cpp's own impl_initialize_local(), the caller of this
    // constructor) -- there is no table to describe()/bind() through,
    // by construction, so this branch never reaches the plugin's own
    // shared object at all. It registers exactly what a
    // "mount everywhere, then discover count == 0" path would
    // produce for a non-hosting rank: the SAME fields, in the SAME
    // graph-declared order, each an EMPTY (zero-count) Host view --
    // reading each field's shape (components) from the GRAPH's own
    // FieldSpec, the only source of it available here, instead of from
    // the plugin's own describe() response.
    //
    // WHAT THIS LOSES, STATED: the E-ABI-
    // DESCRIBE cross-check against the graph (does the plugin's own
    // describe() agree with what the graph declares for this
    // component) never runs on this rank, because this rank never asks
    // the plugin anything. This does not weaken the check itself: EVERY
    // rank that DOES host this component (there must be at least one --
    // RankSet validate()'s own E-PLACEMENT-EMPTY refuses a component
    // with channels and no placement at all) still calls the real
    // describe() and runs the identical cross-check, against the
    // identical graph, against the identical compiled plugin -- the
    // mismatch this check exists to catch is a property of the PLUGIN
    // BINARY and the GRAPH, neither of which varies by rank, so a
    // hosting rank catches it exactly as it always did. What DOES
    // change: a plugin/graph mismatch is now a RANK-LOCAL refusal
    // (only hosting ranks ever see it) rather than one every rank
    // independently re-derived and agreed on by construction.
    //
    // The SAME field's own `memspace` (FUSE_MEMSPACE_HOST/_DEVICE) is
    // also unavailable here for the identical reason: it is reported by
    // describe(), not declared anywhere in the GRAPH's own FieldSpec.
    // Every field this shell registers is therefore Host-memspace,
    // regardless of what a hosting rank's own describe() would report
    // for the SAME field name -- harmless for an empty, zero-count
    // view (nothing is ever read through it: Runner::hosts_component()
    // never calls advance() for a component this rank does not host,
    // the same existing gate that already makes the plugin's own bind()
    // a no-op for any zero-count field), and it can
    // only ever make src/runtime/runner.cpp's own E-RT-SPACE check
    // STRICTER in this rank's favour (Host never trips that check),
    // never looser.
    const auto cid = graph.find_component(name_);
    if (!cid.has_value()) {
      return fuse::Failure{"E-ABI-PROVIDER-GRAPH",
                           "provider '" + name_ + "' is not a component of the loaded graph"};
    }
    for (const fuse::FieldId fid : graph.component(*cid).fields) {
      const fuse::FieldSpec& spec = graph.field(fid);
      const fuse::Layout layout{.stride = static_cast<std::size_t>(spec.components),
                                  .component_stride = 1,
                                  .count = 0,
                                  .components = spec.components};
      fuse::Result<void> bound =
          registry.bind(fid, fuse::FieldRef{nullptr, layout, fuse::MemorySpace::Host});
      if (!bound.ok()) return bound;
    }
    return fuse::Result<void>::success();
  }
  if (table_->create == nullptr || table_->destroy == nullptr ||
      table_->describe == nullptr || table_->bind == nullptr ||
      table_->advance == nullptr) {
    return fuse::Failure{"E-ABI-PROVIDER-TABLE",
                         "provider '" + name_ + "': table is missing a required entry "
                         "(create, destroy, describe, bind or advance)"};
  }

  try {
    // ---- create() ---------------------------------------------------
    fuse_null in0{sizeof(fuse_null)};
    fuse_provider_instance created{sizeof(fuse_provider_instance)};
    const fuse_status cr = table_->create(nullptr, &in0, &created);
    if (cr != FUSE_OK) {
      return fuse::Failure{"E-ABI-PROVIDER-CREATE",
                           "provider '" + name_ + "': create() returned " + std::to_string(cr)};
    }
    if (!struct_size_ok(&created)) {
      return fuse::Failure{"E-ABI-VERSION",
                           "provider '" + name_ + "': create() wrote a struct_size this host does not recognise"};
    }
    self_ = created.self;
    if (self_ == nullptr) {
      return fuse::Failure{"E-ABI-PROVIDER-CREATE",
                           "provider '" + name_ + "': create() returned FUSE_OK with a null instance"};
    }

    // ---- capabilities() (optional; absence means "no checkpoint") ----
    if (table_->capabilities != nullptr) {
      fuse_null cin{sizeof(fuse_null)};
      fuse_capability_set caps{sizeof(fuse_capability_set)};
      const fuse_status cs = table_->capabilities(self_, &cin, &caps);
      // An ENTRY THAT EXISTS but returns a non-OK status (or a struct_size
      // this host does not recognise) is refused: ignoring it would degrade
      // to "no checkpoint" with no refusal, indistinguishable from the
      // provider correctly reporting it has nothing to checkpoint. A plugin that offers the entry and then
      // cannot honour it is a defect this ABI boundary must surface, not
      // paper over.
      if (cs != FUSE_OK) {
        return fuse::Failure{"E-ABI-PROVIDER-CAPABILITIES",
                             "provider '" + name_ + "': capabilities() returned " +
                                 std::to_string(cs)};
      }
      if (!struct_size_ok(&caps)) {
        return fuse::Failure{"E-ABI-VERSION",
                             "provider '" + name_ +
                                 "': capabilities() wrote a struct_size this host does not "
                                 "recognise"};
      }
      // Determinism is left UNDECLARED (the determinism field's raw 0,
      // capabilities.hpp's own default) rather than claimed: the provider
      // ABI has no entry through which a plugin states a determinism
      // level, so silence must decode to std::nullopt, not a guess.
      capabilities_ = (((caps.flags & FUSE_CAP_CHECKPOINT) != 0) ? fuse::kCapabilityRollback
                                                                  : std::uint64_t{0}) |
                      // A plugin that sets
                      // FUSE_CAP_RESTART_FORM_B is declaring save_restart_blob()/
                      // load_restart_blob() are real, not the inherited no-op --
                      // the SAME posture FUSE_CAP_CHECKPOINT already takes for
                      // save_state/restore_state, one bit over.
                      (((caps.flags & FUSE_CAP_RESTART_FORM_B) != 0)
                           ? fuse::kCapabilityRestartFormB
                           : std::uint64_t{0}) |
                      // A provider's own instance is opaque past the fields it
                      // binds; FUSE's byte-for-byte rollback check can never see
                      // state the plugin keeps that was never described and
                      // bound (component.hpp's own caveat on carries_hidden_state()).
                      fuse::kCapabilityHiddenState;
      // FUSE_CAP_DEVICE_FIELDS
      // has no fuse::Component-side bit of its own (unlike FUSE_CAP_CHECKPOINT/
      // FUSE_CAP_RESTART_FORM_B above) -- nothing else in the runtime queries
      // "does this component support device fields" as a general capability
      // the way it queries rollback/determinism/hidden-state; this is purely
      // an internal consistency check register_fields() (below) makes between
      // TWO values this SAME plugin reported (this flag, and each field's own
      // describe()-row memspace), so it is kept as its own raw bool rather
      // than folded into capabilities_'s fuse::Component bitmask.
      device_fields_capability_ = (caps.flags & FUSE_CAP_DEVICE_FIELDS) != 0;
    }

    // ---- describe() ---------------------------------------------------
    fuse_null din{sizeof(fuse_null)};
    fuse_field_table count_query{sizeof(fuse_field_table)};
    count_query.buffer = nullptr;
    count_query.capacity = 0;
    const fuse_status ds0 = table_->describe(self_, &din, &count_query);
    if (ds0 != FUSE_OK) {
      return fuse::Failure{"E-ABI-PROVIDER-DESCRIBE",
                           "provider '" + name_ + "': describe() (size query) returned " +
                               std::to_string(ds0)};
    }
    std::vector<fuse_field_spec> rows(count_query.written);
    for (auto& row : rows) row.struct_size = sizeof(fuse_field_spec);
    if (!rows.empty()) {
      fuse_field_table fill{sizeof(fuse_field_table)};
      fill.buffer = rows.data();
      fill.capacity = rows.size();
      const fuse_status ds1 = table_->describe(self_, &din, &fill);
      if (ds1 != FUSE_OK) {
        return fuse::Failure{"E-ABI-PROVIDER-DESCRIBE",
                             "provider '" + name_ + "': describe() (fill) returned " +
                                 std::to_string(ds1)};
      }
    }
    std::unordered_map<std::string, const fuse_field_spec*> described;
    for (const auto& row : rows) {
      if (row.name != nullptr) described.emplace(row.name, &row);
    }

    // ---- cross-check against the graph's own declaration (E-ABI-DESCRIBE) --
    const auto cid = graph.find_component(name_);
    if (!cid.has_value()) {
      return fuse::Failure{"E-ABI-PROVIDER-GRAPH",
                           "provider '" + name_ + "' is not a component of the loaded graph"};
    }
    const auto& fields = graph.component(*cid).fields;
    storage_.clear();
    storage_.resize(fields.size());
    device_storage_.clear();
    device_storage_.resize(fields.size());
    has_device_fields_ = false;

    // Whether THIS process actually hosts this component on the REAL base
    // communicator -- hosting_rank_ is nullopt (treat as hosted) for every
    // single-process caller.
    const bool hosted =
        !hosting_rank_.has_value() || graph.component(*cid).ranks.contains(*hosting_rank_);

    // The cross-check below also refuses a plugin reporting a SUPERSET (an
    // extra row the graph never declared for this component), not only a
    // missing field. providers.md's own text
    // ("a describe() that disagrees with the graph's own field
    // declaration ... never silently skipped or substituted") does not
    // carve out an exception for extra rows, so this direction is refused
    // with the same E-ABI-DESCRIBE code the missing-field direction uses.
    {
      std::unordered_set<std::string> declared_names;
      declared_names.reserve(fields.size());
      for (const fuse::FieldId fid : fields) declared_names.insert(graph.field(fid).name);
      for (const auto& row : rows) {
        if (row.name != nullptr && !declared_names.contains(row.name)) {
          return fuse::Failure{
              "E-ABI-DESCRIBE",
              "provider '" + name_ + "': describe() reports field '" + row.name +
                  "', which the graph does not declare for this component"};
        }
      }
    }

    // SerialMeshBackend::region_size() indexes its OWN loaded-mesh vector
    // directly by MeshId (src/backend/serial_backend.cpp), which requires
    // every mesh 0..N-1 to have been load()ed in that order first -- the
    // same reason src/abi/participant.cpp's impl_initialize() loads every
    // mesh in the graph before registering any field, not just the ones a
    // given component's own fields happen to reference.
    for (std::uint32_t m = 0; m < graph.mesh_count(); ++m) {
      const fuse::Result<void> lr = backend_.load(graph.mesh(fuse::MeshId{m}));
      if (!lr.ok()) return lr;
    }

    for (std::size_t i = 0; i < fields.size(); ++i) {
      const fuse::FieldId   fid  = fields[i];
      const fuse::FieldSpec& spec = graph.field(fid);
      const auto it = described.find(spec.name);
      if (it == described.end()) {
        return fuse::Failure{"E-ABI-DESCRIBE",
                             "provider '" + name_ + "': the graph declares field '" + spec.name +
                                 "' but describe() does not report it"};
      }
      const fuse_field_spec& row = *it->second;
      const std::int32_t expected_intent = static_cast<std::int32_t>(spec.intent);
      if (row.components != spec.components ||
          (row.intent != expected_intent && row.intent != 2 && expected_intent != 2)) {
        return fuse::Failure{
            "E-ABI-DESCRIBE",
            "provider '" + name_ + "': describe() disagrees with the graph on field '" +
                spec.name + "' (graph: components=" + std::to_string(spec.components) +
                " intent=" + std::to_string(expected_intent) +
                "; describe(): components=" + std::to_string(row.components) +
                " intent=" + std::to_string(row.intent) + ")"};
      }
      if (row.element_type != FUSE_ELEMENT_F64) {
        return fuse::Failure{"E-ABI-PROVIDER-ELEMENT",
                             "provider '" + name_ + "': field '" + spec.name +
                                 "' is not FUSE_ELEMENT_F64, the only element type this host supports"};
      }

      // A field whose describe() row requests FUSE_MEMSPACE_DEVICE needs BOTH gates
      // below to pass before this adapter allocates it device-resident --
      // neither one alone is trusted. `row.memspace` defaults to
      // FUSE_MEMSPACE_HOST (0) for a struct_size an old caller set
      // before this field existed (abi-design.md rule 2's documented
      // default), so an old plugin recompiled against this header but
      // never touching the new field behaves exactly as it always did.
      const bool wants_device = row.memspace == FUSE_MEMSPACE_DEVICE;
      // A value neither HOST nor DEVICE is a request this host's
      // FUSE_MEMSPACE_* enum does not recognise -- refused by name
      // rather than silently treated as host, the same "no silent
      // fallback" posture the two gates below apply to a device request.
      if (row.memspace != FUSE_MEMSPACE_HOST && row.memspace != FUSE_MEMSPACE_DEVICE) {
        return fuse::Failure{"E-ABI-PROVIDER-MEMSPACE",
                             "provider '" + name_ + "': field '" + spec.name +
                                 "' describe() reports memspace=" + std::to_string(row.memspace) +
                                 ", which is neither FUSE_MEMSPACE_HOST nor FUSE_MEMSPACE_DEVICE"};
      }
      if (wants_device) {
        // Gate 1: the plugin's OWN capabilities() must have granted this
        // BEFORE describe() is trusted to ask for it -- a plugin that
        // asks for device storage without declaring the capability is an
        // internally inconsistent plugin, refused by name rather than
        // silently downgraded to host (this project's own "no silent
        // fallback" convention, named repeatedly elsewhere in this tree,
        // e.g. lowfive_attach_decision.hpp's own E-LOWFIVE-ATTACH-
        // NOCHANNELS). The opposite -- capabilities() sets the bit but no
        // field ever asks for FUSE_MEMSPACE_DEVICE -- is unremarkable: a
        // provider is always free to offer more than any one graph uses.
        if (!device_fields_capability_) {
          return fuse::Failure{
              "E-ABI-PROVIDER-DEVICE-UNDECLARED",
              "provider '" + name_ + "': field '" + spec.name +
                  "' describe() reports FUSE_MEMSPACE_DEVICE, but this provider's own "
                  "capabilities() did not report FUSE_CAP_DEVICE_FIELDS"};
        }
        // Gate 2: this BUILD must compile in a real device backend. A
        // host-only build must refuse a device-field request by name, never silently hand
        // back host-addressable memory under the device label, even
        // though Kokkos::DefaultExecutionSpace::memory_space legally
        // equals Kokkos::HostSpace on such a build and nothing would
        // crash if it did. Compiled out cleanly, not a runtime branch
        // that happens to always take the refusal arm: on a build with
        // no CUDA/HIP/SYCL, the #else arm below is the ONLY code that
        // exists for this condition.
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
        // (allocation happens below, in the per-field storage block)
#else
        return fuse::Failure{
            "E-ABI-PROVIDER-NO-DEVICE",
            "provider '" + name_ + "': field '" + spec.name +
                "' describe() reports FUSE_MEMSPACE_DEVICE, but this build has no device "
                "backend (Kokkos was compiled without CUDA, HIP or SYCL) -- a device-resident "
                "field cannot be honoured here, and this host never silently allocates one on "
                "the CPU under the device label"};
#endif
      }

      // A component that does not live on this rank gets an empty view of
      // the right dimension rather than a failure, because every rank
      // resolves every placement (the exchange plan is collective) while
      // only hosting ranks read one. The REAL plan (built from
      // fuse_ctx::backend, a rank-aware fuse::native::NativeMeshBackend)
      // follows this rule; this adapter's own private backend_
      // (fuse::SerialMeshBackend, this class's own header comment: a
      // single-address-space backend with no concept of a rank or a RankSet
      // at all) cannot -- it always reports the region's FULL size, on every
      // rank, regardless of whether this component is even placed there.
      // A Plugin component mounted on a rank SUBSET would register the full
      // size on a NON-hosting rank, disagreeing with the plan's own 0-entry
      // expectation there -- "a short registration is a heap overflow in the
      // hot loop" is the Runner's own refusal for the OPPOSITE direction;
      // this is that same contract, violated from the adapter's side.
      // So `hosted` (above), not backend_.region_size(), decides whether
      // this rank registers the region's real size or an empty view, rather
      // than teaching backend_ itself to understand a RankSet
      // (SerialMeshBackend's whole point, this class's own header comment,
      // is NOT to).
      //
      // `hosted` alone is still too coarse: it only answers RankSet
      // MEMBERSHIP, a binary question -- correct for a ONE-rank component,
      // wrong for a component spanning MORE than one rank. A 3-rank run
      // placed "upstream" on RankSet{1, 2} (two ranks): the REAL plan
      // decomposes its one-cell region over THOSE two ranks by POSITION
      // (the rule NativeMeshBackend applies to every placement), giving
      // rank 1 (position 0 of 2) the one cell and rank 2 (position 1 of 2)
      // NONE -- but `hosted` is true for BOTH (both are RankSet members), so
      // registering the FULL size (1) on rank 2 too would disagree with the
      // plan's own 0-entry expectation there ("field 'upstream.out' is
      // registered with 1x1 values but the plan sizes its region at 0x1").
      // This is separate from the question of WHETHER a non-hosting rank's
      // own ProviderAdapter exists at all (Registry::mount_dynamic() mounts
      // the plugin on every rank regardless of RankSet): rank 2 is squarely
      // a RankSet member here.
      //
      // So the SAME per-position split the plan's own backend applies is
      // applied here, via fuse::abi_detail::even_rank_share()
      // (include/fuse/abi/rank_share.hpp -- a duplicate of fuse::native::
      // rank_slice's own formula, not a call to it; see that header's own
      // doc comment for why duplicated rather than reused). For a
      // single-process caller (`hosting_rank_` unset), `group_size` is
      // always effectively 1 (nothing to split), so this degenerates to
      // the full region size.
      const fuse::Region& region = graph.region(spec.region);
      std::size_t count = 0;
      if (hosted) {
        const fuse::Result<std::size_t> sz = backend_.region_size(region);
        if (!sz.ok()) return fuse::Result<void>{sz.error()};
        const std::size_t full = sz.value();
        if (hosting_rank_.has_value()) {
          const fuse::RankSet& ranks = graph.component(*cid).ranks;
          const int position = ranks.index_of(*hosting_rank_);
          // `hosted` is already true, so `position >= 0` here: this rank
          // IS a member of its own component's RankSet by construction.
          const auto [lo, hi] = fuse::abi_detail::even_rank_share(
              full, position, static_cast<int>(ranks.size()));
          count = hi - lo;
        } else {
          count = full;  // single-process caller: the full size.
        }
      }

      // `base` and the MemorySpace FUSE's own FieldRegistry is told about both depend on
      // `wants_device` (validated above, by this point either impossible
      // or already proven to pass both gates) -- exactly ONE of
      // `storage_[i]`/`device_storage_[i]` is ever populated for a given
      // field; the other stays default-constructed and empty.
      double* base = nullptr;
      fuse::MemorySpace field_space = fuse::MemorySpace::Host;
      if (wants_device) {
        Kokkos::resize(device_storage_[i], count * static_cast<std::size_t>(spec.components));
        base = device_storage_[i].data();
        field_space = fuse::MemorySpace::Device;
        has_device_fields_ = true;
      } else {
        storage_[i].assign(count * static_cast<std::size_t>(spec.components), 0.0);
        base = storage_[i].data();
      }

      const fuse::Layout layout{.stride = static_cast<std::size_t>(spec.components),
                                .component_stride = 1,
                                .count = count,
                                .components = spec.components};
      fuse::Result<void> bound =
          registry.bind(fid, fuse::FieldRef{base, layout, field_space});
      if (!bound.ok()) return bound;

      // The PLUGIN's own bind() is skipped entirely when `count == 0` --
      // `base` (storage_[i].data() on a just-.assign(0, ...)'d, possibly
      // EMPTY vector) is not guaranteed non-null by the standard, and
      // linear_provider.c's own p_bind() (the reference plugin) refuses any
      // bind() whose `data` is NULL, by design (a defensive check every
      // well-behaved provider is entitled to make). FUSE's own
      // registry.bind() above already recorded the empty view -- every rank
      // still resolves this field's own placement -- so the plugin loses
      // nothing it was ever going to read from (Runner::hosts_component()
      // never calls advance() for this component on a non-hosting rank
      // either).
      //
      // This is gated on `count > 0`, not on `hosted` alone: `hosted` and
      // `count > 0` agree for a ONE-rank component, but a MULTI-rank
      // component's own hosting rank can still own ZERO elements at this
      // rank's particular POSITION (the even-split remainder rule above),
      // and calling the plugin's own bind() with a NULL `base` would be
      // refused by the SAME p_bind() check, for the SAME reason.
      // `count > 0` is exactly the condition `base` is non-null for in
      // practice (and the condition that matters: whether this rank has
      // anything of the region to hand the plugin at all, not merely
      // whether it is a RankSet member).
      if (count > 0) {
        fuse_field_view view{sizeof(fuse_field_view)};
        view.handle            = static_cast<std::int32_t>(fid.value());
        view.element_type      = FUSE_ELEMENT_F64;
        view.memspace          = wants_device ? FUSE_MEMSPACE_DEVICE : FUSE_MEMSPACE_HOST;
        view.ownership         = FUSE_OWNERSHIP_OWNED;
        view.components        = spec.components;
        view.count             = count;
        view.owned_count       = count;
        view.stride             = static_cast<std::int64_t>(spec.components);
        view.component_stride    = 1;
        view.data               = base;
        fuse_null bout{sizeof(fuse_null)};
        const fuse_status br = table_->bind(self_, &view, &bout);
        if (br != FUSE_OK) {
          return fuse::Failure{"E-ABI-PROVIDER-BIND",
                               "provider '" + name_ + "': bind() of field '" + spec.name +
                                   "' returned " + std::to_string(br)};
        }
      }
    }
    return fuse::Result<void>::success();
  } catch (const std::exception& e) {
    return fuse::Failure{"E-ABI-PROVIDER-EXCEPTION",
                         "provider '" + name_ + "': " + e.what()};
  }
}

fuse::Result<void> ProviderAdapter::advance(const fuse::StepContext& ctx) {
  if (table_ == nullptr || table_->advance == nullptr) {
    return fuse::Failure{"E-ABI-PROVIDER-TABLE",
                         "provider '" + name_ + "': table has no advance() entry"};
  }
  try {
    fuse_step_request req{sizeof(fuse_step_request)};
    req.component = name_.c_str();
    req.t_begin = to_abi_time(ctx.t_begin);
    req.t_end   = to_abi_time(ctx.t_end);
    req.step    = ctx.step;
    req.iteration = ctx.iteration;
    req.requires_checkpoint = 0;
    req.requires_restore    = 0;
    req.reserved = 0;
    fuse_step_report rep{sizeof(fuse_step_report)};
    const fuse_status rc = table_->advance(self_, &req, &rep);
    if (rc != FUSE_OK) {
      return fuse::Failure{"E-ABI-PROVIDER-ADVANCE",
                           "provider '" + name_ + "': advance() returned " + std::to_string(rc)};
    }
    if (!step_report_struct_size_ok(&rep)) {
      return fuse::Failure{"E-ABI-VERSION",
                           "provider '" + name_ + "': advance() wrote a struct_size this host does not recognise"};
    }
    // Rule 6: the window a provider echoes back must be EXACTLY the one it
    // was asked to run -- not rounded, not approximated.
    if (!same_time(rep.t_begin, req.t_begin) || !same_time(rep.t_end, req.t_end)) {
      return fuse::Failure{"E-ABI-PROVIDER-WINDOW",
                           "provider '" + name_ + "': advance() echoed a different window than it was asked to run"};
    }
    // A device-touching provider's own advance() may have launched an ASYNCHRONOUS device
    // kernel through whichever device pointer bind() handed it -- the
    // table's own fuse_status return says the LAUNCH succeeded, never
    // that the kernel has FINISHED. Without this fence, the Runner's own
    // very next read of that same registered storage (a native kernel,
    // a Transport pack, a Remap) could race the plugin's own kernel.
    // This fence is DEFENSIVE. The design never depends on it, nor on a device pointer being visible across
    // streams; a plugin completes its own device work before advance()
    // returns, and no test, row or doc states a property in terms of this
    // fence. It stays (robustness over efficiency).
    // Unconditional for ANY provider with at least one device field --
    // no escape hatch, not even for a provider that claims to fence
    // internally (a redundant fence from an already-quiescent device costs
    // one no-op call, not a measurable regression; see docs/dev/abi-design.md's own device-fields
    // subsection for the host_pending measurement). Never paid by a
    // host-only provider (has_device_fields_ stays false for it).
    if (has_device_fields_) {
      Kokkos::fence("provider_adapter_advance_fence");
    }
    return fuse::Result<void>::success();
  } catch (const std::exception& e) {
    return fuse::Failure{"E-ABI-PROVIDER-EXCEPTION",
                         "provider '" + name_ + "': " + e.what()};
  }
}

fuse::Result<void> ProviderAdapter::save_state() {
  if (table_ == nullptr || table_->save_state == nullptr) {
    return fuse::Result<void>::success();  // the default no-op (component.hpp)
  }
  fuse_null in{sizeof(fuse_null)};
  fuse_null out{sizeof(fuse_null)};
  const fuse_status rc = table_->save_state(self_, &in, &out);
  if (rc != FUSE_OK) {
    return fuse::Failure{"E-ABI-PROVIDER-SAVE",
                         "provider '" + name_ + "': save_state() returned " + std::to_string(rc)};
  }
  return fuse::Result<void>::success();
}

fuse::Result<void> ProviderAdapter::restore_state() {
  if (table_ == nullptr || table_->restore_state == nullptr) {
    return fuse::Result<void>::success();
  }
  fuse_null in{sizeof(fuse_null)};
  fuse_null out{sizeof(fuse_null)};
  const fuse_status rc = table_->restore_state(self_, &in, &out);
  if (rc != FUSE_OK) {
    return fuse::Failure{"E-ABI-PROVIDER-RESTORE",
                         "provider '" + name_ + "': restore_state() returned " + std::to_string(rc)};
  }
  return fuse::Result<void>::success();
}

/// Two calls to the SAME table entry, the
/// identical query-then-fill shape fuse_graph_sink's own CALLEE side
/// already establishes (participant.cpp's impl_save_graph) -- here as the
/// CALLER: first with buffer=NULL to learn `written`, then with a buffer
/// of that exact size. `table_->save_restart_blob == nullptr` (the entry
/// was never offered) and `out.written == 0` (the plugin has nothing to
/// capture) are BOTH the Component base class's own "empty blob" default
/// -- never an error, since "nothing to restart" is a legitimate answer,
/// not a missing feature.
fuse::Result<std::vector<std::byte>> ProviderAdapter::save_restart_blob() {
  last_blob_version_ = 0;
  if (table_ == nullptr || table_->save_restart_blob == nullptr) {
    return std::vector<std::byte>{};
  }
  fuse_null in{sizeof(fuse_null)};
  fuse_restart_blob_sink query{sizeof(fuse_restart_blob_sink)};
  fuse_status rc = table_->save_restart_blob(self_, &in, &query);
  if (rc != FUSE_OK) {
    return fuse::Failure{"E-ABI-PROVIDER-RESTART-SAVE",
                         "provider '" + name_ +
                             "': save_restart_blob() (size query) returned " +
                             std::to_string(rc)};
  }
  if (query.written == 0) {
    last_blob_version_ = query.blob_version;
    return std::vector<std::byte>{};
  }
  std::vector<std::byte> bytes(query.written);
  fuse_restart_blob_sink fill{sizeof(fuse_restart_blob_sink)};
  fill.buffer   = bytes.data();
  fill.capacity = bytes.size();
  rc = table_->save_restart_blob(self_, &in, &fill);
  if (rc != FUSE_OK) {
    return fuse::Failure{"E-ABI-PROVIDER-RESTART-SAVE",
                         "provider '" + name_ +
                             "': save_restart_blob() (fill) returned " +
                             std::to_string(rc)};
  }
  if (fill.written != query.written) {
    return fuse::Failure{
        "E-ABI-PROVIDER-RESTART-SAVE",
        "provider '" + name_ + "': save_restart_blob() reported " +
            std::to_string(query.written) +
            " bytes on the size query and " + std::to_string(fill.written) +
            " on the fill call -- the two calls must agree, or the buffer "
            "this adapter allocated from the FIRST answer does not hold "
            "everything the SECOND call tried to write"};
  }
  last_blob_version_ = fill.blob_version;
  return bytes;
}

fuse::Result<void> ProviderAdapter::load_restart_blob(
    std::span<const std::byte> bytes, std::uint32_t written_by_version) {
  if (table_ == nullptr || table_->load_restart_blob == nullptr) {
    return fuse::Result<void>::success();
  }
  fuse_restart_blob_source in{sizeof(fuse_restart_blob_source)};
  in.bytes        = bytes.data();
  in.byte_count   = bytes.size();
  in.blob_version = written_by_version;
  fuse_null out{sizeof(fuse_null)};
  const fuse_status rc = table_->load_restart_blob(self_, &in, &out);
  if (rc != FUSE_OK) {
    return fuse::Failure{"E-ABI-PROVIDER-RESTART-LOAD",
                         "provider '" + name_ +
                             "': load_restart_blob() returned " +
                             std::to_string(rc)};
  }
  return fuse::Result<void>::success();
}

}  // namespace fuse::abi_detail
