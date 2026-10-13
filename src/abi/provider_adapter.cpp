#include "provider_adapter.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <unordered_set>

#include "fuse/base/rational.hpp"
#include "fuse/field/field_ref.hpp"
#include "fuse/model/capabilities.hpp"

namespace fuse::abi_detail {

namespace {

// abi-design.md rule 2. Correct for fuse_provider_instance and
// fuse_capability_set, which (c-abi.md's own growth table) have each
// shipped exactly one frozen layout so far -- "smaller than the oldest
// known layout" and "smaller than sizeof(T)" agree for those two. NOT
// correct for fuse_step_report (see step_report_struct_size_ok below,
// Review 2026-10-11 AB-3): that struct HAS grown, and this template's
// ">= sizeof(T) [current]" bound refuses a plugin built against an
// older header's smaller-but-still-valid struct_size, the exact
// backward-compatibility deviation AB-2 found in the opposite
// direction (participant.cpp's own FirstFrozenStructSize<T>).
template <class T>
[[nodiscard]] bool struct_size_ok(const T* s) noexcept {
  return s != nullptr && s->struct_size >= sizeof(T) &&
         s->struct_size <= static_cast<std::uint32_t>(FUSE_ABI_MAX_STRUCT_SIZE);
}

// Review 2026-10-11 AB-3: the read-side mirror of
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
    return fuse::Failure{"E-ABI-PROVIDER-TABLE",
                         "provider '" + name_ + "': no table (dlopen or version negotiation failed earlier)"};
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
      // Review 2026-10-11 AB-6(a): an ENTRY THAT EXISTS but returns a
      // non-OK status (or a struct_size this host does not recognise) was
      // silently ignored, degrading to "no checkpoint" with no refusal --
      // indistinguishable from the provider correctly reporting it has
      // nothing to checkpoint. A plugin that offers the entry and then
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
                      // A provider's own instance is opaque past the fields it
                      // binds; FUSE's byte-for-byte rollback check can never see
                      // state the plugin keeps that was never described and
                      // bound (component.hpp's own caveat on carries_hidden_state()).
                      fuse::kCapabilityHiddenState;
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

    // Review 2026-10-11 AB-6(b): the cross-check below only walked the
    // graph's OWN fields, asking "does describe() report this one" -- a
    // plugin reporting a SUPERSET (an extra row the graph never declared
    // for this component) passed silently. providers.md's own text
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

      const fuse::Region& region = graph.region(spec.region);
      const fuse::Result<std::size_t> sz = backend_.region_size(region);
      if (!sz.ok()) return fuse::Result<void>{sz.error()};
      const std::size_t count = sz.value();

      storage_[i].assign(count * static_cast<std::size_t>(spec.components), 0.0);
      double* base = storage_[i].data();

      const fuse::Layout layout{.stride = static_cast<std::size_t>(spec.components),
                                .component_stride = 1,
                                .count = count,
                                .components = spec.components};
      fuse::Result<void> bound =
          registry.bind(fid, fuse::FieldRef{base, layout, fuse::MemorySpace::Host});
      if (!bound.ok()) return bound;

      fuse_field_view view{sizeof(fuse_field_view)};
      view.handle            = static_cast<std::int32_t>(fid.value());
      view.element_type      = FUSE_ELEMENT_F64;
      view.memspace          = FUSE_MEMSPACE_HOST;
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

}  // namespace fuse::abi_detail
