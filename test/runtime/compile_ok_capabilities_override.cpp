// The CONTROL for compile_fail_stale_rollback_override.cpp: the same shape,
// written the way KNOWN_ISSUES.md B3 permits -- override `capabilities()`,
// not the retired `bool supports_rollback()`. Must compile, which is what
// proves the other file's failure is the missing virtual and not a broken
// include or an unrelated typo.

#include "fuse/runtime/component.hpp"

namespace {
class CurrentOverride final : public fuse::Component {
 public:
  [[nodiscard]] std::string name() const override { return "current"; }
  fuse::Result<void> register_fields(fuse::FieldRegistry&,
                                     const fuse::CouplingGraph&) override {
    return fuse::Result<void>::success();
  }
  fuse::Result<void> advance(const fuse::StepContext&) override {
    return fuse::Result<void>::success();
  }
  [[nodiscard]] std::uint64_t capabilities() const override {
    return fuse::kCapabilityRollback;
  }
};
}  // namespace

int main() {
  CurrentOverride c;
  return c.supports_rollback() ? 0 : 1;
}
