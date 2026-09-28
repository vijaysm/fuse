// MUST NOT COMPILE. Registered with WILL_FAIL, paired with the control below
// it so that "the build failed" cannot pass for the wrong reason.
//
// KNOWN_ISSUES.md B3: `Component::capabilities()` is now the one virtual;
// `supports_rollback()` is a non-virtual reader of it, declared with no
// `virtual` keyword at all. A derived class that still writes
// `bool supports_rollback() const override` -- the pre-B3 spelling -- names
// no virtual function to override, so `override` on it is ill-formed
// (a C++20 requirement, not merely a warning): this line, and nothing else,
// is what must fail to compile.

#include "fuse/runtime/component.hpp"

namespace {
class StaleOverride final : public fuse::Component {
 public:
  [[nodiscard]] std::string name() const override { return "stale"; }
  fuse::Result<void> register_fields(fuse::FieldRegistry&,
                                     const fuse::CouplingGraph&) override {
    return fuse::Result<void>::success();
  }
  fuse::Result<void> advance(const fuse::StepContext&) override {
    return fuse::Result<void>::success();
  }
  [[nodiscard]] bool supports_rollback() const override { return true; }
};
}  // namespace

int main() {
  StaleOverride s;
  return s.supports_rollback() ? 0 : 1;
}
