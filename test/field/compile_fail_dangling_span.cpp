// MUST NOT COMPILE. Registered with WILL_FAIL, paired with the control below
// it so that "the build failed" cannot pass for the wrong reason.
//
// U6: taking a BORROWED view out of a temporary field is the dangling shape
// the rule exists to prevent, so `span() const&&` is deleted. The allowed
// spelling on a temporary is `slice(...)`, which returns an OWNING value and
// is exercised in pipeline_test.cpp.

#include "fuse/field/pipeline.hpp"

int main() {
  // The field dies at the end of this full-expression; the span would outlive
  // the storage it points at.
  auto dangling = fuse::OwnedField<>("temp", 10, 3).span();
  return static_cast<int>(dangling.element_count());
}
