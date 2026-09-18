// The CONTROL for compile_fail_dangling_span.cpp: the same shape, written the
// way U6 permits. It must compile, which is what proves the other file's
// failure is the deleted overload and not a broken include or a typo.

#include "fuse/field/pipeline.hpp"

int main() {
  // Chaining on a temporary is allowed: the result OWNS, sharing the managed
  // view's reference count, so nothing dangles.
  auto owning = fuse::OwnedField<>("temp", 10, 3).slice({2, 5}, fuse::all);
  return static_cast<int>(owning.element_count()) == 3 ? 0 : 1;
}
