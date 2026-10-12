// Ported from upstream/soma/test/unit/remap_test.cpp's RemapApply suite
// (docs/dev/soma-import.md, case-level table): `fuse::apply_weights()` has
// carried the E-REMAP-ARG/E-REMAP-COO/E-REMAP-SIZE contract documented on its
// own declaration (include/fuse/backend/remap_plan.hpp) since the soma import,
// but no test anywhere calls it with a malformed RemapWeights or a null
// buffer -- every call site in test/{backend,parallel,verification} passes a
// well-formed triplet set and asserts success. This file closes that gap: the
// function's OWN error contract, exercised directly, not through a higher
// wrapper (fuse::WeightsOperator has its own, different E-OP-SHAPE contract;
// see test/backend/operator_conformance_test.cpp).

#include "fuse/backend/remap_plan.hpp"

#include <vector>

#include <gtest/gtest.h>

using fuse::RemapWeights;
using fuse::apply_weights;

namespace {

/// A trivial 2x2 identity map: row[k] == col[k] == k, val == 1.
RemapWeights identity_2x2() {
  RemapWeights w;
  w.n_rows = 2;
  w.n_cols = 2;
  w.row = {0, 1};
  w.col = {0, 1};
  w.val = {1.0, 1.0};
  return w;
}

}  // namespace

TEST(RemapApply, RejectsBadArguments) {
  const RemapWeights w = identity_2x2();
  std::vector<double> a(2, 1.0), b(2, 0.0);

  EXPECT_EQ(apply_weights(w, a.data(), b.data(), 0).error().code, "E-REMAP-ARG")
      << "components == 0 must be refused, not read as \"nothing to do\"";
  EXPECT_EQ(apply_weights(w, a.data(), b.data(), -1).error().code, "E-REMAP-ARG")
      << "a negative component count must be refused before it reaches the "
         "unsigned arithmetic below it";
  EXPECT_EQ(apply_weights(w, nullptr, b.data(), 1).error().code, "E-REMAP-ARG")
      << "a null source buffer must be refused, not dereferenced";
  EXPECT_EQ(apply_weights(w, a.data(), nullptr, 1).error().code, "E-REMAP-ARG")
      << "a null destination buffer must be refused, not dereferenced";

  // The identical, well-formed call must still succeed: the checks above
  // reject exactly the bad arguments, not the matrix itself.
  EXPECT_TRUE(apply_weights(w, a.data(), b.data(), 1).ok());
}

TEST(RemapApply, RejectsOutOfRangeTriplets) {
  RemapWeights w = identity_2x2();
  w.row = {0, 5};  // 5 is outside [0, n_rows) == [0, 2)
  std::vector<double> a(2, 1.0), b(2, 0.0);

  const auto out_of_range = apply_weights(w, a.data(), b.data(), 1);
  ASSERT_FALSE(out_of_range.ok());
  EXPECT_EQ(out_of_range.error().code, "E-REMAP-COO");

  // A ragged triplet (row/col/val of different lengths) is the same error
  // code: both are "this is not a valid sparse matrix", not two different
  // failure modes a caller would need to distinguish.
  RemapWeights ragged = identity_2x2();
  ragged.row = {0};  // col/val still have 2 entries
  const auto ragged_result = apply_weights(ragged, a.data(), b.data(), 1);
  ASSERT_FALSE(ragged_result.ok());
  EXPECT_EQ(ragged_result.error().code, "E-REMAP-COO");
}
