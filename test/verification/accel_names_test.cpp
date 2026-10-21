// The examples' shared `--accel` name table (examples/common/accel_names.hpp).
//
// Three properties, each with a mutation row in
// build-evidence/mutations/example-accel-names.toml:
//   1. every name maps to exactly the kind it names;
//   2. a name outside the table is REFUSED (std::nullopt), never mapped to a
//      default -- before 2026-10-20 two of the three example programs mapped
//      an unrecognised name to None or to Aitken without a word;
//   3. the printed list `kAccelNames` and the table agree, so a usage line
//      built from the list never offers a name the table refuses nor hides
//      one it accepts.
// No example binary is run here: each example's own refusal (E-EXAMPLE-ACCEL
// before any setup) is exercised by the example.*.accel_refused rows.

#include <gtest/gtest.h>

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "accel_names.hpp"

namespace {

using fuse::AccelKind;
using fuse_examples::accel_kind_from_name;
using fuse_examples::kAccelNames;

std::vector<std::string_view> split_bar(std::string_view s) {
  std::vector<std::string_view> out;
  std::size_t start = 0;
  while (start <= s.size()) {
    const std::size_t bar = s.find('|', start);
    if (bar == std::string_view::npos) {
      out.push_back(s.substr(start));
      break;
    }
    out.push_back(s.substr(start, bar - start));
    start = bar + 1;
  }
  return out;
}

TEST(AccelNames, EveryNameMapsToTheKindItNames) {
  EXPECT_EQ(accel_kind_from_name("none"), AccelKind::None);
  EXPECT_EQ(accel_kind_from_name("constant"), AccelKind::Constant);
  EXPECT_EQ(accel_kind_from_name("aitken"), AccelKind::Aitken);
  EXPECT_EQ(accel_kind_from_name("secant"), AccelKind::Secant);
  EXPECT_EQ(accel_kind_from_name("steffensen"), AccelKind::Steffensen);
  EXPECT_EQ(accel_kind_from_name("iqn-ils"), AccelKind::IQN_ILS);
  EXPECT_EQ(accel_kind_from_name("iqn-imvj"), AccelKind::IQN_IMVJ);
}

TEST(AccelNames, IqnIsAnAliasOfIqnIlsAndOfNothingElse) {
  EXPECT_EQ(accel_kind_from_name("iqn"), AccelKind::IQN_ILS);
  EXPECT_NE(accel_kind_from_name("iqn"), accel_kind_from_name("iqn-imvj"));
}

TEST(AccelNames, AnUnknownNameIsRefusedNotMappedToADefault) {
  // The two defaults the old parsers fell back to are None and Aitken; an
  // unknown name must be neither, it must be absent.
  for (const char* bad : {"", " ", "picard", "Aitken", "AITKEN", "aitken ", " aitken",
                          "iqn_ils", "iqn-ils2", "anderson", "wynn", "epsilon",
                          "newton", "iqn-", "-aitken", "aitken|none"}) {
    const std::optional<AccelKind> k = accel_kind_from_name(bad);
    EXPECT_FALSE(k.has_value()) << "'" << bad << "' must be refused";
  }
}

TEST(AccelNames, NewtonIsNotInTheSharedTable) {
  // Newton replaces the whole sweep and only the Brusselator example drives
  // it; that example checks the name itself (examples/brusselator/main.cpp).
  EXPECT_FALSE(accel_kind_from_name("newton").has_value());
}

TEST(AccelNames, ThePrintedListAndTheTableAgree) {
  const std::vector<std::string_view> names = split_bar(kAccelNames);
  ASSERT_EQ(names.size(), 7U);
  for (const std::string_view n : names) {
    EXPECT_TRUE(accel_kind_from_name(n).has_value())
        << "'" << std::string(n) << "' is printed in the usage list but refused";
  }
  // Distinct names print distinct kinds: the list has no duplicate entry and
  // no two printed names select one method.
  for (std::size_t i = 0; i < names.size(); ++i) {
    for (std::size_t j = i + 1; j < names.size(); ++j) {
      EXPECT_NE(accel_kind_from_name(names[i]), accel_kind_from_name(names[j]))
          << std::string(names[i]) << " and " << std::string(names[j]);
    }
  }
}

}  // namespace
