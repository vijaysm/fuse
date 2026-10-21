#pragma once
/// \file accel_names.hpp
/// One name -> `fuse::AccelKind` table for the examples' `--accel` flag.
///
/// Before this header each example carried its own parser and the three
/// disagreed about what an unrecognised name means: `brusselator` ran with no
/// acceleration, `gray_scott` ran with Aitken, `fsi_channel` refused. A
/// misspelled name therefore silently selected a different method in two of
/// the three programs. Here an unrecognised name is `std::nullopt`, and every
/// caller refuses it by name (`E-EXAMPLE-ACCEL`) and prints `kAccelNames`.
///
/// Scope: this table maps a NAME to a KIND only. Per-example settings
/// (initial relaxation, column history, filter) stay where each example sets
/// them, because those are tuned per problem. `newton` is not in the table:
/// it replaces the whole sweep rather than post-processing one, only the
/// Brusselator example drives it, and that example checks the name itself.
///
/// No alias for Anderson acceleration is provided. `iqn-ils` is the
/// least-squares quasi-Newton method documented in
/// `src/num/accel_quasi_newton.cpp`; whether it coincides with Anderson
/// acceleration (type II) is a claim to be measured against a reference
/// implementation, not assumed here (KNOWN_ISSUES.md D76).

#include <optional>
#include <string_view>

#include "fuse/model/iteration.hpp"

namespace fuse_examples {

/// The names accepted, in the order they should be listed in a usage line.
inline constexpr std::string_view kAccelNames =
    "none|constant|aitken|secant|steffensen|iqn-ils|iqn-imvj";

/// Exact, case-sensitive match. `iqn` is kept as an alias of `iqn-ils`
/// because the examples already accepted it.
[[nodiscard]] constexpr std::optional<fuse::AccelKind> accel_kind_from_name(
    std::string_view s) noexcept {
  if (s == "none") return fuse::AccelKind::None;
  if (s == "constant") return fuse::AccelKind::Constant;
  if (s == "aitken") return fuse::AccelKind::Aitken;
  if (s == "secant") return fuse::AccelKind::Secant;
  if (s == "steffensen") return fuse::AccelKind::Steffensen;
  if (s == "iqn-ils" || s == "iqn") return fuse::AccelKind::IQN_ILS;
  if (s == "iqn-imvj") return fuse::AccelKind::IQN_IMVJ;
  return std::nullopt;
}

}  // namespace fuse_examples
