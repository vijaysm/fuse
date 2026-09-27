# SOMA Phase 0 — Skeleton and vocabulary types

**Parent plan:** `2026-09-02-soma-implementation.md` (read its Global Constraints first)
**Spec sections:** §3.1 (tier A), §4.3 (time levels), §9 (build)
**Milestone:** `cmake -B build && cmake --build build && ctest --test-dir build` succeeds with zero third-party dependencies.

Phase 0 builds tier A: header-only vocabulary with no dependencies. Everything here is `constexpr`-friendly, trivially testable, and used by every later phase.

---

## File structure

| File | Responsibility |
|---|---|
| `CMakeLists.txt` | top-level build, options, dependency discovery |
| `cmake/SomaOptions.cmake` | option declarations and their defaults |
| `test/CMakeLists.txt` | GoogleTest discovery and test registration |
| `include/soma/base/result.hpp` | `Result<T>`, `Error`, `Diagnostics` |
| `include/soma/base/rational.hpp` | exact rational arithmetic for time |
| `include/soma/base/units.hpp` | dimension exponent vector |
| `include/soma/base/time_level.hpp` | `Anchor`, `TimeLevel` |
| `include/soma/base/ids.hpp` | strong-typed IDs and core enums |
| `test/unit/*_test.cpp` | one test file per header |

---

### Task 0.1: Repository, CMake skeleton, and a passing smoke test

**Files:**
- Create: `modules/coupler/CMakeLists.txt`
- Create: `modules/coupler/cmake/SomaOptions.cmake`
- Create: `modules/coupler/include/soma/soma.hpp`
- Create: `modules/coupler/test/CMakeLists.txt`
- Create: `modules/coupler/test/unit/smoke_test.cpp`
- Create: `modules/coupler/.gitignore`

**Interfaces:**
- Consumes: nothing.
- Produces: CMake targets `soma` (INTERFACE for now) and `soma_tests`; the `SOMA_ENABLE_*` options; `soma::version()` returning `const char*`.

- [ ] **Step 1: Initialise the repository**

```bash
cd /Users/mahadevan/Code/AI/coupling/coupe/modules/coupler
git init
printf 'build*/\n.DS_Store\ncompile_commands.json\n' > .gitignore
git add .gitignore && git commit -m "chore: initialise soma repository"
```

- [ ] **Step 2: Write the options module**

`cmake/SomaOptions.cmake`:

```cmake
option(SOMA_ENABLE_TESTS    "Build unit tests"                ON)
option(SOMA_ENABLE_MPI      "Enable MPI transport"            OFF)
option(SOMA_ENABLE_MOAB     "Enable MOAB mesh backend"        OFF)
option(SOMA_ENABLE_PETSC    "Enable PETSc solvers"            OFF)
option(SOMA_ENABLE_KOKKOS   "Enable Kokkos execution spaces"  OFF)
option(SOMA_ENABLE_LOWFIVE  "Enable LowFive transport"        OFF)
option(SOMA_ENABLE_PYTHON   "Build Python bindings"           OFF)
option(SOMA_ENABLE_FORTRAN  "Build Fortran shim"              OFF)

function(soma_print_summary)
  message(STATUS "SOMA configuration:")
  foreach(opt TESTS MPI MOAB PETSC KOKKOS LOWFIVE PYTHON FORTRAN)
    message(STATUS "  SOMA_ENABLE_${opt}: ${SOMA_ENABLE_${opt}}")
  endforeach()
endfunction()
```

- [ ] **Step 3: Write the top-level CMakeLists**

```cmake
cmake_minimum_required(VERSION 3.20)
project(SOMA VERSION 0.1.0 LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(SomaOptions)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# Tier A/B/C are header-only, so the base target is INTERFACE.
# Phase 4 adds a compiled `soma_core` target alongside it.
add_library(soma INTERFACE)
target_include_directories(soma INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_features(soma INTERFACE cxx_std_20)

if(SOMA_ENABLE_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()

soma_print_summary()
```

- [ ] **Step 4: Write the umbrella header**

`include/soma/soma.hpp` — nothing internal may include this; it exists for user convenience only.

```cpp
#pragma once

namespace soma {

/// Library version as "MAJOR.MINOR.PATCH".
constexpr const char* version() noexcept { return "0.1.0"; }

}  // namespace soma
```

- [ ] **Step 5: Write the test CMakeLists**

```cmake
include(FetchContent)
FetchContent_Declare(googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.17.0
  FIND_PACKAGE_ARGS NAMES GTest)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

function(soma_add_test name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE soma GTest::gtest_main)
  target_compile_options(${name} PRIVATE -Wall -Wextra -Wpedantic)
  gtest_discover_tests(${name})
endfunction()

soma_add_test(smoke_test unit/smoke_test.cpp)
```

`FIND_PACKAGE_ARGS` makes CMake prefer the installed GoogleTest 1.17.0 at `/opt/homebrew/opt/googletest` and fall back to fetching.

- [ ] **Step 6: Write the failing test**

`test/unit/smoke_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include <soma/soma.hpp>

#include <string_view>

TEST(Smoke, VersionIsNonEmpty) {
  EXPECT_FALSE(std::string_view{soma::version()}.empty());
}
```

- [ ] **Step 7: Configure, build, and run**

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Expected: configure prints the option summary; `1 test from 1 test suite ran. [ PASSED ] 1 test.`

- [ ] **Step 8: Commit**

```bash
git add CMakeLists.txt cmake include test
git commit -m "feat(soma): CMake skeleton with header-only core and gtest"
```

---

### Task 0.2: `Result<T>`, `Error`, and `Diagnostics`

Per Global Constraints, validation accumulates diagnostics rather than aborting on the first problem. This task builds that machinery because Phase 1's validator depends on it.

**Files:**
- Create: `include/soma/base/result.hpp`
- Test: `test/unit/result_test.cpp`
- Modify: `test/CMakeLists.txt` (add `soma_add_test(result_test unit/result_test.cpp)`)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `enum class Severity { Warning, Error }`
  - `struct Diagnostic { Severity severity; std::string code; std::string message; std::string context; }`
  - `class Diagnostics` with `void add(Severity, std::string code, std::string message, std::string context = {})`, `bool has_errors() const`, `std::size_t size() const`, `const std::vector<Diagnostic>& items() const`, `std::string to_string() const`, `void merge(Diagnostics)`
  - `struct Error { std::string code; std::string message; }`
  - `template <class T> class Result` with `Result(T)`, `Result(Error)`, `bool ok() const`, `explicit operator bool() const`, `T& value()`, `const T& value() const`, `const Error& error() const`
  - `Result<void>` specialisation with `static Result<void> success()`

- [ ] **Step 1: Write the failing test**

`test/unit/result_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include <soma/base/result.hpp>

#include <string>

using soma::Diagnostics;
using soma::Error;
using soma::Result;
using soma::Severity;

TEST(Result, HoldsValue) {
  Result<int> r{42};
  ASSERT_TRUE(r.ok());
  EXPECT_TRUE(static_cast<bool>(r));
  EXPECT_EQ(r.value(), 42);
}

TEST(Result, HoldsError) {
  Result<int> r{Error{"E001", "bad thing"}};
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E001");
  EXPECT_EQ(r.error().message, "bad thing");
}

TEST(Result, VoidSuccess) {
  auto r = Result<void>::success();
  EXPECT_TRUE(r.ok());
}

TEST(Diagnostics, AccumulatesAndReportsErrors) {
  Diagnostics d;
  EXPECT_FALSE(d.has_errors());
  d.add(Severity::Warning, "W001", "just a warning", "channel[0]");
  EXPECT_FALSE(d.has_errors());
  d.add(Severity::Error, "E002", "real problem", "channel[1]");
  EXPECT_TRUE(d.has_errors());
  EXPECT_EQ(d.size(), 2u);
  EXPECT_EQ(d.items()[1].context, "channel[1]");
}

TEST(Diagnostics, MergePreservesOrder) {
  Diagnostics a, b;
  a.add(Severity::Error, "E1", "first", "");
  b.add(Severity::Error, "E2", "second", "");
  a.merge(std::move(b));
  ASSERT_EQ(a.size(), 2u);
  EXPECT_EQ(a.items()[0].code, "E1");
  EXPECT_EQ(a.items()[1].code, "E2");
}

TEST(Diagnostics, ToStringMentionsEveryCode) {
  Diagnostics d;
  d.add(Severity::Error, "E42", "boom", "here");
  const std::string s = d.to_string();
  EXPECT_NE(s.find("E42"), std::string::npos);
  EXPECT_NE(s.find("boom"), std::string::npos);
  EXPECT_NE(s.find("here"), std::string::npos);
}
```

- [ ] **Step 2: Run to verify it fails**

```bash
cmake --build build -j 2>&1 | tail -5
```

Expected: FAIL — `soma/base/result.hpp` file not found.

- [ ] **Step 3: Write the implementation**

`include/soma/base/result.hpp`:

```cpp
#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace soma {

enum class Severity { Warning, Error };

struct Diagnostic {
  Severity    severity;
  std::string code;
  std::string message;
  std::string context;
};

/// Collects validation problems so they can be reported together.
/// Validation never aborts on the first error; see Global Constraints.
class Diagnostics {
 public:
  void add(Severity severity, std::string code, std::string message,
           std::string context = {}) {
    items_.push_back(Diagnostic{severity, std::move(code), std::move(message),
                                std::move(context)});
  }

  [[nodiscard]] bool has_errors() const noexcept {
    for (const auto& d : items_) {
      if (d.severity == Severity::Error) return true;
    }
    return false;
  }

  [[nodiscard]] std::size_t size() const noexcept { return items_.size(); }
  [[nodiscard]] bool empty() const noexcept { return items_.empty(); }
  [[nodiscard]] const std::vector<Diagnostic>& items() const noexcept {
    return items_;
  }

  void merge(Diagnostics other) {
    items_.insert(items_.end(), std::make_move_iterator(other.items_.begin()),
                  std::make_move_iterator(other.items_.end()));
  }

  [[nodiscard]] std::string to_string() const {
    std::string out;
    for (const auto& d : items_) {
      out += (d.severity == Severity::Error) ? "error " : "warning ";
      out += d.code;
      out += ": ";
      out += d.message;
      if (!d.context.empty()) {
        out += " [";
        out += d.context;
        out += "]";
      }
      out += '\n';
    }
    return out;
  }

 private:
  std::vector<Diagnostic> items_;
};

struct Error {
  std::string code;
  std::string message;
};

/// Recoverable-error return type. Programmer errors use SOMA_ASSERT instead.
template <class T>
class Result {
 public:
  Result(T value) : storage_(std::move(value)) {}       // NOLINT: implicit
  Result(Error error) : storage_(std::move(error)) {}   // NOLINT: implicit

  [[nodiscard]] bool ok() const noexcept {
    return std::holds_alternative<T>(storage_);
  }
  explicit operator bool() const noexcept { return ok(); }

  [[nodiscard]] T&       value()       { return std::get<T>(storage_); }
  [[nodiscard]] const T& value() const { return std::get<T>(storage_); }
  [[nodiscard]] const Error& error() const { return std::get<Error>(storage_); }

 private:
  std::variant<T, Error> storage_;
};

template <>
class Result<void> {
 public:
  Result(Error error) : error_(std::move(error)), ok_(false) {}  // NOLINT
  static Result<void> success() { return Result<void>{}; }

  [[nodiscard]] bool ok() const noexcept { return ok_; }
  explicit operator bool() const noexcept { return ok_; }
  [[nodiscard]] const Error& error() const noexcept { return error_; }

 private:
  Result() = default;
  Error error_{};
  bool  ok_ = true;
};

}  // namespace soma
```

- [ ] **Step 4: Register and run the test**

Add to `test/CMakeLists.txt`: `soma_add_test(result_test unit/result_test.cpp)`

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug && cmake --build build -j
ctest --test-dir build --output-on-failure -R Result
ctest --test-dir build --output-on-failure -R Diagnostics
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add include/soma/base/result.hpp test/unit/result_test.cpp test/CMakeLists.txt
git commit -m "feat(soma): Result, Error, and accumulating Diagnostics"
```

---

### Task 0.3: `Rational` — exact time arithmetic

Spec §4.3 and §8.2 make this load-bearing: component intervals of 1/3 and 1/7 must produce a drift-free slab decomposition over 10⁶ steps. Floating-point time would fail silently.

**Files:**
- Create: `include/soma/base/rational.hpp`
- Test: `test/unit/rational_test.cpp`
- Modify: `test/CMakeLists.txt`

**Interfaces:**
- Consumes: nothing.
- Produces: `class Rational` with
  - `constexpr Rational()` (zero), `constexpr Rational(std::int64_t n)`, `constexpr Rational(std::int64_t num, std::int64_t den)` (normalising, sign on numerator, throws-free; `den == 0` is a programmer error)
  - `constexpr std::int64_t num() const`, `constexpr std::int64_t den() const`
  - operators `+ - * /`, compound forms, `==`, `<=>`
  - `constexpr bool is_zero() const`, `constexpr bool is_integer() const`
  - `constexpr double to_double() const`
  - `static constexpr Rational lcm_denominator(Rational a, Rational b)` returning `Rational{1, lcm(a.den(), b.den())}` — the finest common tick of two intervals
  - `std::string to_string() const` → `"3/7"` or `"2"`
  - free `constexpr Rational gcd_rational(Rational, Rational)` — the largest `g` such that both arguments are integer multiples of `g`; this is what §8.3 uses to build the slab grid

- [ ] **Step 1: Write the failing test**

`test/unit/rational_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include <soma/base/rational.hpp>

using soma::gcd_rational;
using soma::Rational;

TEST(Rational, NormalisesOnConstruction) {
  constexpr Rational r{6, 8};
  static_assert(r.num() == 3);
  static_assert(r.den() == 4);
  EXPECT_EQ(r.to_string(), "3/4");
}

TEST(Rational, PutsSignOnNumerator) {
  constexpr Rational r{1, -2};
  static_assert(r.num() == -1);
  static_assert(r.den() == 2);
}

TEST(Rational, IntegerPrintsWithoutDenominator) {
  EXPECT_EQ(Rational{5}.to_string(), "5");
  EXPECT_TRUE(Rational(4, 2).is_integer());
}

TEST(Rational, Arithmetic) {
  constexpr Rational a{1, 3};
  constexpr Rational b{1, 7};
  static_assert((a + b) == Rational(10, 21));
  static_assert((a - b) == Rational(4, 21));
  static_assert((a * b) == Rational(1, 21));
  static_assert((a / b) == Rational(7, 3));
}

TEST(Rational, Ordering) {
  EXPECT_LT(Rational(1, 3), Rational(1, 2));
  EXPECT_EQ(Rational(2, 4), Rational(1, 2));
  EXPECT_GT(Rational(3, 4), Rational(2, 3));
}

// The property that matters for §8.2: no drift over long runs.
TEST(Rational, NoDriftOverManySteps) {
  const Rational dt{1, 3};
  Rational t{0};
  for (int i = 0; i < 1'000'000; ++i) t += dt;
  EXPECT_EQ(t, Rational(1'000'000, 3));
  EXPECT_EQ(t.den(), 3);  // still exact, not renormalised to something huge
}

TEST(Rational, LcmDenominatorGivesFinestCommonTick) {
  EXPECT_EQ(Rational::lcm_denominator(Rational(1, 3), Rational(1, 7)),
            Rational(1, 21));
  EXPECT_EQ(Rational::lcm_denominator(Rational(1, 4), Rational(1, 2)),
            Rational(1, 4));
}

// gcd_rational(a,b) is the largest g with a = i*g and b = j*g for integers i,j.
TEST(Rational, GcdRationalIsCommonDivisor) {
  EXPECT_EQ(gcd_rational(Rational(1, 3), Rational(1, 7)), Rational(1, 21));
  EXPECT_EQ(gcd_rational(Rational(2, 3), Rational(4, 3)), Rational(2, 3));
  EXPECT_EQ(gcd_rational(Rational(3), Rational(2)), Rational(1));
}

TEST(Rational, ToDouble) {
  EXPECT_DOUBLE_EQ(Rational(1, 4).to_double(), 0.25);
}
```

- [ ] **Step 2: Run to verify it fails**

```bash
cmake --build build -j 2>&1 | tail -5
```

Expected: FAIL — `soma/base/rational.hpp` file not found.

- [ ] **Step 3: Write the implementation**

`include/soma/base/rational.hpp`:

```cpp
#pragma once

#include <cassert>
#include <compare>
#include <cstdint>
#include <numeric>
#include <string>

namespace soma {

/// Exact rational time. Spec §4.3: floating-point time accumulation would
/// destroy slab decomposition over long runs, so every time quantity in SOMA
/// is a Rational.
class Rational {
 public:
  constexpr Rational() = default;
  constexpr Rational(std::int64_t n) : num_(n), den_(1) {}  // NOLINT: implicit

  constexpr Rational(std::int64_t n, std::int64_t d) : num_(n), den_(d) {
    assert(d != 0 && "Rational denominator must be non-zero");
    normalise();
  }

  [[nodiscard]] constexpr std::int64_t num() const noexcept { return num_; }
  [[nodiscard]] constexpr std::int64_t den() const noexcept { return den_; }

  [[nodiscard]] constexpr bool is_zero() const noexcept { return num_ == 0; }
  [[nodiscard]] constexpr bool is_integer() const noexcept { return den_ == 1; }
  [[nodiscard]] constexpr double to_double() const noexcept {
    return static_cast<double>(num_) / static_cast<double>(den_);
  }

  constexpr Rational& operator+=(const Rational& o) {
    num_ = num_ * o.den_ + o.num_ * den_;
    den_ *= o.den_;
    normalise();
    return *this;
  }
  constexpr Rational& operator-=(const Rational& o) {
    num_ = num_ * o.den_ - o.num_ * den_;
    den_ *= o.den_;
    normalise();
    return *this;
  }
  constexpr Rational& operator*=(const Rational& o) {
    num_ *= o.num_;
    den_ *= o.den_;
    normalise();
    return *this;
  }
  constexpr Rational& operator/=(const Rational& o) {
    assert(o.num_ != 0 && "division by zero Rational");
    num_ *= o.den_;
    den_ *= o.num_;
    normalise();
    return *this;
  }

  friend constexpr Rational operator+(Rational a, const Rational& b) { return a += b; }
  friend constexpr Rational operator-(Rational a, const Rational& b) { return a -= b; }
  friend constexpr Rational operator*(Rational a, const Rational& b) { return a *= b; }
  friend constexpr Rational operator/(Rational a, const Rational& b) { return a /= b; }

  friend constexpr bool operator==(const Rational& a, const Rational& b) noexcept {
    return a.num_ == b.num_ && a.den_ == b.den_;  // always normalised
  }
  friend constexpr std::strong_ordering operator<=>(const Rational& a,
                                                    const Rational& b) noexcept {
    return a.num_ * b.den_ <=> b.num_ * a.den_;
  }

  /// Finest tick that divides both denominators: 1/lcm(den(a), den(b)).
  [[nodiscard]] static constexpr Rational lcm_denominator(const Rational& a,
                                                          const Rational& b) {
    return Rational{1, std::lcm(a.den_, b.den_)};
  }

  [[nodiscard]] std::string to_string() const {
    if (den_ == 1) return std::to_string(num_);
    return std::to_string(num_) + "/" + std::to_string(den_);
  }

 private:
  constexpr void normalise() {
    if (den_ < 0) {
      num_ = -num_;
      den_ = -den_;
    }
    const std::int64_t g = std::gcd(num_ < 0 ? -num_ : num_, den_);
    if (g > 1) {
      num_ /= g;
      den_ /= g;
    }
    if (num_ == 0) den_ = 1;
  }

  std::int64_t num_ = 0;
  std::int64_t den_ = 1;
};

/// Largest g such that a and b are both integer multiples of g.
/// gcd(a/b, c/d) = gcd(a,c) / lcm(b,d). Used by §8.3 slab decomposition.
[[nodiscard]] constexpr Rational gcd_rational(const Rational& a,
                                              const Rational& b) {
  const std::int64_t an = a.num() < 0 ? -a.num() : a.num();
  const std::int64_t bn = b.num() < 0 ? -b.num() : b.num();
  return Rational{std::gcd(an, bn), std::lcm(a.den(), b.den())};
}

}  // namespace soma
```

- [ ] **Step 4: Register and run**

Add `soma_add_test(rational_test unit/rational_test.cpp)` to `test/CMakeLists.txt`.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug && cmake --build build -j
ctest --test-dir build --output-on-failure -R Rational
```

Expected: all 9 tests PASS. `NoDriftOverManySteps` is the one that matters — if `den()` is not 3 at the end, normalisation is broken.

- [ ] **Step 5: Commit**

```bash
git add include/soma/base/rational.hpp test/unit/rational_test.cpp test/CMakeLists.txt
git commit -m "feat(soma): exact Rational time arithmetic"
```

---

### Task 0.4: `Units` — dimension exponent vector

Spec §4.2: units are seven `constexpr`-comparable integer exponents, not strings, so connecting W/m² to K is a lowering-time error.

**Files:**
- Create: `include/soma/base/units.hpp`
- Test: `test/unit/units_test.cpp`
- Modify: `test/CMakeLists.txt`

**Interfaces:**
- Consumes: nothing.
- Produces: `struct Units` with
  - members `std::int8_t length, mass, time, current, temperature, amount, luminosity` (SI base order)
  - `constexpr bool operator==` (defaulted)
  - `constexpr Units operator*(Units) const`, `constexpr Units operator/(Units) const` (exponent add/subtract)
  - `constexpr bool is_dimensionless() const`
  - `std::string to_string() const` → `"kg·s^-3"` style, `"1"` when dimensionless
  - namespace `soma::units` with `constexpr Units kDimensionless, kMetre, kKilogram, kSecond, kKelvin, kWatt, kJoule, kNewton, kPascal, kHeatFluxDensity` (W/m²), `kTemperature` (alias of `kKelvin`)

- [ ] **Step 1: Write the failing test**

`test/unit/units_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include <soma/base/units.hpp>

using soma::Units;
namespace u = soma::units;

TEST(Units, DimensionlessIsIdentity) {
  static_assert(u::kDimensionless.is_dimensionless());
  static_assert(u::kMetre * u::kDimensionless == u::kMetre);
}

TEST(Units, MultiplicationAddsExponents) {
  static_assert(u::kMetre * u::kMetre == Units{.length = 2});
}

TEST(Units, DivisionSubtractsExponents) {
  static_assert(u::kMetre / u::kMetre == u::kDimensionless);
  static_assert(u::kJoule / u::kSecond == u::kWatt);
}

TEST(Units, HeatFluxDensityIsWattPerSquareMetre) {
  static_assert(u::kHeatFluxDensity == u::kWatt / (u::kMetre * u::kMetre));
}

// The check that makes §4.2 worth having.
TEST(Units, TemperatureIsNotHeatFlux) {
  static_assert(!(u::kKelvin == u::kHeatFluxDensity));
}

TEST(Units, ToStringRoundTripsFamiliarNames) {
  EXPECT_EQ(u::kDimensionless.to_string(), "1");
  EXPECT_EQ(u::kMetre.to_string(), "m");
  EXPECT_EQ(u::kKelvin.to_string(), "K");
  // W = kg·m^2·s^-3
  const std::string w = u::kWatt.to_string();
  EXPECT_NE(w.find("kg"), std::string::npos);
  EXPECT_NE(w.find("m^2"), std::string::npos);
  EXPECT_NE(w.find("s^-3"), std::string::npos);
}
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — `soma/base/units.hpp` file not found.

- [ ] **Step 3: Write the implementation**

`include/soma/base/units.hpp`:

```cpp
#pragma once

#include <cstdint>
#include <string>

namespace soma {

/// SI dimension exponents. Spec §4.2: exponent vectors compose and compare;
/// unit strings do not.
struct Units {
  std::int8_t length = 0;       // m
  std::int8_t mass = 0;         // kg
  std::int8_t time = 0;         // s
  std::int8_t current = 0;      // A
  std::int8_t temperature = 0;  // K
  std::int8_t amount = 0;       // mol
  std::int8_t luminosity = 0;   // cd

  friend constexpr bool operator==(const Units&, const Units&) = default;

  [[nodiscard]] constexpr Units operator*(const Units& o) const noexcept {
    return Units{static_cast<std::int8_t>(length + o.length),
                 static_cast<std::int8_t>(mass + o.mass),
                 static_cast<std::int8_t>(time + o.time),
                 static_cast<std::int8_t>(current + o.current),
                 static_cast<std::int8_t>(temperature + o.temperature),
                 static_cast<std::int8_t>(amount + o.amount),
                 static_cast<std::int8_t>(luminosity + o.luminosity)};
  }

  [[nodiscard]] constexpr Units operator/(const Units& o) const noexcept {
    return Units{static_cast<std::int8_t>(length - o.length),
                 static_cast<std::int8_t>(mass - o.mass),
                 static_cast<std::int8_t>(time - o.time),
                 static_cast<std::int8_t>(current - o.current),
                 static_cast<std::int8_t>(temperature - o.temperature),
                 static_cast<std::int8_t>(amount - o.amount),
                 static_cast<std::int8_t>(luminosity - o.luminosity)};
  }

  [[nodiscard]] constexpr bool is_dimensionless() const noexcept {
    return *this == Units{};
  }

  [[nodiscard]] std::string to_string() const {
    struct Entry { std::int8_t exp; const char* sym; };
    const Entry entries[] = {{mass, "kg"},        {length, "m"},
                             {time, "s"},         {current, "A"},
                             {temperature, "K"},  {amount, "mol"},
                             {luminosity, "cd"}};
    std::string out;
    for (const auto& e : entries) {
      if (e.exp == 0) continue;
      if (!out.empty()) out += "·";  // middle dot
      out += e.sym;
      if (e.exp != 1) {
        out += "^";
        out += std::to_string(static_cast<int>(e.exp));
      }
    }
    return out.empty() ? std::string{"1"} : out;
  }
};

namespace units {

inline constexpr Units kDimensionless{};
inline constexpr Units kMetre{.length = 1};
inline constexpr Units kKilogram{.mass = 1};
inline constexpr Units kSecond{.time = 1};
inline constexpr Units kKelvin{.temperature = 1};
inline constexpr Units kTemperature = kKelvin;

inline constexpr Units kNewton{.length = 1, .mass = 1, .time = -2};
inline constexpr Units kJoule{.length = 2, .mass = 1, .time = -2};
inline constexpr Units kWatt{.length = 2, .mass = 1, .time = -3};
inline constexpr Units kPascal{.length = -1, .mass = 1, .time = -2};
inline constexpr Units kHeatFluxDensity{.mass = 1, .time = -3};  // W/m^2

}  // namespace units
}  // namespace soma
```

- [ ] **Step 4: Register and run**

Add `soma_add_test(units_test unit/units_test.cpp)`.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug && cmake --build build -j
ctest --test-dir build --output-on-failure -R Units
```

Expected: 6 tests PASS. Note the `static_assert`s mean most failures surface at compile time.

- [ ] **Step 5: Commit**

```bash
git add include/soma/base/units.hpp test/unit/units_test.cpp test/CMakeLists.txt
git commit -m "feat(soma): SI dimension exponent Units"
```

---

### Task 0.5: `TimeLevel` and `Anchor`

Spec §4.3. This is the type that makes staggering checkable; Phase 2 Task 2.3 enforces it on DAG arcs.

**Files:**
- Create: `include/soma/base/time_level.hpp`
- Test: `test/unit/time_level_test.cpp`
- Modify: `test/CMakeLists.txt`

**Interfaces:**
- Consumes: `soma::Rational` (Task 0.3).
- Produces:
  - `enum class Anchor { WindowStart, WindowEnd, Absolute }`
  - `struct TimeLevel { Anchor anchor = Anchor::WindowEnd; Rational offset{0}; int iterate = 0; }` with defaulted `operator==`, a `std::string to_string() const` → `"[WindowEnd+1/2, k-1]"`, and `static TimeLevel current()` / `static TimeLevel lagged(int steps)` (offset `-steps`, `Anchor::WindowEnd`) / `static TimeLevel previous_iterate()` (iterate `-1`)
  - `bool compatible(const TimeLevel& produced, const TimeLevel& required)` — exact equality; the escape hatch lives on the channel, not here

- [ ] **Step 1: Write the failing test**

`test/unit/time_level_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include <soma/base/time_level.hpp>

using soma::Anchor;
using soma::compatible;
using soma::Rational;
using soma::TimeLevel;

TEST(TimeLevel, DefaultsToCurrentWindowEnd) {
  const TimeLevel t;
  EXPECT_EQ(t.anchor, Anchor::WindowEnd);
  EXPECT_TRUE(t.offset.is_zero());
  EXPECT_EQ(t.iterate, 0);
}

TEST(TimeLevel, LaggedIsNegativeOffset) {
  const TimeLevel t = TimeLevel::lagged(1);
  EXPECT_EQ(t.offset, Rational(-1));
  EXPECT_EQ(t.anchor, Anchor::WindowEnd);
}

TEST(TimeLevel, PreviousIterate) {
  EXPECT_EQ(TimeLevel::previous_iterate().iterate, -1);
}

TEST(TimeLevel, CompatibilityIsExactEquality) {
  const TimeLevel a{Anchor::WindowEnd, Rational{0}, 0};
  const TimeLevel b{Anchor::WindowEnd, Rational{0}, 0};
  const TimeLevel c{Anchor::WindowStart, Rational{0}, 0};
  const TimeLevel d{Anchor::WindowEnd, Rational{1, 2}, 0};
  EXPECT_TRUE(compatible(a, b));
  EXPECT_FALSE(compatible(a, c));
  EXPECT_FALSE(compatible(a, d));
}

TEST(TimeLevel, ToStringShowsAnchorOffsetAndIterate) {
  const TimeLevel t{Anchor::WindowEnd, Rational{1, 2}, -1};
  const std::string s = t.to_string();
  EXPECT_NE(s.find("WindowEnd"), std::string::npos);
  EXPECT_NE(s.find("1/2"), std::string::npos);
  EXPECT_NE(s.find("k-1"), std::string::npos);
}

TEST(TimeLevel, ToStringOmitsZeroOffsetAndCurrentIterate) {
  EXPECT_EQ(TimeLevel::current().to_string(), "[WindowEnd, k]");
}
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — `soma/base/time_level.hpp` file not found.

- [ ] **Step 3: Write the implementation**

`include/soma/base/time_level.hpp`:

```cpp
#pragma once

#include <string>

#include "soma/base/rational.hpp"

namespace soma {

enum class Anchor { WindowStart, WindowEnd, Absolute };

[[nodiscard]] constexpr const char* to_cstring(Anchor a) noexcept {
  switch (a) {
    case Anchor::WindowStart: return "WindowStart";
    case Anchor::WindowEnd:   return "WindowEnd";
    case Anchor::Absolute:    return "Absolute";
  }
  return "Unknown";
}

/// The time at which a value lives: t_ref + offset*dt, from iterate k.
/// Spec §4.3. Every DAG arc carries one; mismatches are lowering errors.
struct TimeLevel {
  Anchor   anchor = Anchor::WindowEnd;
  Rational offset{0};
  int      iterate = 0;  // 0 = current, -1 = previous iterate

  friend bool operator==(const TimeLevel&, const TimeLevel&) = default;

  [[nodiscard]] static TimeLevel current() { return TimeLevel{}; }

  [[nodiscard]] static TimeLevel lagged(int steps) {
    return TimeLevel{Anchor::WindowEnd, Rational{-steps}, 0};
  }

  [[nodiscard]] static TimeLevel previous_iterate() {
    return TimeLevel{Anchor::WindowEnd, Rational{0}, -1};
  }

  [[nodiscard]] std::string to_string() const {
    std::string s = "[";
    s += to_cstring(anchor);
    if (!offset.is_zero()) {
      if (offset > Rational{0}) s += "+";
      s += offset.to_string();
    }
    s += ", k";
    if (iterate != 0) {
      s += (iterate < 0) ? "-" : "+";
      s += std::to_string(iterate < 0 ? -iterate : iterate);
    }
    s += "]";
    return s;
  }
};

/// Strict by default (spec §4.3). A deliberate mismatch is authorised on the
/// channel via allow_time_level_mismatch(reason), not by relaxing this.
[[nodiscard]] inline bool compatible(const TimeLevel& produced,
                                     const TimeLevel& required) noexcept {
  return produced == required;
}

}  // namespace soma
```

- [ ] **Step 4: Register and run**

Add `soma_add_test(time_level_test unit/time_level_test.cpp)`.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug && cmake --build build -j
ctest --test-dir build --output-on-failure -R TimeLevel
```

Expected: 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add include/soma/base/time_level.hpp test/unit/time_level_test.cpp test/CMakeLists.txt
git commit -m "feat(soma): TimeLevel with anchor, rational offset, iterate"
```

---

### Task 0.6: Strong-typed identifiers and core enums

The vocabulary Phase 1 needs to declare its structures. Strong typing prevents passing a `ComponentId` where a `ChannelId` belongs — a real hazard once the graph has four parallel index spaces.

**Files:**
- Create: `include/soma/base/ids.hpp`
- Test: `test/unit/ids_test.cpp`
- Modify: `test/CMakeLists.txt`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `template <class Tag> class Id` — wraps `std::uint32_t`, `constexpr` default = invalid (`0xFFFFFFFF`), `explicit constexpr Id(std::uint32_t)`, `constexpr std::uint32_t value() const`, `constexpr bool valid() const`, defaulted `==` and `<=>`, and a `std::hash` specialisation
  - aliases `ComponentId`, `RegionId`, `FieldId`, `ChannelId`, `MeshId`, `NodeId` (DAG operator node), `ArcId`, `RegionIdx` (iteration region)
  - `enum class Discretization { FV, ContinuousGLL, DiscontinuousGLL, PointCloud }`
  - `enum class Conservation { None, Extensive, Intensive }`
  - `enum class Intent { Provides, Requires, Bidirectional }`
  - `enum class AccumOp { Instant, Sum, Mean, Min, Max, Integral }`
  - `enum class TemporalBasis { None, Constant, Linear, DG, Spline }`
  - `to_cstring(...)` for each enum

- [ ] **Step 1: Write the failing test**

`test/unit/ids_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include <soma/base/ids.hpp>

#include <unordered_set>

using soma::ChannelId;
using soma::ComponentId;

TEST(Id, DefaultIsInvalid) {
  constexpr ComponentId c;
  static_assert(!c.valid());
}

TEST(Id, ConstructedIsValid) {
  constexpr ComponentId c{3};
  static_assert(c.valid());
  static_assert(c.value() == 3u);
}

TEST(Id, ComparesWithinSameTag) {
  EXPECT_EQ(ComponentId{1}, ComponentId{1});
  EXPECT_NE(ComponentId{1}, ComponentId{2});
  EXPECT_LT(ComponentId{1}, ComponentId{2});
}

TEST(Id, IsHashable) {
  std::unordered_set<ComponentId> s;
  s.insert(ComponentId{1});
  s.insert(ComponentId{1});
  s.insert(ComponentId{2});
  EXPECT_EQ(s.size(), 2u);
}

// Distinct tags are distinct types: this must not compile if uncommented.
//   ComponentId c{1}; ChannelId ch = c;
TEST(Id, TagsAreDistinctTypes) {
  static_assert(!std::is_convertible_v<ComponentId, ChannelId>);
}

TEST(Enums, HaveStableNames) {
  EXPECT_STREQ(to_cstring(soma::Discretization::FV), "FV");
  EXPECT_STREQ(to_cstring(soma::Conservation::Extensive), "Extensive");
  EXPECT_STREQ(to_cstring(soma::Intent::Provides), "Provides");
  EXPECT_STREQ(to_cstring(soma::AccumOp::Mean), "Mean");
  EXPECT_STREQ(to_cstring(soma::TemporalBasis::DG), "DG");
}
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — `soma/base/ids.hpp` file not found.

- [ ] **Step 3: Write the implementation**

`include/soma/base/ids.hpp`:

```cpp
#pragma once

#include <compare>
#include <cstdint>
#include <functional>
#include <type_traits>

namespace soma {

/// Strong-typed index. Distinct Tag types are distinct C++ types, so a
/// ComponentId cannot be passed where a ChannelId is expected.
template <class Tag>
class Id {
 public:
  static constexpr std::uint32_t kInvalid = 0xFFFFFFFFu;

  constexpr Id() = default;
  explicit constexpr Id(std::uint32_t v) : value_(v) {}

  [[nodiscard]] constexpr std::uint32_t value() const noexcept { return value_; }
  [[nodiscard]] constexpr bool valid() const noexcept { return value_ != kInvalid; }

  friend constexpr bool operator==(const Id&, const Id&) = default;
  friend constexpr std::strong_ordering operator<=>(const Id&, const Id&) = default;

 private:
  std::uint32_t value_ = kInvalid;
};

struct ComponentTag {};
struct RegionTag {};
struct FieldTag {};
struct ChannelTag {};
struct MeshTag {};
struct NodeTag {};
struct ArcTag {};
struct IterRegionTag {};

using ComponentId = Id<ComponentTag>;
using RegionId    = Id<RegionTag>;
using FieldId     = Id<FieldTag>;
using ChannelId   = Id<ChannelTag>;
using MeshId      = Id<MeshTag>;
using NodeId      = Id<NodeTag>;       // operator node in the DAG
using ArcId       = Id<ArcTag>;        // SSA value
using RegionIdx   = Id<IterRegionTag>; // iteration region

enum class Discretization { FV, ContinuousGLL, DiscontinuousGLL, PointCloud };
enum class Conservation   { None, Extensive, Intensive };
enum class Intent         { Provides, Requires, Bidirectional };
enum class AccumOp        { Instant, Sum, Mean, Min, Max, Integral };
enum class TemporalBasis  { None, Constant, Linear, DG, Spline };

[[nodiscard]] constexpr const char* to_cstring(Discretization d) noexcept {
  switch (d) {
    case Discretization::FV:               return "FV";
    case Discretization::ContinuousGLL:    return "cGLL";
    case Discretization::DiscontinuousGLL: return "dGLL";
    case Discretization::PointCloud:       return "PointCloud";
  }
  return "Unknown";
}

[[nodiscard]] constexpr const char* to_cstring(Conservation c) noexcept {
  switch (c) {
    case Conservation::None:      return "None";
    case Conservation::Extensive: return "Extensive";
    case Conservation::Intensive: return "Intensive";
  }
  return "Unknown";
}

[[nodiscard]] constexpr const char* to_cstring(Intent i) noexcept {
  switch (i) {
    case Intent::Provides:      return "Provides";
    case Intent::Requires:      return "Requires";
    case Intent::Bidirectional: return "Bidirectional";
  }
  return "Unknown";
}

[[nodiscard]] constexpr const char* to_cstring(AccumOp a) noexcept {
  switch (a) {
    case AccumOp::Instant:  return "Instant";
    case AccumOp::Sum:      return "Sum";
    case AccumOp::Mean:     return "Mean";
    case AccumOp::Min:      return "Min";
    case AccumOp::Max:      return "Max";
    case AccumOp::Integral: return "Integral";
  }
  return "Unknown";
}

[[nodiscard]] constexpr const char* to_cstring(TemporalBasis t) noexcept {
  switch (t) {
    case TemporalBasis::None:     return "None";
    case TemporalBasis::Constant: return "Constant";
    case TemporalBasis::Linear:   return "Linear";
    case TemporalBasis::DG:       return "DG";
    case TemporalBasis::Spline:   return "Spline";
  }
  return "Unknown";
}

}  // namespace soma

template <class Tag>
struct std::hash<soma::Id<Tag>> {
  std::size_t operator()(const soma::Id<Tag>& id) const noexcept {
    return std::hash<std::uint32_t>{}(id.value());
  }
};
```

- [ ] **Step 4: Register and run**

Add `soma_add_test(ids_test unit/ids_test.cpp)`.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug && cmake --build build -j
ctest --test-dir build --output-on-failure
```

Expected: all Phase 0 tests PASS (smoke, Result, Diagnostics, Rational, Units, TimeLevel, Id, Enums).

- [ ] **Step 5: Commit**

```bash
git add include/soma/base/ids.hpp test/unit/ids_test.cpp test/CMakeLists.txt
git commit -m "feat(soma): strong-typed ids and core enums"
```

---

## Phase 0 exit criteria

- [ ] `cmake -B build -DCMAKE_BUILD_TYPE=Debug` configures with **no** optional dependency enabled.
- [ ] `cmake --build build -j` compiles clean under `-Wall -Wextra -Wpedantic`.
- [ ] `ctest --test-dir build --output-on-failure` is fully green.
- [ ] No file under `include/` includes any third-party header.
- [ ] `Rational::NoDriftOverManySteps` passes — this is the load-bearing one for Phase 3.
