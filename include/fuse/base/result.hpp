#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <fmt/format.h>

namespace fuse {

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

/// The payload of a failed Result: what went wrong, as data. Distinct from
/// fuse::Error (error.hpp), which is the exception. A boundary converts one
/// into the other; nothing else should hold both.
struct Failure {
  std::string code;
  std::string message;
  /// Provenance of a failure that reached this rank through a collective
  /// agreement (fuse/field/collective_agreement.hpp): the rank whose own
  /// refusal is reported and that rank's own code. -1 and empty for a
  /// failure computed on this rank itself. `Failure{code, message}` keeps
  /// working: both are defaulted (KNOWN_ISSUES.md D97, 2026-10-30).
  int origin_rank = -1;
  std::string origin_code;
  int origin_status = 0;  ///< the origin's numeric status, for callers that have one
};

/// Recoverable-error return type for the execution boundary (U2). Invariants
/// use FUSE_ASSERT; deliberate rejections inside a call use FUSE_REFUSE; a
/// Result carries the outcome OUT of validate()/run()/initialize().
template <class T>
class Result {
 public:
  Result(T value) : storage_(std::move(value)) {}       // NOLINT: implicit
  Result(Failure error) : storage_(std::move(error)) {}   // NOLINT: implicit

  [[nodiscard]] bool ok() const noexcept {
    return std::holds_alternative<T>(storage_);
  }
  explicit operator bool() const noexcept { return ok(); }

  [[nodiscard]] T&       value() &      { return std::get<T>(storage_); }
  [[nodiscard]] const T& value() const & { return std::get<T>(storage_); }
  /// Moves the value out. Without this overload `std::move(r).value()` binds
  /// to the lvalue accessor above and silently COPIES -- which compiled
  /// wherever T happened to be copyable, so the mistake was invisible, and
  /// failed to compile only on a move-only T like `unique_ptr<LinearOperator>`.
  /// The ref-qualifiers on the two above are what make this overload reachable.
  [[nodiscard]] T&&      value() &&     { return std::get<T>(std::move(storage_)); }
  [[nodiscard]] const Failure& error() const { return std::get<Failure>(storage_); }

 private:
  std::variant<T, Failure> storage_;
};

template <>
class Result<void> {
 public:
  Result(Failure error) : error_(std::move(error)), ok_(false) {}  // NOLINT
  static Result<void> success() { return Result<void>{}; }

  [[nodiscard]] bool ok() const noexcept { return ok_; }
  explicit operator bool() const noexcept { return ok_; }
  [[nodiscard]] const Failure& error() const noexcept { return error_; }

 private:
  Result() = default;
  Failure error_{};
  bool  ok_ = true;
};

}  // namespace fuse

// Diagnostics format through fmt (U2), so `fmt::print("{}", r.diagnostics())`
// is the whole reporting story and no caller re-implements to_string().
template <>
struct fmt::formatter<fuse::Diagnostic> : fmt::formatter<std::string_view> {
  template <class Ctx>
  auto format(const fuse::Diagnostic& d, Ctx& ctx) const {
    auto out = fmt::format_to(ctx.out(), "{} {}: {}",
                              d.severity == fuse::Severity::Error ? "error" : "warning", d.code, d.message);
    if (!d.context.empty()) out = fmt::format_to(out, " [{}]", d.context);
    return out;
  }
};

template <>
struct fmt::formatter<fuse::Diagnostics> : fmt::formatter<std::string_view> {
  template <class Ctx>
  auto format(const fuse::Diagnostics& ds, Ctx& ctx) const {
    auto out = ctx.out();
    for (const auto& d : ds.items()) out = fmt::format_to(out, "{}\n", d);
    return out;
  }
};

template <>
struct fmt::formatter<fuse::Failure> : fmt::formatter<std::string_view> {
  template <class Ctx>
  auto format(const fuse::Failure& f, Ctx& ctx) const {
    return fmt::format_to(ctx.out(), "[{}] {}", f.code, f.message);
  }
};
