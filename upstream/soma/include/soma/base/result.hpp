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
