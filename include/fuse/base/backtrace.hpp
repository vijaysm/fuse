#pragma once

#include <array>
#include <cstddef>
#include <string>

namespace fuse {

/// A captured call stack: raw return addresses at capture time, symbolized
/// only when formatted. Capture is what happens on every throw, so it must be
/// cheap and must not allocate (invariant I5 holds on the error path too);
/// symbolization allocates freely because it happens once, when a human
/// reads the result.
///
/// Uses `execinfo.h` (macOS, glibc, musl) rather than `std::stacktrace`:
/// libc++ does not ship the C++23 facility as of clang 21, and the library
/// is C++20. When `std::stacktrace` is available everywhere FUSE builds, this
/// class becomes a thin wrapper over it with the same interface.
class Backtrace {
 public:
  static constexpr std::size_t kMaxFrames = 64;

  /// Capture the current stack, skipping `skip` innermost frames so the
  /// trace starts at the caller rather than inside the capture machinery.
  [[nodiscard]] static Backtrace capture(int skip = 1) noexcept;

  /// An empty trace: what `Error` holds when capture is disabled by policy.
  Backtrace() = default;

  [[nodiscard]] std::size_t size() const noexcept { return count_; }
  [[nodiscard]] bool empty() const noexcept { return count_ == 0; }

  /// One frame per line: index, address, demangled symbol when the dynamic
  /// symbol table has it, module name otherwise. Never throws; a frame that
  /// cannot be symbolized is printed as its address.
  [[nodiscard]] std::string format() const;

 private:
  std::array<void*, kMaxFrames> frames_{};
  std::size_t count_ = 0;
};

}  // namespace fuse
