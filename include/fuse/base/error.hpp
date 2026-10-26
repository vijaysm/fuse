#pragma once

#include <cstdint>
#include <source_location>

#include <Kokkos_Abort.hpp>   // Kokkos::abort, for FUSE_DEBUG_ASSERT on the device
#include <Kokkos_Macros.hpp>  // KOKKOS_IF_ON_HOST / KOKKOS_IF_ON_DEVICE
#include <stdexcept>
#include <string>
#include <string_view>

#include <fmt/format.h>

#include "fuse/base/backtrace.hpp"
#include "fuse/base/config.hpp"

// An undefined macro reads as 0 in `#if`, which would silently turn this into a
// no-exceptions build. The first run of the test suite did exactly that.
#ifndef FUSE_ENABLE_EXCEPTIONS
#error "fuse/base/config.hpp did not define FUSE_ENABLE_EXCEPTIONS"
#endif

/// Errors, invariants and the on-error hook.
///
/// Three kinds of failure, three spellings, one mechanism underneath:
///
///   FUSE_ASSERT(cond, "msg {}", x)     an INVARIANT: "this cannot happen".
///                                       Checked in EVERY build (I6). Throws
///                                       InvariantViolation.
///   FUSE_THROW(Type, "msg {}", x)      a recoverable error of a named type.
///   FUSE_REFUSE("E-CODE", "msg {}", x) a deliberate rejection (I4), carrying
///                                       the error code a test can assert on.
///
/// Every one captures a Backtrace at the throw site, records the MPI rank
/// when one has been announced, runs the on-error policy (print the trace,
/// attach a debugger, abort) and then throws -- or aborts, in a build with
/// FUSE_ENABLE_EXCEPTIONS=OFF, after the same policy has run. A call site
/// that names an error names it whether or not the build will throw it, so
/// the exception TYPES exist unconditionally; only the final verb changes.
///
/// FUSE_DEBUG_ASSERT(cond) is the one debug-only check, for device kernels
/// and inner loops where a throw is impossible or unaffordable. It is a
/// diagnostic aid and NEVER the only guard on an invariant -- the same
/// condition must be checked by a FUSE_ASSERT at the kernel's boundary.
namespace fuse {

// ------------------------------------------------------------------ types

/// Base of every FUSE exception. `what()` is the formatted message; the
/// code, location, rank and backtrace are separately accessible so a
/// diagnostic can carry them structurally instead of parsing a string.
class Error : public std::runtime_error {
 public:
  Error(std::string code, std::string message, std::source_location where,
        Backtrace trace, int rank);

  [[nodiscard]] std::string_view code() const noexcept { return code_; }
  [[nodiscard]] std::string_view message() const noexcept { return message_; }
  [[nodiscard]] const std::source_location& where() const noexcept { return where_; }
  [[nodiscard]] const Backtrace& backtrace() const noexcept { return trace_; }
  /// -1 until fuse::initialize (or the test harness) announces a rank.
  [[nodiscard]] int rank() const noexcept { return rank_; }

  /// Multi-line: header, location, and the backtrace if one was captured.
  [[nodiscard]] std::string report() const;

 private:
  std::string code_;
  std::string message_;
  std::source_location where_;
  Backtrace trace_;
  int rank_;
};

/// A FUSE_ASSERT fired: the library's own logic is wrong. Never recoverable.
class InvariantViolation : public Error { using Error::Error; };
/// The caller broke a documented precondition.
class InvalidArgument : public Error { using Error::Error; };
/// A null where an object was required (a specific InvalidArgument).
class NullPointer : public InvalidArgument { using InvalidArgument::InvalidArgument; };
/// A deliberate rejection under invariant I4, with the E-code tests assert on.
class Refusal : public Error { using Error::Error; };
/// A declared interface with no implementation behind it yet.
class NotImplemented : public Error { using Error::Error; };
/// File or stream access failed; the message names the path.
class IoError : public Error { using Error::Error; };

// ------------------------------------------------------------- the hook

/// What happens between "an error has been constructed" and "it is thrown".
/// Mirrors PETSc's -on_error_* options because that is the workflow people
/// already know for debugging parallel codes.
enum class OnError : std::uint8_t {
  Throw,           ///< just throw (default in tests: they assert on the throw)
  PrintBacktrace,  ///< print the report to stderr, then throw (default otherwise)
  AttachDebugger,  ///< spawn a debugger on this process, wait, then throw
  Abort,           ///< print the report, then std::abort (a core file, no unwinding)
};

/// The debugger to attach. Looked up on PATH when spawned.
enum class Debugger : std::uint8_t { Lldb, Gdb };

struct ErrorPolicy {
  OnError on_error = OnError::PrintBacktrace;
  bool capture_backtrace = true;    ///< off = Error carries an empty Backtrace
  Debugger debugger = Debugger::Lldb;
  std::string terminal = "xterm";   ///< "" = attach in the current terminal
  /// Ranks that get a debugger; empty = every rank. Others print and throw.
  std::string debugger_ranks;
  unsigned debugger_wait_seconds = 15;  ///< time given to the debugger to attach
};

/// Process-wide policy. Set by fuse::initialize from FUSE_ON_ERROR and
/// friends, or programmatically; read on every throw.
///
/// Returns a COPY, not a reference: REVIEW.md R-16 -- set_error_policy() is
/// documented as callable at any time, not only during single-threaded
/// startup, and takes policy_mutex() while copy-assigning ErrorPolicy's two
/// std::string members. A reference into that storage could be observed by
/// a reader mid-assignment (a genuine data race, not just a stale read); a
/// copy taken under the same lock cannot.
[[nodiscard]] ErrorPolicy error_policy() noexcept;
void set_error_policy(ErrorPolicy policy) noexcept;

/// Parse the environment: FUSE_ON_ERROR = throw | backtrace | debugger | abort,
/// FUSE_DEBUGGER = lldb | gdb, FUSE_DEBUGGER_TERMINAL, FUSE_DEBUGGER_RANKS,
/// FUSE_DEBUGGER_WAIT, FUSE_BACKTRACE = 0 | 1. Unset variables leave the
/// corresponding default; an unrecognised value is an InvalidArgument, not a
/// silent default (I4).
[[nodiscard]] ErrorPolicy error_policy_from_environment();

/// Announce this process's rank so every Error and report carries it. Called
/// by fuse::initialize once MPI is up; base itself never touches MPI.
void set_error_rank(int rank) noexcept;
[[nodiscard]] int error_rank() noexcept;

/// Process-wide set-up of the diagnostics layer -- the `fuse::initialize`
/// docs/dev/conventions.md and docs/dev/debugging.md describe. Reads the
/// FUSE_ON_ERROR family of variables once (`error_policy_from_environment()`),
/// installs the result (`set_error_policy()`) and announces `rank`
/// (`set_error_rank()`; -1 = no rank, the default for a process not under
/// MPI), so every Error raised afterwards follows the environment's policy
/// and carries the rank. Until a program calls it, the built-in defaults
/// apply and the environment is NOT read: FUSE_ON_ERROR and its companions
/// are effective only through this call (or a program's own
/// `set_error_policy(error_policy_from_environment())`).
///
/// Every entry point in this tree calls it: the examples' `main()`, the test
/// harness (test/support/fuse_test_main.cpp), pfuse's module import
/// (python/src/module.cpp) and the C ABI's `fuse_create()`/`fuse_setup()`
/// (src/abi/participant.cpp). A program that embeds the library directly
/// must call it itself, after MPI is up if it wants the rank.
///
/// An unrecognised variable value is the `InvalidArgument`
/// `error_policy_from_environment()` raises (I4): it propagates out of here
/// under the DEFAULT policy, which is still in force, and nothing is
/// installed. Calling it again re-reads the environment and re-announces.
void initialize(int rank = -1);

/// Restores the built-in default policy and rank -1. Nothing in the library
/// requires it; a program that pairs it with `initialize()` gets a process
/// whose diagnostics are back to the state before the environment was read.
void finalize() noexcept;

/// The command line `attach_debugger` would run, as a pure function so it
/// can be tested without spawning anything. `pid` is the process to attach.
[[nodiscard]] std::string debugger_command(const ErrorPolicy& policy, long pid);

/// Whether `rank` is in the policy's debugger rank list ("" = all).
[[nodiscard]] bool debugger_wanted_on_rank(const ErrorPolicy& policy, int rank);

/// Spawn the debugger from the policy on this process and block for
/// `debugger_wait_seconds`. POSIX only; returns false if the spawn failed.
/// IMPLEMENTED, UNTESTED: exercising it needs a terminal and a debugger,
/// which CI does not have. docs/dev/debugging.md tracks it.
bool attach_debugger(const ErrorPolicy& policy) noexcept;

/// Install a std::terminate handler that prints the report of an uncaught
/// fuse::Error (or the what() of any other exception) with a backtrace,
/// before aborting. Idempotent.
void install_terminate_handler() noexcept;

// -------------------------------------------------------------- plumbing

namespace detail {
/// Runs the on-error policy for a constructed error. Returns only if the
/// policy says "now throw"; aborts otherwise. Every macro below passes here.
void run_policy(const Error& error);
/// The exceptions-disabled verb: report, then abort. Never returns.
[[noreturn]] void abort_no_exceptions();

template <class E>
[[noreturn]] void raise(std::string code, std::string message, std::source_location where) {
  const auto& policy = error_policy();
  E error(std::move(code), std::move(message), where,
          policy.capture_backtrace ? Backtrace::capture(2) : Backtrace{}, error_rank());
  run_policy(error);
#if FUSE_ENABLE_EXCEPTIONS
  throw error;  // the DYNAMIC type E, so `catch (const Refusal&)` works
#else
  abort_no_exceptions();
#endif
}
}  // namespace detail
}  // namespace fuse

// ---------------------------------------------------------------- macros
//
// Variadic so a bare message and a fmt format-with-arguments both work.
// `__VA_OPT__` is C++20, which the library requires anyway.

#define FUSE_THROW(Type, ...)                                                        \
  ::fuse::detail::raise<Type>(#Type, ::fmt::format(__VA_ARGS__),                     \
                              ::std::source_location::current())

#define FUSE_REFUSE(code, ...)                                                       \
  ::fuse::detail::raise<::fuse::Refusal>((code), ::fmt::format(__VA_ARGS__),         \
                                         ::std::source_location::current())

/// Checked in every build. The condition is evaluated exactly once.
#define FUSE_ASSERT(cond, ...)                                                       \
  do {                                                                               \
    if (!(cond)) {                                                                   \
      ::fuse::detail::raise<::fuse::InvariantViolation>(                             \
          "FUSE_ASSERT(" #cond ")", ::fmt::format(__VA_ARGS__),                      \
          ::std::source_location::current());                                        \
    }                                                                                \
  } while (false)

/// Debug-only, never the sole guard on an invariant. Aborts rather than
/// throws so it is usable from device code and noexcept kernels.
///
/// Two bodies, selected at compile time by where the code is running. On the
/// host it reports the condition, location and a backtrace through
/// `debug_abort` -- which formats with fmt and captures a stack, both host-only
/// facilities, so that function cannot be called from a kernel. On the device
/// it calls `Kokkos::abort` with the condition text, which is what the backend
/// provides for exactly this. Before 2026-09-20 the macro called `debug_abort`
/// unconditionally, so a `KOKKOS_INLINE_FUNCTION` using it (the native mesh's
/// `measure()`/`centroid()`/`normal()`, R-14) could not compile for a
/// GPU-default build in Debug; no such build exists on this machine, which is
/// why it went unnoticed (REVIEW.md pass 2, backend R-14 residual).
#ifdef NDEBUG
#define FUSE_DEBUG_ASSERT(cond) \
  do {                          \
  } while (false)
#else
#define FUSE_DEBUG_ASSERT(cond)                                                    \
  do {                                                                             \
    if (!(cond)) {                                                                 \
      KOKKOS_IF_ON_HOST((::fuse::detail::debug_abort(                              \
          #cond, ::std::source_location::current());))                             \
      KOKKOS_IF_ON_DEVICE((::Kokkos::abort("FUSE_DEBUG_ASSERT(" #cond ") failed");)) \
    }                                                                              \
  } while (false)
#endif

namespace fuse::detail {
[[noreturn]] void debug_abort(const char* condition, std::source_location where) noexcept;
}
