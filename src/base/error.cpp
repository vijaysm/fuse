#include "fuse/base/config.hpp"
#include "fuse/base/error.hpp"

#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <string>
#include <vector>

#include <fmt/format.h>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/wait.h>
#define FUSE_HAVE_POSIX_SPAWN_PATH 1
#endif

namespace fuse {

// ---------------------------------------------------------------- Error

Error::Error(std::string code, std::string message, std::source_location where, Backtrace trace,
             int rank)
    : std::runtime_error(fmt::format("[{}] {}", code, message)),
      code_(std::move(code)),
      message_(std::move(message)),
      where_(where),
      trace_(std::move(trace)),
      rank_(rank) {}

std::string Error::report() const {
  std::string out;
  if (rank_ >= 0) fmt::format_to(std::back_inserter(out), "[rank {}] ", rank_);
  fmt::format_to(std::back_inserter(out), "{}\n    at {}:{} in {}\n", what(), where_.file_name(),
                 where_.line(), where_.function_name());
  if (!trace_.empty()) {
    out += "  backtrace:\n";
    out += trace_.format();
  }
  return out;
}

// --------------------------------------------------------------- policy

namespace {
std::mutex& policy_mutex() {
  static std::mutex m;
  return m;
}
ErrorPolicy& policy_storage() {
  static ErrorPolicy p;
  return p;
}
std::atomic<int> g_rank{-1};

void write_stderr(const std::string& s) noexcept {
  // fputs, not fmt::print: this runs on the failure path, possibly under a
  // terminate handler, and must not throw or allocate more than it already has.
  std::fputs(s.c_str(), stderr);  // NOLINT(fuse-raw-io): failure path, must not throw
  std::fflush(stderr);
}
}  // namespace

// REVIEW.md R-16: policy_storage() is mutated under policy_mutex() by
// set_error_policy() below; a reader that returned a reference into it with
// no lock could observe an ErrorPolicy (two std::string members) mid
// copy-assignment -- a data race, not merely a stale-read hazard. Taking the
// same lock here and returning a copy means the lock is never held past this
// function, so this stays cheap on the hot error path without widening the
// mutex's scope to every raise() call downstream.
ErrorPolicy error_policy() noexcept {
  const std::lock_guard<std::mutex> lock(policy_mutex());
  return policy_storage();
}

void set_error_policy(ErrorPolicy policy) noexcept {
  const std::lock_guard<std::mutex> lock(policy_mutex());
  policy_storage() = std::move(policy);
}

void set_error_rank(int rank) noexcept { g_rank.store(rank, std::memory_order_relaxed); }
int error_rank() noexcept { return g_rank.load(std::memory_order_relaxed); }

namespace {
const char* env(const char* name) noexcept {
  const char* v = std::getenv(name);  // NOLINT(concurrency-mt-unsafe): read once at startup
  return (v != nullptr && *v != '\0') ? v : nullptr;
}
}  // namespace

ErrorPolicy error_policy_from_environment() {
  ErrorPolicy p;
  if (const char* v = env("FUSE_ON_ERROR")) {
    const std::string_view s(v);
    if (s == "throw") p.on_error = OnError::Throw;
    else if (s == "backtrace") p.on_error = OnError::PrintBacktrace;
    else if (s == "debugger") p.on_error = OnError::AttachDebugger;
    else if (s == "abort") p.on_error = OnError::Abort;
    else FUSE_THROW(InvalidArgument, "FUSE_ON_ERROR='{}': expected throw|backtrace|debugger|abort", s);
  }
  if (const char* v = env("FUSE_DEBUGGER")) {
    const std::string_view s(v);
    if (s == "lldb") p.debugger = Debugger::Lldb;
    else if (s == "gdb") p.debugger = Debugger::Gdb;
    else FUSE_THROW(InvalidArgument, "FUSE_DEBUGGER='{}': expected lldb|gdb", s);
  }
  if (const char* v = env("FUSE_DEBUGGER_TERMINAL")) p.terminal = (std::string_view(v) == "none") ? "" : v;
  if (const char* v = env("FUSE_DEBUGGER_RANKS")) p.debugger_ranks = v;
  if (const char* v = env("FUSE_DEBUGGER_WAIT")) {
    char* end = nullptr;
    const long n = std::strtol(v, &end, 10);
    if (end == v || *end != '\0' || n < 0) FUSE_THROW(InvalidArgument, "FUSE_DEBUGGER_WAIT='{}': expected a non-negative integer", v);
    p.debugger_wait_seconds = static_cast<unsigned>(n);
  }
  if (const char* v = env("FUSE_BACKTRACE")) {
    const std::string_view s(v);
    if (s == "0") p.capture_backtrace = false;
    else if (s == "1") p.capture_backtrace = true;
    else FUSE_THROW(InvalidArgument, "FUSE_BACKTRACE='{}': expected 0|1", s);
  }
  return p;
}

// ------------------------------------------------------------- debugger

std::string debugger_command(const ErrorPolicy& policy, long pid) {
  const std::string attach =
      fmt::format("{} -p {}", policy.debugger == Debugger::Gdb ? "gdb" : "lldb", pid);
  if (policy.terminal.empty()) return attach;
  // `-e` is honoured by xterm, urxvt, gnome-terminal (as --) and Terminal
  // wrappers alike; the title carries the rank so eight windows are tellable.
  return fmt::format("{} -T 'fuse rank {} pid {}' -e {}", policy.terminal, error_rank(), pid, attach);
}

bool debugger_wanted_on_rank(const ErrorPolicy& policy, int rank) {
  if (policy.debugger_ranks.empty()) return true;
  std::string_view s = policy.debugger_ranks;
  while (!s.empty()) {
    const auto comma = s.find(',');
    const std::string_view tok = s.substr(0, comma);
    char* end = nullptr;
    const std::string tmp(tok);
    const long n = std::strtol(tmp.c_str(), &end, 10);
    if (end != tmp.c_str() && *end == '\0' && n == rank) return true;
    if (comma == std::string_view::npos) break;
    s.remove_prefix(comma + 1);
  }
  return false;
}

bool attach_debugger(const ErrorPolicy& policy) noexcept {
#if FUSE_HAVE_POSIX_SPAWN_PATH
  const long pid = static_cast<long>(::getpid());
  const std::string cmd = debugger_command(policy, pid);
  write_stderr(fmt::format("[fuse] attaching debugger: {}\n", cmd));
  const pid_t child = ::fork();
  if (child < 0) return false;
  if (child == 0) {
    ::execl("/bin/sh", "sh", "-c", cmd.c_str(), static_cast<char*>(nullptr));
    ::_exit(127);
  }
  // Give the debugger time to attach; it will stop us itself once it has.
  ::sleep(policy.debugger_wait_seconds);
  return true;
#else
  (void)policy;
  return false;
#endif
}

// ------------------------------------------------------------ terminate

namespace {
std::terminate_handler g_previous = nullptr;
std::once_flag g_installed;

[[noreturn]] void fuse_terminate() noexcept {
  if (const auto ex = std::current_exception()) {
    try {
      std::rethrow_exception(ex);
    } catch (const Error& e) {
      write_stderr("[fuse] terminating on uncaught fuse::Error\n" + e.report());
    } catch (const std::exception& e) {
      write_stderr(fmt::format("[fuse] terminating on uncaught exception: {}\n  backtrace:\n{}", e.what(),
                               Backtrace::capture(1).format()));
    } catch (...) {
      write_stderr("[fuse] terminating on uncaught non-standard exception\n");
    }
  } else {
    write_stderr("[fuse] std::terminate called without an active exception\n  backtrace:\n" +
                 Backtrace::capture(1).format());
  }
  // REVIEW.md R-35: chain to whatever terminate handler was installed
  // before ours, rather than silently discarding it. An embedding
  // application (or a third-party library FUSE is linked into -- MOAB,
  // PETSc, Kokkos) may have installed its own handler for its own cleanup
  // (flushing output, calling MPI_Abort in a way its own runtime expects);
  // FUSE's own diagnostic is already on stderr by this point, and
  // std::terminate's contract is that a handler never returns, so this is
  // the last thing fuse_terminate() does. std::abort() below is the
  // fallback for the one case with nothing to chain to, and for a chained
  // handler that violates its own contract and returns anyway.
  if (g_previous != nullptr) g_previous();
  std::abort();
}
}  // namespace

void install_terminate_handler() noexcept {
  std::call_once(g_installed, [] { g_previous = std::set_terminate(&fuse_terminate); });
}

// ----------------------------------------------------------------- fail

namespace detail {

void run_policy(const Error& error) {
  const ErrorPolicy& policy = error_policy();
  switch (policy.on_error) {
    case OnError::Throw:
      return;
    case OnError::PrintBacktrace:
      write_stderr(error.report());
      return;
    case OnError::AttachDebugger:
      write_stderr(error.report());
      if (debugger_wanted_on_rank(policy, error_rank())) attach_debugger(policy);
      return;
    case OnError::Abort:
      write_stderr(error.report());
      std::abort();
  }
}

void abort_no_exceptions() {
  // The report has already been written by run_policy; fail fast, keep the core.
  write_stderr("[fuse] exceptions are disabled in this build; aborting\n");
  std::abort();
}

void debug_abort(const char* condition, std::source_location where) noexcept {
  write_stderr(fmt::format("[fuse] FUSE_DEBUG_ASSERT({}) failed at {}:{}\n  backtrace:\n{}", condition,
                           where.file_name(), where.line(), Backtrace::capture(1).format()));
  std::abort();
}

}  // namespace detail
}  // namespace fuse
