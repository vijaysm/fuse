// The error machinery, tested for the properties that make it an invariant
// mechanism rather than a logging convenience.
#include "fuse/base/error.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

using namespace fuse;

struct QuietPolicy {
  // Tests assert on the throw; printing every expected failure to stderr
  // would bury real ones. Restored on scope exit.
  ErrorPolicy previous = error_policy();
  QuietPolicy() {
    ErrorPolicy p = previous;
    p.on_error = OnError::Throw;
    set_error_policy(p);
  }
  ~QuietPolicy() { set_error_policy(previous); }
};

// ------------------------------------------------------------ FUSE_ASSERT

TEST(Error, AssertIsCheckedInEveryBuild) {
  QuietPolicy quiet;
  // If this test is compiled with -DNDEBUG and still passes, FUSE_ASSERT is
  // not compiled out. The Release CI leg runs this same binary (I6).
  EXPECT_THROW(FUSE_ASSERT(1 == 2, "arithmetic broke: {} vs {}", 1, 2), InvariantViolation);
#ifdef NDEBUG
  SUCCEED() << "verified under NDEBUG";
#endif
}

TEST(Error, AssertEvaluatesTheConditionExactlyOnce) {
  QuietPolicy quiet;
  int evaluations = 0;
  FUSE_ASSERT((++evaluations, true), "never");
  EXPECT_EQ(evaluations, 1);
}

TEST(Error, AssertCarriesConditionCodeMessageAndLocation) {
  QuietPolicy quiet;
  try {
    FUSE_ASSERT(false, "context {}", 42);
    FAIL() << "did not throw";
  } catch (const InvariantViolation& e) {
    EXPECT_EQ(e.code(), "FUSE_ASSERT(false)");
    EXPECT_EQ(e.message(), "context 42");
    EXPECT_NE(std::string(e.where().file_name()).find("error_test.cpp"), std::string::npos);
    EXPECT_NE(std::string(e.what()).find("context 42"), std::string::npos);
  }
}

// -------------------------------------------------------------- FUSE_THROW

TEST(Error, ThrowPreservesTheDynamicType) {
  QuietPolicy quiet;
  // The catch site names the specific type; slicing to Error would make every
  // catch(const Refusal&) in the library dead code.
  EXPECT_THROW(FUSE_THROW(InvalidArgument, "bad {}", "arg"), InvalidArgument);
  EXPECT_THROW(FUSE_THROW(NullPointer, "null"), InvalidArgument);  // is-a
  EXPECT_THROW(FUSE_THROW(NotImplemented, "later"), NotImplemented);
  EXPECT_THROW(FUSE_THROW(IoError, "cannot open {}", "x.h5"), IoError);
  bool caught_as_base = false;
  try {
    FUSE_THROW(IoError, "x");
  } catch (const Error&) {
    caught_as_base = true;
  }
  EXPECT_TRUE(caught_as_base);
}

TEST(Error, RefusalCarriesItsCode) {
  QuietPolicy quiet;
  try {
    FUSE_REFUSE("E-RT-TRANSPORT-OVERLAP", "rank sets {} and {} overlap", "fluid", "solid");
    FAIL();
  } catch (const Refusal& e) {
    EXPECT_EQ(e.code(), "E-RT-TRANSPORT-OVERLAP");
    EXPECT_EQ(e.message(), "rank sets fluid and solid overlap");
  }
}

// --------------------------------------------------------------- backtrace

[[gnu::noinline]] void deep_three() { FUSE_THROW(InvalidArgument, "from deep_three"); }
[[gnu::noinline]] void deep_two() { deep_three(); }
[[gnu::noinline]] void deep_one() { deep_two(); }

TEST(Error, BacktraceIsCapturedAtTheThrowSite) {
  QuietPolicy quiet;
  try {
    deep_one();
    FAIL();
  } catch (const Error& e) {
    // At least the three deep_* frames plus this test body.
    EXPECT_GE(e.backtrace().size(), 4u);
    const std::string report = e.report();
    EXPECT_NE(report.find("backtrace:"), std::string::npos);
    EXPECT_NE(report.find("error_test.cpp"), std::string::npos);
  }
}

TEST(Error, BacktraceCaptureCanBeDisabledByPolicy) {
  QuietPolicy quiet;
  ErrorPolicy p = error_policy();
  p.capture_backtrace = false;
  set_error_policy(p);
  try {
    FUSE_THROW(InvalidArgument, "no trace");
    FAIL();
  } catch (const Error& e) {
    EXPECT_TRUE(e.backtrace().empty());
    EXPECT_EQ(e.report().find("backtrace:"), std::string::npos);
  }
}

// REVIEW.md R-16: error_policy() used to return a reference into
// policy_storage() with no lock, while set_error_policy() writes that same
// storage under policy_mutex() -- a data race on ErrorPolicy's two
// std::string members, invisible to any functional test and only reliably
// caught by a race detector. This does not assert anything beyond "no crash,
// no sanitizer report": a plain build cannot observe the race directly, but
// -fsanitize=thread instruments every load/store this loop performs and
// would have flagged the pre-fix code (a reference read of policy_storage()
// racing set_error_policy()'s locked copy-assignment) as a data race.
TEST(Error, ErrorPolicyIsSafeUnderConcurrentReadAndWrite) {
  const ErrorPolicy previous = error_policy();
  std::atomic<bool> stop{false};

  std::thread writer([&] {
    ErrorPolicy p = previous;
    p.on_error = OnError::Throw;
    while (!stop.load(std::memory_order_relaxed)) {
      p.terminal = (p.terminal == "xterm") ? "alacritty" : "xterm";
      p.debugger_ranks = (p.debugger_ranks.empty()) ? "0-3" : "";
      set_error_policy(p);
    }
  });

  std::thread reader([&] {
    QuietPolicy quiet;
    for (int i = 0; i < 2000; ++i) {
      try {
        FUSE_REFUSE("E-TEST-RACE", "concurrent read {}", i);
        FAIL();
      } catch (const Error&) {
        // Expected; the assertion here is that error_policy() returns a
        // consistent snapshot, not any particular value.
      }
    }
  });

  reader.join();
  stop.store(true, std::memory_order_relaxed);
  writer.join();
  set_error_policy(previous);
}

TEST(Error, ReportCarriesTheRankOnceAnnounced) {
  QuietPolicy quiet;
  set_error_rank(7);
  try {
    FUSE_THROW(InvalidArgument, "ranked");
    FAIL();
  } catch (const Error& e) {
    EXPECT_EQ(e.rank(), 7);
    EXPECT_EQ(e.report().rfind("[rank 7] ", 0), 0u);
  }
  set_error_rank(-1);
}

// ---------------------------------------------------------- policy parsing

struct ScopedEnv {
  std::string name;
  ScopedEnv(const char* n, const char* v) : name(n) { ::setenv(n, v, 1); }
  ~ScopedEnv() { ::unsetenv(name.c_str()); }
};

TEST(ErrorPolicy, EnvironmentSelectsEveryMode) {
  QuietPolicy quiet;
  {
    ScopedEnv e("FUSE_ON_ERROR", "debugger");
    EXPECT_EQ(error_policy_from_environment().on_error, OnError::AttachDebugger);
  }
  {
    ScopedEnv e("FUSE_ON_ERROR", "abort");
    EXPECT_EQ(error_policy_from_environment().on_error, OnError::Abort);
  }
  {
    ScopedEnv e("FUSE_ON_ERROR", "throw");
    EXPECT_EQ(error_policy_from_environment().on_error, OnError::Throw);
  }
  {
    ScopedEnv a("FUSE_DEBUGGER", "gdb");
    ScopedEnv b("FUSE_DEBUGGER_RANKS", "0,3");
    ScopedEnv c("FUSE_DEBUGGER_WAIT", "42");
    ScopedEnv d("FUSE_BACKTRACE", "0");
    ScopedEnv t("FUSE_DEBUGGER_TERMINAL", "none");
    const auto p = error_policy_from_environment();
    EXPECT_EQ(p.debugger, Debugger::Gdb);
    EXPECT_EQ(p.debugger_ranks, "0,3");
    EXPECT_EQ(p.debugger_wait_seconds, 42u);
    EXPECT_FALSE(p.capture_backtrace);
    EXPECT_TRUE(p.terminal.empty());
  }
}

TEST(ErrorPolicy, UnrecognisedEnvironmentValueIsRefusedNotDefaulted) {
  QuietPolicy quiet;
  ScopedEnv e("FUSE_ON_ERROR", "sometimes");
  EXPECT_THROW(error_policy_from_environment(), InvalidArgument);  // I4
  ScopedEnv w("FUSE_DEBUGGER_WAIT", "soon");
  ::unsetenv("FUSE_ON_ERROR");
  EXPECT_THROW(error_policy_from_environment(), InvalidArgument);
}

// `fuse::initialize()` is what makes the environment effective: parsing
// alone (`error_policy_from_environment()`) installs nothing, and until this
// existed no program in the tree ever called `set_error_policy()` with the
// parsed result, so FUSE_ON_ERROR and its companions were documented but
// inert. It also announces the rank, and `finalize()` undoes both.
TEST(ErrorPolicy, InitializeInstallsTheEnvironmentPolicyAndTheRank) {
  QuietPolicy quiet;  // restores the policy this test replaces
  const int previous_rank = error_rank();
  {
    ScopedEnv e("FUSE_ON_ERROR", "throw");
    ScopedEnv b("FUSE_BACKTRACE", "0");
    fuse::initialize(3);
    EXPECT_EQ(error_policy().on_error, OnError::Throw);
    EXPECT_FALSE(error_policy().capture_backtrace);
    EXPECT_EQ(error_rank(), 3);
  }
  fuse::finalize();
  EXPECT_EQ(error_policy().on_error, ErrorPolicy{}.on_error);
  EXPECT_TRUE(error_policy().capture_backtrace);
  EXPECT_EQ(error_rank(), -1);
  set_error_rank(previous_rank);  // the harness's own announcement
}

TEST(ErrorPolicy, InitializeLeavesThePolicyAloneWhenTheEnvironmentIsRefused) {
  QuietPolicy quiet;
  ErrorPolicy before = error_policy();
  before.debugger_wait_seconds = 7;  // something the environment below does not set
  set_error_policy(before);
  ScopedEnv e("FUSE_ON_ERROR", "sometimes");
  EXPECT_THROW(fuse::initialize(), InvalidArgument);  // I4: refused, not defaulted
  EXPECT_EQ(error_policy().debugger_wait_seconds, 7u)
      << "a refused environment must not install a partial policy";
}

TEST(ErrorPolicy, DebuggerCommandIsWhatWouldBeRun) {
  ErrorPolicy p;
  p.debugger = Debugger::Lldb;
  p.terminal = "";
  EXPECT_EQ(debugger_command(p, 1234), "lldb -p 1234");
  p.debugger = Debugger::Gdb;
  p.terminal = "xterm";
  set_error_rank(2);
  EXPECT_EQ(debugger_command(p, 1234), "xterm -T 'fuse rank 2 pid 1234' -e gdb -p 1234");
  set_error_rank(-1);
}

TEST(ErrorPolicy, DebuggerRankSelection) {
  ErrorPolicy p;
  EXPECT_TRUE(debugger_wanted_on_rank(p, 5));  // "" = all ranks
  p.debugger_ranks = "0,3";
  EXPECT_TRUE(debugger_wanted_on_rank(p, 0));
  EXPECT_TRUE(debugger_wanted_on_rank(p, 3));
  EXPECT_FALSE(debugger_wanted_on_rank(p, 1));
  EXPECT_FALSE(debugger_wanted_on_rank(p, 30));  // no prefix matching
}

// ---------------------------------------------------- debug-only assertion

TEST(Error, DebugAssertIsSilentWhenTrue) {
  FUSE_DEBUG_ASSERT(2 + 2 == 4);
  SUCCEED();
}

TEST(ErrorDeath, DebugAssertAbortsInDebugBuildsOnly) {
#ifdef NDEBUG
  FUSE_DEBUG_ASSERT(false);  // compiled out: must NOT abort
  SUCCEED();
#else
  EXPECT_DEATH(FUSE_DEBUG_ASSERT(false), "FUSE_DEBUG_ASSERT");
#endif
}

TEST(ErrorDeath, AbortPolicyAbortsWithReport) {
  ErrorPolicy p;
  p.on_error = OnError::Abort;
  set_error_policy(p);
  EXPECT_DEATH(FUSE_THROW(InvalidArgument, "fatal by policy"), "fatal by policy");
  set_error_policy(ErrorPolicy{});
}

// ------------------------------------------------- the terminate handler

// An uncaught fuse::Error must not die silently: the handler is the last place a
// report can still be printed, and on a parallel run it is often the only output a
// rank produces.
//
// These call std::terminate() from inside a catch block rather than letting an
// exception escape, because GoogleTest wraps a death-test statement in its own
// try/catch and reports an escaping exception instead of letting it reach
// std::terminate. The handler's contract is "report whatever exception is in
// flight", and current_exception() is exactly that inside a catch -- so this
// exercises the handler itself. What it does NOT cover is the runtime's own path
// from an uncaught throw to std::terminate, which is the language's behaviour and
// not ours.
TEST(ErrorDeath, TerminateHandlerReportsAnUncaughtFuseError) {
  EXPECT_DEATH(
      {
        fuse::install_terminate_handler();
        fuse::ErrorPolicy p;
        p.on_error = fuse::OnError::Throw;  // no report before the throw
        fuse::set_error_policy(p);
        try {
          FUSE_REFUSE("E-TEST-UNCAUGHT", "nobody catches this");
        } catch (...) {
          std::terminate();
        }
      },
      "terminating on uncaught fuse::Error");
}

TEST(ErrorDeath, TerminateHandlerReportCarriesCodeLocationAndBacktrace) {
  EXPECT_DEATH(
      {
        fuse::install_terminate_handler();
        fuse::ErrorPolicy p;
        p.on_error = fuse::OnError::Throw;
        fuse::set_error_policy(p);
        try {
          FUSE_THROW(fuse::IoError, "cannot open {}", "mesh.h5m");
        } catch (...) {
          std::terminate();
        }
      },
      "IoError.*cannot open mesh.h5m(.|\\n)*error_test.cpp(.|\\n)*backtrace:");
}

// A foreign exception reaching terminate still gets a trace -- captured in the
// handler rather than at the throw site, which is the best available.
TEST(ErrorDeath, TerminateHandlerReportsANonFuseException) {
  EXPECT_DEATH(
      {
        fuse::install_terminate_handler();
        try {
          throw std::runtime_error("from somebody else");
        } catch (...) {
          std::terminate();
        }
      },
      "uncaught exception: from somebody else(.|\\n)*backtrace:");
}

// REVIEW.md R-35: g_previous (the terminate handler installed before
// fuse::install_terminate_handler() ran) used to be captured and never
// used -- an embedding application's own handler (flushing output,
// MPI_Abort, ...) was silently discarded. install_terminate_handler() is
// call_once and every earlier ErrorDeath test's death-test child already
// ran it once inside its own forked child (which never returns to this
// process), so the flag is still unset here: set_terminate() below installs
// a genuine "previous" handler for THIS child to chain to.
[[noreturn]] void previous_handler_marker() noexcept {
  std::fputs("[embedder] previous handler ran\n", stderr);
  std::fflush(stderr);
  std::abort();
}

TEST(ErrorDeath, TerminateHandlerChainsToAPreviouslyInstalledHandler) {
  EXPECT_DEATH(
      {
        std::set_terminate(&previous_handler_marker);
        fuse::install_terminate_handler();
        fuse::ErrorPolicy p;
        p.on_error = fuse::OnError::Throw;
        fuse::set_error_policy(p);
        try {
          FUSE_REFUSE("E-TEST-CHAIN", "trigger chaining");
        } catch (...) {
          std::terminate();
        }
      },
      "terminating on uncaught fuse::Error(.|\\n)*\\[embedder\\] previous handler ran");
}

TEST(ErrorDeath, TerminateHandlerIsIdempotent) {
  EXPECT_DEATH(
      {
        for (int i = 0; i < 3; ++i) fuse::install_terminate_handler();
        std::terminate();
      },
      "std::terminate called without an active exception");
}

}  // namespace
