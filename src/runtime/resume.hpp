#pragma once

/// Private to src/runtime/. A minimal, hand-written C++20 coroutine task
/// type that lets Runner::Impl::advance() (runner.cpp) and the two direct
/// `comp.advance(ctx)` sites in advance_window() (iteration.cpp) suspend
/// mid-period, at exactly the point an externally driven component's
/// advance() would otherwise run, and resume later from
/// src/abi/participant.cpp's fuse_begin_step()/fuse_end_step() --
/// so a host program can drive an externally driven component step by step.
///
/// WHY HAND-WRITTEN AND NOT A LIBRARY. The fix is the smallest
/// symmetric-transfer task type laid over the EXISTING function bodies:
/// step_period(), run_region(), advance_window() and advance() keep every
/// statement and its order -- the only new control flow anywhere is the
/// conditional suspend at a component whose `is_externally_driven()`
/// (component.hpp) is true.
///
/// WHY THE FRAME ARENA. A coroutine frame is heap-allocated through the
/// promise type's own `operator new`/`operator delete`; left undefined,
/// that resolves to the GLOBAL one, which several tests
/// (test/runtime/runner_test.cpp's `Runner.AllocatesNothingWhileRunning`,
/// test/parallel/runner_transport_test.cpp's transport allocation test,
/// ...) intercept to pin the in-process hot loop's allocation count at
/// zero after one uncounted warm-up period (spec Section 2 layer 3: "the
/// interpreted loop allocates nothing"). advance() is called once per
/// Tick, every period -- without reuse, converting it to a coroutine would
/// cost one real heap allocation and one free PER TICK, PER PERIOD,
/// forever: not just a number a test happens to pin, but the actual
/// allocation the spec asks the hot loop not to make. `FrameArena` below
/// is a size-keyed free list: the first call of each distinct coroutine
/// function (step_period, run_region, advance_window and advance each have
/// a DIFFERENT frame size, fixed by the compiler at compile time for that
/// function) mallocs once; every later create/destroy cycle of a frame OF
/// THAT SAME SIZE pops/pushes the free list instead of touching the
/// allocator again, so steady state is allocation-free by the same
/// measure those tests use. (A class-scoped `operator new` that called
/// straight through to `::operator new` would already be invisible to
/// those tests' override of the GLOBAL new/delete pair, since a
/// class-scoped overload hides rather than calls the global one by
/// default -- but invisible to the test and actually allocation-free are
/// different claims, and the reuse below is what makes the second one
/// true, not merely untestable.)
///
/// NOT THREAD SAFE, deliberately: `Component::advance()` runs on one
/// thread per rank, never concurrently with itself (component.hpp's own
/// THREADING contract), and the suspension itself happens on the calling
/// thread; nothing runs concurrently. An unsynchronised free list
/// is correct under that contract and would not be under a looser one.

#include <coroutine>
#include <cstdlib>
#include <exception>
#include <new>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include "fuse/base/error.hpp"

namespace fuse::detail {

/// Size-keyed free list backing every `Task<T>::promise_type` allocation
/// in this process. See this file's header comment for why reuse, not
/// just avoiding the global operator new, is the point.
class FrameArena {
 public:
  void* allocate(std::size_t n) {
    std::vector<void*>& free_list = free_lists_[n];
    if (!free_list.empty()) {
      void* p = free_list.back();
      free_list.pop_back();
      return p;
    }
    void* p = std::malloc(n == 0 ? 1 : n);
    if (p == nullptr) {
      FUSE_THROW(InvariantViolation,
                 "FrameArena::allocate: std::malloc({}) returned nullptr", n);
    }
    return p;
  }

  void deallocate(void* p, std::size_t n) noexcept {
    if (p == nullptr) return;
    free_lists_[n].push_back(p);
  }

  ~FrameArena() {
    for (auto& [size, free_list] : free_lists_) {
      for (void* p : free_list) std::free(p);
    }
  }

 private:
  std::unordered_map<std::size_t, std::vector<void*>> free_lists_;
};

/// One process-wide arena. `Component::advance()` -- the only call this
/// file's coroutines ever suspend inside -- is single-threaded per rank
/// (component.hpp's THREADING contract), so a function-local static
/// (constructed on first use, outliving every coroutine frame that uses
/// it) needs no synchronisation.
inline FrameArena& frame_arena() {
  static FrameArena arena;
  return arena;
}

/// A single-consumer, lazily-started coroutine task. `resume()` runs (or
/// resumes) the coroutine until it either suspends via `ExternalSuspend`
/// (below) or runs to completion via `co_return`; `done()` distinguishes
/// the two outcomes.
///
/// `co_await`ing a `Task<T>` from another coroutine of the SAME
/// specialisation chains via symmetric transfer (`await_suspend` returns
/// the child's own handle, a tail call the compiler makes with no added
/// stack frame): a suspend inside a task nested three levels deep
/// (advance(), awaited by run_region(), awaited by step_period()) returns
/// control directly to whichever caller invoked `.resume()` on the
/// OUTERMOST task. Nothing about that path depends on how deep the
/// nesting was, so a suspension cannot desynchronise two processes any
/// differently than the statement order it leaves unchanged already could.
template <class T>
class Task {
 public:
  class promise_type {
   public:
    static void* operator new(std::size_t n) { return frame_arena().allocate(n); }
    static void operator delete(void* p, std::size_t n) noexcept {
      frame_arena().deallocate(p, n);
    }

    Task get_return_object() {
      return Task{std::coroutine_handle<promise_type>::from_promise(*this)};
    }
    std::suspend_always initial_suspend() noexcept { return {}; }

    struct FinalAwaiter {
      bool await_ready() noexcept { return false; }
      std::coroutine_handle<> await_suspend(
          std::coroutine_handle<promise_type> h) noexcept {
        std::coroutine_handle<> c = h.promise().continuation;
        return c ? c : std::noop_coroutine();
      }
      void await_resume() noexcept {}
    };
    FinalAwaiter final_suspend() noexcept { return {}; }

    void return_value(T v) { result.emplace(std::move(v)); }

    // Runner::run_period()/Runner::resume_period() (runner.cpp) wrap the
    // outermost `.resume()`/`take_result()` call of this chain in the
    // SAME try/catch that wrapped a plain function call to step_period()
    // before it became a coroutine ("A function returning Result must
    // never throw" is initialize()'s own words for the same boundary).
    // Capturing here and rethrowing from `take_result()` (never from
    // `unhandled_exception()` itself) is what lets that boundary keep
    // working unchanged: the exception propagates through every
    // intervening symmetric-transfer level exactly where a direct call to
    // advance()/run_region()/advance_window() would have thrown before
    // this file existed, and is finally raised again at `take_result()`'s
    // own call site -- the one call that already sat inside a try/catch.
    void unhandled_exception() noexcept { exception = std::current_exception(); }

    std::optional<T>        result;
    std::exception_ptr      exception;
    std::coroutine_handle<> continuation{};
  };

  using handle_type = std::coroutine_handle<promise_type>;

  explicit Task(handle_type h) noexcept : handle_(h) {}
  Task(Task&& other) noexcept : handle_(std::exchange(other.handle_, {})) {}
  Task& operator=(Task&& other) noexcept {
    if (this != &other) {
      destroy();
      handle_ = std::exchange(other.handle_, {});
    }
    return *this;
  }
  Task(const Task&) = delete;
  Task& operator=(const Task&) = delete;
  ~Task() { destroy(); }

  /// Runs the coroutine from wherever it last suspended (or from the
  /// start, on the first call). Returns true if it ran to completion (a
  /// `co_return` was reached, and `take_result()` is now valid); false if
  /// it suspended again at `ExternalSuspend`, the only awaiter below that
  /// does not itself resume something.
  bool resume() {
    if (!handle_ || handle_.done()) return true;
    handle_.resume();
    return handle_.done();
  }

  [[nodiscard]] bool done() const noexcept { return !handle_ || handle_.done(); }

  /// The type-erased handle this Task wraps. NOT a substitute for
  /// `resume()`/`done()` on a Task that may have suspended somewhere
  /// below its own top frame via `co_await` on another `Task` -- see
  /// this class's own comment. Exists only so an external driver can
  /// seed its OWN "which handle is currently actually suspended" tracker
  /// (Runner::Impl::active_handle, runner_impl.hpp) the FIRST time a
  /// fresh Task is created, before anything inside it has run.
  [[nodiscard]] std::coroutine_handle<> raw_handle() const noexcept {
    return handle_;
  }

  /// Valid once done() is true. Rethrows whatever exception the coroutine
  /// body let escape (see promise_type::unhandled_exception()'s own
  /// comment on why this, not `unhandled_exception()` itself, is where
  /// that happens); otherwise moves the result out of the promise. Call
  /// at most once per completion.
  T take_result() {
    if (handle_.promise().exception) {
      std::rethrow_exception(handle_.promise().exception);
    }
    return std::move(*handle_.promise().result);
  }

  // Awaitable interface, for `co_await`ing this Task from a parent
  // coroutine of the same Task<T> specialisation -- see the class
  // comment above for why this is what makes nested suspension transit
  // transparently through every enclosing level.
  bool await_ready() const noexcept { return false; }
  handle_type await_suspend(std::coroutine_handle<> awaiting) noexcept {
    handle_.promise().continuation = awaiting;
    return handle_;  // symmetric transfer: the compiler tail-calls
                      // handle_.resume() with no added stack frame.
  }
  T await_resume() { return take_result(); }

 private:
  void destroy() {
    if (handle_) handle_.destroy();
  }

  handle_type handle_;
};

/// Suspends the CURRENT coroutine chain unconditionally, returning control
/// all the way up to whoever called `.resume()` -- see Task's own class
/// comment on symmetric transfer for why that return unwinds through
/// every enclosing `co_await` with no added stack frame. Used exactly
/// where a component's own `advance()` would otherwise run, so that by
/// the time this point is reached every instruction that feeds the
/// component has already executed and its Requires storage is current.
///
/// WHY THIS RECORDS ITS OWN HANDLE, AND WHY THAT MATTERS. Symmetric
/// transfer means a coroutine that merely `co_await`s a child Task is
/// NEVER itself the frame left "actually suspended" once that child goes
/// on to suspend deeper still -- the awaiting coroutine's own resume
/// point was already consumed by the tail call into the child, and
/// calling `.resume()` on the AWAITING coroutine's handle again does not
/// re-enter the child: it jumps straight to that resume point and calls
/// `await_resume()` on a child whose body never actually ran further,
/// reading an unset result (undefined behaviour, caught empirically while
/// building this file, not reasoned out in advance). The coroutine that
/// IS actually left suspended, at any depth, is always the one whose OWN
/// `co_await` evaluated THIS awaiter -- `await_suspend`'s own argument.
/// Writing that into `*active_handle` (Runner::Impl::active_handle,
/// runner_impl.hpp) is what lets `Runner::Impl::resume_period()` resume
/// the RIGHT frame on every call after the first, instead of re-entering
/// a stale top-level handle.
struct ExternalSuspend {
  /// Where to record the handle that is about to genuinely suspend.
  /// Never null at either of this file's two call sites (both pass
  /// `&Runner::Impl::active_handle`).
  std::coroutine_handle<>* active_handle;

  bool await_ready() const noexcept { return false; }
  void await_suspend(std::coroutine_handle<> h) const noexcept {
    *active_handle = h;
  }
  void await_resume() const noexcept {}
};

}  // namespace fuse::detail
