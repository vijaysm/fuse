#pragma once

// ---------------------------------------------------------------------------
// Allocation counting.
//
// Spec §2 layer 3 requires the interpreted loop to allocate nothing. Replacing
// the global operator new is the only way to *measure* that rather than assert
// it by inspection: it catches an allocation anywhere below run_period(),
// including ones the Runner does not make directly (a std::string returned by
// Component::name(), a container grown on first use, a std::function's heap
// state). Only the plain, non-aligned pair is replaced, so the library's
// aligned new/delete stay matched with each other.
//
// This header defines the global `operator new`/`operator delete` overloads,
// so it must be included by exactly one translation unit per test binary --
// including it from two .cpp files linked into the same executable would be a
// duplicate-definition link error.
// ---------------------------------------------------------------------------

#include <cstddef>
#include <cstdlib>
#include <new>

namespace {
std::size_t g_allocations = 0;
bool        g_counting = false;

/// Counts allocations for its lifetime. Not reentrant, not thread safe; these
/// tests are single threaded.
class AllocationCounter {
 public:
  AllocationCounter() {
    g_allocations = 0;
    g_counting = true;
  }
  ~AllocationCounter() { g_counting = false; }
  AllocationCounter(const AllocationCounter&) = delete;
  AllocationCounter& operator=(const AllocationCounter&) = delete;

  [[nodiscard]] std::size_t count() const { return g_allocations; }
};
}  // namespace

void* operator new(std::size_t n) {
  if (g_counting) ++g_allocations;
  void* p = std::malloc(n == 0 ? 1 : n);
  if (p == nullptr) throw std::bad_alloc();
  return p;
}
void* operator new[](std::size_t n) { return ::operator new(n); }
void* operator new(std::size_t n, const std::nothrow_t&) noexcept {
  if (g_counting) ++g_allocations;
  return std::malloc(n == 0 ? 1 : n);
}
void* operator new[](std::size_t n, const std::nothrow_t& t) noexcept {
  return ::operator new(n, t);
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept {
  std::free(p);
}
