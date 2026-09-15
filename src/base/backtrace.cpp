#include "fuse/base/backtrace.hpp"

#include <cxxabi.h>
#include <dlfcn.h>
#include <execinfo.h>

#include <cstdlib>
#include <memory>

#include <fmt/format.h>

namespace fuse {

Backtrace Backtrace::capture(int skip) noexcept {
  Backtrace bt;
  // ::backtrace writes return addresses; nothing is allocated (I5).
  const int n = ::backtrace(bt.frames_.data(), static_cast<int>(kMaxFrames));
  const int drop = skip + 1;  // + this function's own frame
  if (n > drop) {
    for (int i = drop; i < n; ++i) bt.frames_[static_cast<std::size_t>(i - drop)] = bt.frames_[static_cast<std::size_t>(i)];
    bt.count_ = static_cast<std::size_t>(n - drop);
  }
  return bt;
}

namespace {
std::string demangle(const char* mangled) {
  int status = 0;
  std::unique_ptr<char, void (*)(void*)> out(abi::__cxa_demangle(mangled, nullptr, nullptr, &status),
                                             std::free);
  return (status == 0 && out) ? std::string(out.get()) : std::string(mangled);
}
}  // namespace

std::string Backtrace::format() const {
  std::string out;
  for (std::size_t i = 0; i < count_; ++i) {
    void* addr = frames_[i];
    Dl_info info{};
    if (::dladdr(addr, &info) != 0 && info.dli_sname != nullptr) {
      const auto offset = static_cast<const char*>(addr) - static_cast<const char*>(info.dli_saddr);
      fmt::format_to(std::back_inserter(out), "  #{:<2} {:p}  {} + {}\n", i, addr,
                     demangle(info.dli_sname), offset);
    } else if (info.dli_fname != nullptr) {
      fmt::format_to(std::back_inserter(out), "  #{:<2} {:p}  in {}\n", i, addr, info.dli_fname);
    } else {
      fmt::format_to(std::back_inserter(out), "  #{:<2} {:p}\n", i, addr);
    }
  }
  return out;
}

}  // namespace fuse
