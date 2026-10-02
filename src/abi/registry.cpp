#include "registry.hpp"

#include <dlfcn.h>

#include "provider_adapter.hpp"

namespace fuse::abi_detail {

Registry::~Registry() {
  for (void* h : handles_) {
    if (h != nullptr) dlclose(h);
  }
}

namespace {
[[nodiscard]] bool already_mounted(
    const std::vector<std::pair<std::string, std::unique_ptr<fuse::Component>>>& mounted,
    const std::string& name) {
  for (const auto& [n, c] : mounted) {
    if (n == name) return true;
  }
  return false;
}
}  // namespace

fuse::Result<void> Registry::mount_component(std::string name,
                                             std::unique_ptr<fuse::Component> c) {
  if (c == nullptr) {
    return fuse::Failure{"E-ABI-REGISTRY-NULL", "mount_component: null component"};
  }
  if (already_mounted(mounted_, name)) {
    return fuse::Failure{"E-ABI-REGISTRY-DUPLICATE",
                         "a provider named '" + name + "' is already mounted"};
  }
  mounted_.emplace_back(std::move(name), std::move(c));
  return fuse::Result<void>::success();
}

fuse::Result<void> Registry::mount_table(std::string name,
                                         const fuse_provider_abi_v1* table) {
  if (table == nullptr) {
    return fuse::Failure{"E-ABI-REGISTRY-NULL",
                         "mount_table: null table for provider '" + name + "'"};
  }
  if (table->struct_size < sizeof(fuse_provider_abi_v1) || table->abi_version != FUSE_ABI_VERSION) {
    return fuse::Failure{"E-ABI-VERSION",
                         "provider '" + name + "': table struct_size/abi_version (" +
                             std::to_string(table->struct_size) + "/" +
                             std::to_string(table->abi_version) +
                             ") does not speak FUSE_ABI_VERSION (" +
                             std::to_string(FUSE_ABI_VERSION) + ")"};
  }
  if (already_mounted(mounted_, name)) {
    return fuse::Failure{"E-ABI-REGISTRY-DUPLICATE",
                         "a provider named '" + name + "' is already mounted"};
  }
  mounted_.emplace_back(name, std::make_unique<ProviderAdapter>(name, table));
  return fuse::Result<void>::success();
}

fuse::Result<void> Registry::mount_dynamic(std::string name,
                                           const std::string& shared_object_path) {
  // RTLD_NOW: a plugin with an unresolved symbol fails HERE, naming the
  // real cause, rather than crashing later at the first call through a
  // half-resolved table. RTLD_LOCAL: two plugins may each define
  // process-local statics (e.g. their own g_table) with no collision --
  // the whole reason dlopen(), not static linking, is this mechanism.
  dlerror();  // clear any stale error before this call (dlerror's own contract)
  void* handle = dlopen(shared_object_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    const char* msg = dlerror();
    return fuse::Failure{"E-ABI-LOAD",
                         "provider '" + name + "': dlopen('" + shared_object_path +
                             "') failed: " + (msg != nullptr ? msg : "unknown dlopen failure")};
  }
  dlerror();
  void* sym = dlsym(handle, "fuse_provider_abi");
  if (sym == nullptr) {
    const char* msg = dlerror();
    dlclose(handle);
    return fuse::Failure{"E-ABI-LOAD",
                         "provider '" + name + "': dlsym('fuse_provider_abi') on '" +
                             shared_object_path +
                             "' failed: " + (msg != nullptr ? msg : "symbol not found")};
  }
  // POSIX dlsym() returns void*; a function pointer and a data pointer are
  // not guaranteed interconvertible by the C++ standard, but this cast is
  // the documented way every dlopen()-based plugin loader in this codebase
  // and its parents resolves a bootstrap symbol (participant.h's own "one
  // dlsym" is the identical shape on the other direction).
  using BootstrapFn = const fuse_provider_abi_v1* (*)(std::uint32_t);
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
  const auto fn = reinterpret_cast<BootstrapFn>(sym);
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
  const fuse_provider_abi_v1* table = fn(FUSE_ABI_VERSION);
  if (table == nullptr) {
    dlclose(handle);
    return fuse::Failure{"E-ABI-VERSION",
                         "provider '" + name + "' (" + shared_object_path +
                             "): fuse_provider_abi(FUSE_ABI_VERSION) returned NULL -- the "
                             "plugin does not speak this host's ABI version"};
  }
  fuse::Result<void> mounted = mount_table(name, table);
  if (!mounted.ok()) {
    dlclose(handle);
    return mounted;
  }
  handles_.push_back(handle);
  return fuse::Result<void>::success();
}

fuse::Result<void> Registry::install(fuse::ComponentSet& set) {
  for (auto& [name, c] : mounted_) {
    try {
      set.add(std::move(c));
    } catch (const std::exception& e) {
      return fuse::Failure{"E-ABI-REGISTRY-INSTALL", e.what()};
    }
  }
  mounted_.clear();
  return fuse::Result<void>::success();
}

fuse::Result<std::unique_ptr<fuse::Component>> Registry::extract(const std::string& name) {
  for (auto it = mounted_.begin(); it != mounted_.end(); ++it) {
    if (it->first == name) {
      std::unique_ptr<fuse::Component> c = std::move(it->second);
      mounted_.erase(it);
      return c;
    }
  }
  return fuse::Failure{"E-ABI-REGISTRY-NOTFOUND",
                       "no provider named '" + name + "' is mounted"};
}

}  // namespace fuse::abi_detail
