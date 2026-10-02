#pragma once

/// Mounts providers -- of any of the three kinds docs/dev/abi-design.md's
/// "Provider direction" names -- into a `fuse::ComponentSet`, so the four
/// conformance-test kinds (test/abi/conformance_test.cpp) differ only in
/// HOW a `Registry` is told about a component, never in how the Runner
/// drives it afterwards:
///
///   - a native `fuse::Component`, mounted directly: no ABI boundary at
///     all, `mount_component()`.
///   - a `fuse_provider_abi_v1*` table already in memory -- built by a C++
///     test, or already resolved from a dlopen'd object -- wrapped in a
///     `fuse::abi_detail::ProviderAdapter`, `mount_table()`.
///   - a shared object exporting exactly one symbol, `fuse_provider_abi`
///     (include/fuse/abi/provider.h, point 4 of docs/dev/abi-design.md): a
///     plain C plugin and a Fortran plugin built through `ffuse_provider`
///     export the IDENTICAL C symbol, so one mechanism, `dlopen()` +
///     `dlsym()`, covers both -- `mount_dynamic()`.
///
/// Never substitutes and never guesses: a name already mounted is refused
/// (E-ABI-REGISTRY-DUPLICATE), a `dlopen()` failure is refused with
/// `dlerror()`'s own text (E-ABI-LOAD), and a plugin whose table does not
/// speak `FUSE_ABI_VERSION` -- absent, too small, or a different
/// `abi_version` -- is refused (E-ABI-VERSION) rather than run against
/// whatever subset happened to be there.

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "fuse/abi/provider.h"
#include "fuse/base/result.hpp"
#include "fuse/runtime/component.hpp"

namespace fuse::abi_detail {

class Registry {
 public:
  Registry() = default;
  ~Registry();
  Registry(const Registry&) = delete;
  Registry& operator=(const Registry&) = delete;

  [[nodiscard]] fuse::Result<void> mount_component(std::string name,
                                                    std::unique_ptr<fuse::Component> c);
  [[nodiscard]] fuse::Result<void> mount_table(std::string name,
                                               const fuse_provider_abi_v1* table);
  [[nodiscard]] fuse::Result<void> mount_dynamic(std::string name,
                                                 const std::string& shared_object_path);

  [[nodiscard]] std::size_t mounted_count() const noexcept { return mounted_.size(); }

  /// Moves every mounted component into `set`, in mount order. Consumes
  /// this registry's own component list; dlopen handles are kept open
  /// (and closed at this object's destructor) because every ProviderAdapter
  /// built from a dynamic mount keeps calling through `table`, which is
  /// static storage inside the still-open shared object.
  [[nodiscard]] fuse::Result<void> install(fuse::ComponentSet& set);

  /// Moves the single component named `name` out, for a caller that has its
  /// own placement mechanism (test/abi/test_support.hpp's inject_component,
  /// used by test/abi/conformance_test.cpp to put a mounted provider under
  /// the SAME seam the native and participant-direction kinds go through).
  /// Refuses `E-ABI-REGISTRY-NOTFOUND` when no such name was mounted.
  [[nodiscard]] fuse::Result<std::unique_ptr<fuse::Component>> extract(const std::string& name);

 private:
  std::vector<std::pair<std::string, std::unique_ptr<fuse::Component>>> mounted_;
  std::vector<void*>                                                     handles_;
};

}  // namespace fuse::abi_detail
