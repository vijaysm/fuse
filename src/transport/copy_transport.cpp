#include "fuse/base/config.hpp"
#include "fuse/transport/transport.hpp"

#include <algorithm>
#include <cstring>
#include <string>
#include <utility>

namespace fuse {

// Deliberately not `spec_ = spec`. The runner calls prepare() inside the hot
// loop, and TransportSpec now carries three std::vectors: copying the whole
// struct would allocate on every step as soon as any of them is non-empty,
// which is invisible in a build where they never are.
Result<void> CopyTransport::prepare(const TransportSpec& spec) {
  elements_ = spec.elements;
  components_ = spec.components;
  return Result<void>::success();
}

// The size guard below is deliberately a returned Error, not an assert: this
// header is compiled with -O3 -DNDEBUG in release, where an assert vanishes.
Result<void> CopyTransport::move(const double* src, double* dst,
                                 std::size_t n) {
  if (src == nullptr || dst == nullptr) {
    return Failure{"E-TRANSPORT-NULL", "copy transport given a null pointer"};
  }
  const std::size_t capacity =
      elements_ * static_cast<std::size_t>(std::max(components_, 1));
  if (n > capacity) {
    return Failure{"E-TRANSPORT-SIZE",
                 "move of " + std::to_string(n) + " elements exceeds the " +
                     std::to_string(capacity) + " prepared"};
  }
  if (src == dst) return Result<void>::success();  // legitimate no-op
  std::memcpy(dst, src, n * sizeof(double));
  return Result<void>::success();
}

void TransportRegistry::register_backend(std::string name, Factory factory) {
  for (auto& [n, f] : factories_) {
    if (n == name) {
      f = std::move(factory);
      return;
    }
  }
  factories_.emplace_back(std::move(name), std::move(factory));
}

Result<std::unique_ptr<Transport>> TransportRegistry::create(
    const std::string& name) const {
  for (const auto& [n, f] : factories_) {
    if (n == name) return f();
  }
  return Failure{"E-TRANSPORT-BACKEND",
               "no transport backend named '" + name + "'"};
}

std::vector<std::string> TransportRegistry::available() const {
  std::vector<std::string> names;
  names.reserve(factories_.size());
  for (const auto& entry : factories_) names.push_back(entry.first);
  return names;
}

TransportRegistry& TransportRegistry::instance() {
  // Intentionally leaked: backends may be registered from other translation
  // units' static initialisers, and the registry must outlive every consumer
  // regardless of static destruction order.
  static TransportRegistry* reg = [] {
    auto* r = new TransportRegistry;
    r->register_backend("copy",
                        [] { return std::make_unique<CopyTransport>(); });
#if FUSE_HAVE_MPI
    // Registered from HERE, not from a static initialiser in
    // src/transport/mpi_transport.cpp. soma_core is a static archive: an
    // object file nothing references is not pulled in, and its static
    // initialisers never run, so a self-registering backend would be present
    // or absent depending on what else the executable happened to link. The
    // reference to MpiTransport's out-of-line constructor and destructor below
    // is what makes the linker keep the translation unit.
    r->register_backend("mpi", [] { return std::make_unique<MpiTransport>(); });
#endif
    // Registered UNCONDITIONALLY, and from here for the same linker reason.
    // LowFiveTransport exists in every configuration -- on a build without
    // LowFive its setup calls return E-TRANSPORT-NOLOWFIVE naming the CMake
    // option, which is a far more useful diagnostic than "no transport
    // backend named 'lowfive'" from a registry that silently lost the entry.
    // The reference to the out-of-line constructor below is what makes the
    // linker keep src/transport/lowfive_transport.cpp in the static archive.
    r->register_backend("lowfive",
                        [] { return std::make_unique<LowFiveTransport>(); });
    return r;
  }();
  return *reg;
}

}  // namespace fuse
