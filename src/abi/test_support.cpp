#include "test_support.hpp"

#include "context.hpp"

namespace fuse::abi_test {

bool inject_component(fuse_ctx* ctx, const std::string& component_name,
                      std::unique_ptr<fuse::Component> c) {
  if (ctx == nullptr || c == nullptr) return false;
  if (ctx->state != fuse_ctx::State::Loaded) return false;
  if (!ctx->graph.find_component(component_name).has_value()) return false;
  if (ctx->injected.contains(component_name)) return false;
  ctx->injected.emplace(component_name, std::move(c));
  return true;
}

}  // namespace fuse::abi_test
