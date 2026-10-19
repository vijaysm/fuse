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

bool force_this_process(fuse_ctx* ctx, int rank) {
  if (ctx == nullptr) return false;
  if (ctx->state != fuse_ctx::State::Loaded) return false;
  ctx->test_force_this_process = rank;
  return true;
}

// KNOWN_ISSUES.md D57 ("Collectives" track): see this function's own
// declaration (test_support.hpp) for the full account of what this sets
// and why it is safe to do with no MPI runtime at all.
bool force_communicator_set(fuse_ctx* ctx, std::int64_t value) {
  if (ctx == nullptr) return false;
  // Callable in EITHER the Loaded or the Initialized state (unlike
  // force_this_process(), which fuse_initialize() itself consumes and so
  // must run before): fuse_initialize() reads `c.communicator_set` once,
  // to pick "mpi" over "copy" as the Runner's own transport AND to set
  // `RunnerConfig::base_communicator` -- setting it before initialize()
  // on a build with no "mpi" transport backend registered (tier 0,
  // FUSE_HAVE_MPI off) makes initialize() itself fail, which is not what
  // KNOWN_ISSUES.md D57's own test needs: it is testing prime_next_
  // request()'s own gate in ISOLATION, a plain `fuse_ctx::
  // communicator_set` read with no other consumer at run time, so
  // setting it AFTER a successful initialize() (transport already
  // "copy", already baked into the live Runner, genuinely unaffected by
  // this call) reaches exactly that gate and nothing upstream of it.
  if (ctx->state != fuse_ctx::State::Loaded &&
      ctx->state != fuse_ctx::State::Initialized) {
    return false;
  }
  ctx->communicator_set = true;
  ctx->communicator = value;
  return true;
}

fuse::RunStats run_stats(const fuse_ctx* ctx) {
  if (ctx == nullptr || ctx->state != fuse_ctx::State::Initialized ||
      !ctx->runner.has_value()) {
    return fuse::RunStats{};
  }
  return ctx->runner->stats();
}

fuse::Runner* runner_of(fuse_ctx* ctx) {
  if (ctx == nullptr) return nullptr;
  if (ctx->state != fuse_ctx::State::Initialized) return nullptr;
  if (!ctx->runner.has_value()) return nullptr;
  return &*ctx->runner;
}

}  // namespace fuse::abi_test
