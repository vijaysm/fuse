#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include "runner_impl.hpp"
#include "fuse/num/kernels.hpp"
#include "fuse/runtime/fixed_point_residual.hpp"

/// S7.1's Newton-Krylov seam: `Runner::region_fixed_point_residual()` and
/// the `Runner::Impl::evaluate_region_residual()` it wraps.
///
/// WHAT ONE EVALUATION DOES, and why it leaves no trace. Calling `evaluate`
/// at `x` does exactly what one Picard sweep of `run_region()`
/// (src/runtime/iteration.cpp) does to compute `G(y)` for a GIVEN `y` --
/// deliver `x` into every member's own `Requires` storage, re-advance every
/// checkpointed member over the region's own interval, run the region's
/// body instructions so its dataflow produces each block's value -- except
/// that `run_region()`'s own sweep loop learns `G(y)` for the PREVIOUS
/// sweep's delivered value one loop iteration later (the restore/deliver/
/// re-advance at the bottom of sweep k feeds sweep k+1's own body), while a
/// Newton loop needs `G(x)` for an ARBITRARY `x` chosen by a line search or
/// a Krylov iteration, not only the sequence Picard would have chosen
/// itself. This function therefore does not reuse `run_region()`'s loop; it
/// performs the same restore -> deliver -> advance -> execute-body sequence
/// directly, once, for whatever `x` the caller hands it.
///
/// `derive_checkpoint_set()` (src/ir/checkpoint.cpp) already proves that the
/// ONLY state that survives across a sweep is a checkpointed component's own
/// state and the region's loop-carried coupler buffers (`RegionRuntime::
/// carried_arcs`) -- "everything else in the body is recomputed, not
/// saved". Bracketing both of those in save/restore (component
/// save_state()/restore_state(), and a snapshot/restore of the carried
/// arcs' storage) is therefore sufficient, not merely convenient, to leave
/// the Runner's state exactly as this call found it: any later sweep
/// `run_region()` itself performs recomputes every other intermediate
/// unconditionally anyway.
///
/// SCOPE. A window-scoped region's own body can run several times per
/// period and reads/writes state (`window_member_state`,
/// `window_lagged_import_arcs`) this function never touches; a multi-rank
/// region's residual needs the same cross-rank reduction
/// `decide_window_convergence()`/the R-10 participant reduction already
/// gives an ordinary sweep. Lifting either is follow-on work for the JFNK
/// track (S7.2), not this seam -- both refuse with
/// `E-NEWTON-UNSUPPORTED-REGION` rather than compute an answer whose
/// window/parallel semantics have not been worked out here.
namespace fuse {

Result<void> Runner::Impl::check_region_supports_newton(
    ComponentSet& components, std::size_t region_index) const {
  if (region_index >= regions.size()) {
    return Failure{"E-RT-REGION",
                 "region index " + std::to_string(region_index) +
                     " is out of range (" + std::to_string(regions.size()) +
                     " region(s))"};
  }
  const detail::RegionRuntime& R = regions[region_index];

  if (R.window_scoped) {
    return Failure{"E-NEWTON-UNSUPPORTED-REGION",
                 "iteration region '" + R.name +
                     "' is window-scoped; S7.1's fixed-point residual seam "
                     "covers only an ordinary (non-window) region"};
  }
  if (R.participants.size() > 1) {
    return Failure{"E-NEWTON-UNSUPPORTED-REGION",
                 "iteration region '" + R.name +
                     "' has more than one cross-rank participant; S7.1's "
                     "fixed-point residual seam covers only a single-"
                     "participant region"};
  }
  for (const std::size_t t : R.tick_indices) {
    if (components.at(ticks[t].slot).is_externally_driven()) {
      return Failure{"E-NEWTON-UNSUPPORTED-REGION",
                   "iteration region '" + R.name + "' has an externally "
                       "driven member ('" +
                       components.at(ticks[t].slot).name() +
                       "'); S7.1's fixed-point residual seam covers only "
                       "in-process members"};
    }
  }
  return Result<void>::success();
}

Result<void> Runner::Impl::evaluate_region_residual(
    ComponentSet& components, std::size_t region_index,
    std::span<const double> x, std::span<double> f_out, bool commit) {
  if (const Result<void> supported =
          check_region_supports_newton(components, region_index);
      !supported.ok()) {
    return supported;
  }
  detail::RegionRuntime& R = regions[region_index];

  if (x.size() != R.n || f_out.size() != R.n) {
    return Failure{"E-RT-REGION",
                 "iteration region '" + R.name + "' has a stacked interface "
                     "vector of length " + std::to_string(R.n) +
                     "; got in.size()=" + std::to_string(x.size()) +
                     ", out.size()=" + std::to_string(f_out.size())};
  }

  // --- save: component state, the region's own loop-carried buffers, and
  // every block's own feed/value arc --------------------------------------
  //
  // `derive_checkpoint_set()` (src/ir/checkpoint.cpp) proves that a
  // component's own state and the carried arcs are the only state that
  // survives SWEEP TO SWEEP inside an ordinary Picard loop -- but THIS
  // function is not run_region()'s own sweep loop; it delivers an
  // ARBITRARY `x` directly into each block's feed arc (component Requires
  // storage), which the sweep loop itself only ever overwrites with its
  // OWN computed iterate, never a caller-chosen one. Restoring only
  // component state and the carried arcs would leave the feed/value
  // arcs holding THIS call's `x`/`G(x)` instead of whatever a later,
  // ordinary sweep's own pre-loop read (`run_region()`'s own `y_0` seed)
  // expects to find there -- measured directly: omitting this snapshot
  // made FixedPointResidual.EvaluatingFDoesNotPerturbTheRunnersOwnPicard
  // Iteration (test/runtime/fixed_point_residual_test.cpp) fail, the
  // Runner's own Picard iteration landing on a different fixed point
  // after a single out-of-band evaluation than it did fresh.
  for (const std::size_t slot : R.component_slots) {
    const Result<void> s = components.at(slot).save_state();
    if (!s.ok()) return s;
  }
  std::vector<std::vector<double>> carried_snapshot(R.carried_arcs.size());
  for (std::size_t i = 0; i < R.carried_arcs.size(); ++i) {
    const detail::Slot& s = slots[R.carried_arcs[i]];
    if (s.data != nullptr) {
      carried_snapshot[i].assign(s.data, s.data + s.total());
    }
  }
  std::vector<std::vector<double>> feed_snapshot(R.blocks.size());
  std::vector<std::vector<double>> value_snapshot(R.blocks.size());
  for (std::size_t bi = 0; bi < R.blocks.size(); ++bi) {
    const detail::RegionBlock& b = R.blocks[bi];
    const double* feed = slots[b.feed_arc].data;
    if (feed != nullptr) feed_snapshot[bi].assign(feed, feed + b.length);
    const double* value = slots[b.value_arc].data;
    if (value != nullptr) value_snapshot[bi].assign(value, value + b.length);
  }

  // --- deliver x into every block's own feed arc (component Requires) ----
  for (const detail::RegionBlock& b : R.blocks) {
    num::copy(x.data() + b.offset, slots[b.feed_arc].data, b.length);
  }

  // --- advance every checkpointed member over the SAME interval ----------
  // `ctx.iteration` is fixed at 0: `ImplicitComponent::evaluate_residual()`'s
  // own contract (and `Component::advance()`'s, inside an iteration region)
  // requires the result to be a pure function of (restored state, inputs),
  // never of the iteration count, so any one fixed value is correct here --
  // see Runner::Impl::advance() (runner.cpp) for the StepContext this
  // mirrors.
  Result<void> adv_err = Result<void>::success();
  for (const std::size_t t : R.tick_indices) {
    const detail::Tick& tick = ticks[t];
    StepContext ctx;
    ctx.t_begin = Rational{period_index} * plan.period() + tick.time;
    ctx.t_end = ctx.t_begin + tick.dt;
    ctx.step = period_index * tick.per_period + tick.local_step;
    ctx.iteration = 0;
    if (!ctx.valid()) {
      adv_err = Failure{"E-RT-STEP",
                       "step context [" + ctx.t_begin.to_string() + ", " +
                           ctx.t_end.to_string() + ") for component '" +
                           components.at(tick.slot).name() + "' is not valid"};
      break;
    }
    adv_err = components.at(tick.slot).advance(ctx);
    if (!adv_err.ok()) break;
  }

  // --- execute the region's own body to propagate the advance into G(x) --
  Result<void> body_err = Result<void>::success();
  if (adv_err.ok()) {
    const std::vector<Instruction>& stream = plan.instructions();
    for (std::size_t i = R.enter; i <= R.exit; ++i) {
      body_err = execute(stream[i], resolved[i]);
      if (!body_err.ok()) break;
    }
  }

  // --- F(x) = x - G(x), read before any restore undoes it ----------------
  if (adv_err.ok() && body_err.ok()) {
    for (const detail::RegionBlock& b : R.blocks) {
      const double* g = slots[b.value_arc].data;
      for (std::size_t j = 0; j < b.length; ++j) {
        f_out[b.offset + j] = x[b.offset + j] - g[j];
      }
    }
  }

  // --- restore: undo the advance, the carried buffers, and every block's
  // own feed/value arc -----------------------------------------------------
  // Every component is restored regardless of an earlier failure, so a
  // partial advance never leaves the Runner's state straddling two points
  // -- the same "restore everything this call touched" rule run_region()'s
  // own rollback step follows.
  //
  // `commit`'s one exception: on the SUCCESS path (adv_err/body_err both
  // ok), skip the restore entirely, leaving the delivered x/advanced
  // state/executed body as this call's real, persisted effect --
  // run_region_newton()'s own "verification sweep" (src/runtime/
  // iteration.cpp) is the one caller that passes `commit = true`, exactly
  // once, at the Newton solve's own converged iterate. A FAILURE still
  // restores regardless of `commit`: a partial advance is never something
  // a caller asked to keep.
  const bool skip_restore = commit && adv_err.ok() && body_err.ok();
  Result<void> restore_err = Result<void>::success();
  if (!skip_restore) {
    for (const std::size_t slot : R.component_slots) {
      const Result<void> rs = components.at(slot).restore_state();
      if (!rs.ok() && restore_err.ok()) restore_err = rs;
    }
    for (std::size_t i = 0; i < R.carried_arcs.size(); ++i) {
      detail::Slot& s = slots[R.carried_arcs[i]];
      if (s.data != nullptr && !carried_snapshot[i].empty()) {
        num::copy(carried_snapshot[i].data(), s.data, carried_snapshot[i].size());
      }
    }
  }
  if (!skip_restore) {
    for (std::size_t bi = 0; bi < R.blocks.size(); ++bi) {
      const detail::RegionBlock& b = R.blocks[bi];
      if (!feed_snapshot[bi].empty()) {
        num::copy(feed_snapshot[bi].data(), slots[b.feed_arc].data, b.length);
      }
      if (!value_snapshot[bi].empty()) {
        num::copy(value_snapshot[bi].data(), slots[b.value_arc].data, b.length);
      }
    }
  }

  if (!adv_err.ok()) return adv_err;
  if (!body_err.ok()) return body_err;
  return restore_err;
}

Result<FixedPointResidual> Runner::region_fixed_point_residual(
    std::size_t region_index, ComponentSet& components) {
  if (!impl_->initialized) {
    return Failure{"E-RT-REGION",
                 "region_fixed_point_residual() called before a successful "
                 "initialize()"};
  }
  if (region_index >= impl_->regions.size()) {
    return Failure{"E-RT-REGION",
                 "region index " + std::to_string(region_index) +
                     " is out of range (" +
                     std::to_string(impl_->regions.size()) + " region(s))"};
  }
  const detail::RegionRuntime& R = impl_->regions[region_index];

  FixedPointResidual out;
  out.n = R.n;
  out.offsets.reserve(R.blocks.size());
  out.lengths.reserve(R.blocks.size());
  out.channel_names.reserve(R.blocks.size());
  for (const detail::RegionBlock& b : R.blocks) {
    out.offsets.push_back(b.offset);
    out.lengths.push_back(b.length);
    out.channel_names.push_back(b.channel_name);
  }

  Impl* impl = impl_.get();
  ComponentSet* comps = &components;
  out.evaluate = [impl, region_index, comps](std::span<const double> in,
                                             std::span<double> f_out) {
    return impl->evaluate_region_residual(*comps, region_index, in, f_out);
  };
  return out;
}

}  // namespace fuse
