#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include "runner_impl.hpp"
#include "fuse/num/kernels.hpp"
#include "fuse/runtime/fixed_point_residual.hpp"

/// The Newton-Krylov seam: `Runner::region_fixed_point_residual()` and
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
/// SCOPE. This seam covers the ordinary (non-window) sweep. A window-scoped
/// region's own body can run several times per period and reads/writes state
/// (`window_member_state`, `window_lagged_import_arcs`) this function never
/// touches; `evaluate_window_region_residual()` (src/runtime/iteration.cpp)
/// handles that shape instead. A multi-rank region's residual needs the same
/// cross-rank participant reduction an ordinary sweep already gets
/// (`resolve_regions()` builds it). An externally driven member is refused
/// with `E-NEWTON-UNSUPPORTED-REGION` rather than compute an answer whose
/// semantics have not been worked out here.
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

  // There is no blanket "window-scoped is always refused" rule here --
  // `evaluate_window_region_residual()` (src/runtime/iteration.cpp) is the
  // window-scoped sibling this check ALSO gates, sharing the two checks below
  // with the ordinary seam (`evaluate_region_residual()`'s own
  // ordinary-seam-only refusal, immediately below in that function, is where
  // "window-scoped" is refused for THAT function specifically -- the ordinary
  // seam's own ADVANCE/body machinery is genuinely wrong for a window region,
  // never lifted). A window-scoped region's own
  // `R.participants`/`R.tick_indices` are populated the identical way an
  // ordinary region's are (`resolve_regions()`'s own single derivation, shared
  // by both shapes), so the two checks below need no window-specific branch of
  // their own.
  //
  // There is no blanket "more than one participant is refused" rule, and
  // nothing replaces it with a narrower refusal -- `R.participants`
  // (resolve_regions(), iteration.cpp) is populated with more than the empty
  // set ONLY when `config.base_communicator` has a value in the first place
  // (that construction's own comment has the invariant: with no communicator
  // configured, a region's components on different nominal ranks are run
  // "purely local," on the one process that exists, and `R.participants` stays
  // empty). `R.participants.size() > 1` THEREFORE ALREADY IMPLIES a real
  // communicator is configured -- exactly what `resolve_regions()` needs to
  // build a real `ParticipantSetReducer` (`RegionRuntime::newton_reducer`'s own
  // comment, runner_impl.hpp, has the full account) -- so there is no reachable
  // "more than one participant, no communicator to agree across them" case left
  // for this function to catch: the Reducer a multi-rank Newton region needs is
  // unconditionally buildable whenever this function would otherwise have
  // refused for that reason alone.
  for (const std::size_t t : R.tick_indices) {
    if (components.at(ticks[t].slot).is_externally_driven()) {
      return Failure{"E-NEWTON-UNSUPPORTED-REGION",
                   "iteration region '" + R.name + "' has an externally "
                       "driven member ('" +
                       components.at(ticks[t].slot).name() +
                       "'); the fixed-point residual seam covers only "
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

  // This function's own body (below) is the ORDINARY seam -- it advances
  // `R.tick_indices` directly, once, over `R.enter..R.exit`, with no notion of
  // a window's own multiple substeps, lagged-import predictor, or straddle
  // bookkeeping. A window-scoped region needs
  // `evaluate_window_region_residual()` (this file's own sibling,
  // src/runtime/iteration.cpp) instead -- refused HERE, specifically, rather
  // than folded into `check_region_supports_newton()` above (which now accepts
  // a window-scoped region, since BOTH functions share its two general checks;
  // only this function's own machinery is ordinary-only).
  if (R.window_scoped) {
    return Failure{"E-NEWTON-UNSUPPORTED-REGION",
                 "iteration region '" + R.name +
                     "' is window-scoped; evaluate_region_residual() is the "
                     "ordinary (non-window) seam -- evaluate_window_region_"
                     "residual() is the window-scoped one"};
  }

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
    if (s.data != nullptr) slot_to_vector(s, carried_snapshot[i], s.total());
  }
  std::vector<std::vector<double>> feed_snapshot(R.blocks.size());
  std::vector<std::vector<double>> value_snapshot(R.blocks.size());
  for (std::size_t bi = 0; bi < R.blocks.size(); ++bi) {
    const detail::RegionBlock& b = R.blocks[bi];
    if (slots[b.feed_arc].data != nullptr) {
      slot_to_vector(slots[b.feed_arc], feed_snapshot[bi], b.length);
    }
    if (slots[b.value_arc].data != nullptr) {
      slot_to_vector(slots[b.value_arc], value_snapshot[bi], b.length);
    }
  }

  // --- deliver x into every block's own feed arc (component Requires) ----
  for (const detail::RegionBlock& b : R.blocks) {
    host_to_slot(x.data() + b.offset, slots[b.feed_arc], b.length);
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
    note_component_ran();
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
      const double* g = slot_host_read(slots[b.value_arc], b.length, block_host_scratch);
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
      note_component_ran();
      if (!rs.ok() && restore_err.ok()) restore_err = rs;
    }
    for (std::size_t i = 0; i < R.carried_arcs.size(); ++i) {
      detail::Slot& s = slots[R.carried_arcs[i]];
      if (s.data != nullptr && !carried_snapshot[i].empty()) {
        host_to_slot(carried_snapshot[i].data(), s, carried_snapshot[i].size());
      }
    }
  }
  if (!skip_restore) {
    for (std::size_t bi = 0; bi < R.blocks.size(); ++bi) {
      const detail::RegionBlock& b = R.blocks[bi];
      if (!feed_snapshot[bi].empty()) {
        host_to_slot(feed_snapshot[bi].data(), slots[b.feed_arc], b.length);
      }
      if (!value_snapshot[bi].empty()) {
        host_to_slot(value_snapshot[bi].data(), slots[b.value_arc], b.length);
      }
    }
  }

  if (!adv_err.ok()) return adv_err;
  if (!body_err.ok()) return body_err;
  return restore_err;
}

/// `NewtonResidualPath::Direct`'s own `F(x) = x - G(x)` for an ordinary
/// (non-window) region. Structurally a close twin of
/// `evaluate_region_residual()` immediately above -- same save/restore bracket
/// for the region's own loop-carried buffers and every block's own feed/value
/// arc -- with two differences: (1) no `save_state()`/`restore_state()` call on
/// any component at all (`Implicit Component::evaluate_residual()`'s own
/// contract already guarantees it leaves the component bit-identical to how it
/// found it, so there is nothing to roll back there), and (2) `G(x)` is
/// obtained by calling each providing member's own `evaluate_residual()`
/// directly, writing its `residual_out` into the registered Source-node storage
/// `RegionRuntime::NewtonDirectMember::provides_source_arcs` names -- the SAME
/// storage `advance()` would have written -- so the region's existing body
/// instructions (R.enter..R.exit: Transport/Remap/Constrain/Sink) carry it
/// downstream exactly as they already carry `advance()`'s own output. That
/// write is the one EXTRA state this function must snapshot/restore that the
/// Advance seam does not: Advance's own write there is undone by the
/// component's own `restore_state()` instead.
Result<void> Runner::Impl::evaluate_region_residual_direct(
    ComponentSet& components, std::size_t region_index,
    std::span<const double> x, std::span<double> f_out, bool commit) {
  if (const Result<void> supported =
          check_region_supports_newton(components, region_index);
      !supported.ok()) {
    return supported;
  }
  detail::RegionRuntime& R = regions[region_index];

  if (R.window_scoped) {
    return Failure{"E-NEWTON-UNSUPPORTED-REGION",
                 "iteration region '" + R.name +
                     "' is window-scoped; evaluate_region_residual_direct() "
                     "is the ordinary (non-window) Direct seam"};
  }
  if (x.size() != R.n || f_out.size() != R.n) {
    return Failure{"E-RT-REGION",
                 "iteration region '" + R.name + "' has a stacked interface "
                     "vector of length " + std::to_string(R.n) +
                     "; got in.size()=" + std::to_string(x.size()) +
                     ", out.size()=" + std::to_string(f_out.size())};
  }

  // --- save: the region's own loop-carried buffers, every block's own
  // feed/value arc (IDENTICAL to evaluate_region_residual()'s own bracket),
  // and every Direct member's own Provides-field Source storage, which
  // THIS function pokes directly, standing in for what advance() would
  // have written -----------------------------------------------------
  std::vector<std::vector<double>> carried_snapshot(R.carried_arcs.size());
  for (std::size_t i = 0; i < R.carried_arcs.size(); ++i) {
    const detail::Slot& s = slots[R.carried_arcs[i]];
    // docs/dev/device-resident-arcs.md step 8: `s.data, s.data + s.total()`
    // is a raw pointer range over the arc's own storage -- legal only when
    // that storage is host memory. slot_to_vector() already carries the
    // same "assign the whole slot, or resize and slot_to_host() when it is
    // device-resident" fallback this iterator-pair constructor needs.
    if (s.data != nullptr) {
      slot_to_vector(s, carried_snapshot[i], s.total());
    }
  }
  std::vector<std::vector<double>> feed_snapshot(R.blocks.size());
  std::vector<std::vector<double>> value_snapshot(R.blocks.size());
  for (std::size_t bi = 0; bi < R.blocks.size(); ++bi) {
    const detail::RegionBlock& b = R.blocks[bi];
    const detail::Slot& feed_slot = slots[b.feed_arc];
    if (feed_slot.data != nullptr) slot_to_vector(feed_slot, feed_snapshot[bi], b.length);
    const detail::Slot& value_slot = slots[b.value_arc];
    if (value_slot.data != nullptr) slot_to_vector(value_slot, value_snapshot[bi], b.length);
  }
  std::vector<std::vector<double>> provides_snapshot(R.newton_direct_members.size());
  for (std::size_t mi = 0; mi < R.newton_direct_members.size(); ++mi) {
    const detail::RegionRuntime::NewtonDirectMember& m = R.newton_direct_members[mi];
    std::size_t total = 0;
    for (const std::size_t bi : m.provides_blocks) total += R.blocks[bi].length;
    provides_snapshot[mi].resize(total);
    std::size_t off = 0;
    for (std::size_t k = 0; k < m.provides_blocks.size(); ++k) {
      const detail::Slot& src_slot = slots[m.provides_source_arcs[k]];
      const std::size_t len = R.blocks[m.provides_blocks[k]].length;
      // docs/dev/device-resident-arcs.md step 8: this Newton-Direct
      // member's own Provides arc is reached by place_arcs_on_device()'s
      // ordinary rule (step 3), same as every other region block, so it
      // may be device-resident -- slot_to_host() instead of a raw
      // num::copy() off `Slot::data`.
      if (src_slot.data != nullptr) {
        slot_to_host(src_slot, provides_snapshot[mi].data() + off, len);
      }
      off += len;
    }
  }

  // --- deliver x into every block's own feed arc (component Requires) ----
  for (const detail::RegionBlock& b : R.blocks) {
    host_to_slot(x.data() + b.offset, slots[b.feed_arc], b.length);
  }

  // --- G(x): evaluate_residual() on every Direct member, in place of
  // advance() -- no save_state()/restore_state() anywhere in this loop ---
  Result<void> eval_err = Result<void>::success();
  for (const detail::RegionRuntime::NewtonDirectMember& m : R.newton_direct_members) {
    ImplicitComponent* ic = components.at(m.component_slot).implicit();
    if (ic == nullptr) {
      eval_err = Failure{
          "E-NEWTON-CAPABILITY",
          "component '" + components.at(m.component_slot).name() +
              "' has no ImplicitComponent despite being a "
              "NewtonResidualPath::Direct member of iteration region '" +
              R.name + "'"};
      break;
    }

    std::size_t in_total = 0;
    for (const std::size_t bi : m.requires_blocks) in_total += R.blocks[bi].length;
    std::vector<double> interface_in(in_total, 0.0);
    {
      std::size_t off = 0;
      for (const std::size_t bi : m.requires_blocks) {
        const detail::RegionBlock& b = R.blocks[bi];
        slot_to_host(slots[b.feed_arc], interface_in.data() + off, b.length);
        off += b.length;
      }
    }

    std::size_t out_total = 0;
    for (const std::size_t bi : m.provides_blocks) out_total += R.blocks[bi].length;
    std::vector<double> residual_out(out_total, 0.0);

    // `ctx.iteration` fixed at 0 -- the identical reasoning evaluate_
    // region_residual()'s own Advance loop states just above, and
    // `tick_global_index` is resolved to the SINGLE tick this member has
    // inside the region (resolve_regions()'s own build_newton_direct_
    // members(), which refuses a sub-cycled member outright).
    const detail::Tick& tick = ticks[m.tick_global_index];
    StepContext ctx;
    ctx.t_begin = Rational{period_index} * plan.period() + tick.time;
    ctx.t_end = ctx.t_begin + tick.dt;
    ctx.step = period_index * tick.per_period + tick.local_step;
    ctx.iteration = 0;
    if (!ctx.valid()) {
      eval_err = Failure{"E-RT-STEP",
                        "step context [" + ctx.t_begin.to_string() + ", " +
                            ctx.t_end.to_string() + ") for component '" +
                            components.at(m.component_slot).name() +
                            "' is not valid"};
      break;
    }
    eval_err = ic->evaluate_residual(
        ctx, std::span<const double>(interface_in.data(), interface_in.size()),
        std::span<double>(residual_out.data(), residual_out.size()));
    if (!eval_err.ok()) break;

    std::size_t off = 0;
    for (std::size_t k = 0; k < m.provides_blocks.size(); ++k) {
      const detail::Slot& dst_slot = slots[m.provides_source_arcs[k]];
      const std::size_t len = R.blocks[m.provides_blocks[k]].length;
      if (dst_slot.data != nullptr) host_to_slot(residual_out.data() + off, dst_slot, len);
      off += len;
    }
  }

  // --- execute the region's own body to propagate the write into G(x) ----
  Result<void> body_err = Result<void>::success();
  if (eval_err.ok()) {
    const std::vector<Instruction>& stream = plan.instructions();
    for (std::size_t i = R.enter; i <= R.exit; ++i) {
      body_err = execute(stream[i], resolved[i]);
      if (!body_err.ok()) break;
    }
  }

  // --- F(x) = x - G(x), read before any restore undoes it ----------------
  if (eval_err.ok() && body_err.ok()) {
    for (const detail::RegionBlock& b : R.blocks) {
      // docs/dev/device-resident-arcs.md step 8: `b.value_arc` is an
      // ordinary region block arc (step 3's own rule), so it may be
      // device-resident -- slot_host_read() instead of dereferencing
      // `Slot::data` element by element.
      const double* g = slot_host_read(slots[b.value_arc], b.length, block_host_scratch);
      for (std::size_t j = 0; j < b.length; ++j) {
        f_out[b.offset + j] = x[b.offset + j] - g[j];
      }
    }
  }

  // --- restore: the carried buffers, every block's own feed/value arc,
  // and every Direct member's own Provides-field Source storage -- never
  // a component save_state()/restore_state() call, which Direct never
  // makes in the first place. `commit`'s meaning mirrors evaluate_
  // region_residual()'s own: skip the restore on the success path when
  // true. `run_region_newton()` never passes `true` here -- see that
  // function's own comment -- but the parameter is kept for signature
  // symmetry and so a future/diagnostic caller can use it the same way.
  const bool skip_restore = commit && eval_err.ok() && body_err.ok();
  if (!skip_restore) {
    for (std::size_t i = 0; i < R.carried_arcs.size(); ++i) {
      const detail::Slot& s = slots[R.carried_arcs[i]];
      if (s.data != nullptr && !carried_snapshot[i].empty()) {
        host_to_slot(carried_snapshot[i].data(), s, carried_snapshot[i].size());
      }
    }
    for (std::size_t bi = 0; bi < R.blocks.size(); ++bi) {
      const detail::RegionBlock& b = R.blocks[bi];
      if (!feed_snapshot[bi].empty()) {
        host_to_slot(feed_snapshot[bi].data(), slots[b.feed_arc], b.length);
      }
      if (!value_snapshot[bi].empty()) {
        host_to_slot(value_snapshot[bi].data(), slots[b.value_arc], b.length);
      }
    }
    for (std::size_t mi = 0; mi < R.newton_direct_members.size(); ++mi) {
      const detail::RegionRuntime::NewtonDirectMember& m = R.newton_direct_members[mi];
      std::size_t off = 0;
      for (std::size_t k = 0; k < m.provides_blocks.size(); ++k) {
        const detail::Slot& dst_slot = slots[m.provides_source_arcs[k]];
        const std::size_t len = R.blocks[m.provides_blocks[k]].length;
        if (dst_slot.data != nullptr) {
          host_to_slot(provides_snapshot[mi].data() + off, dst_slot, len);
        }
        off += len;
      }
    }
  }

  if (!eval_err.ok()) return eval_err;
  if (!body_err.ok()) return body_err;
  return Result<void>::success();
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

/// TEST/DIAGNOSTIC-ONLY, the `NewtonResidualPath::Direct` sibling immediately
/// above -- see this function's own declaration (fuse/runtime/runner.hpp) for
/// what it is for. Structurally identical to `region_fixed_point_residual()`
/// immediately above, with `evaluate` calling
/// `evaluate_region_residual_direct()` in place of
/// `evaluate_region_residual()`.
Result<FixedPointResidual> Runner::region_fixed_point_residual_direct(
    std::size_t region_index, ComponentSet& components) {
  if (!impl_->initialized) {
    return Failure{"E-RT-REGION",
                 "region_fixed_point_residual_direct() called before a "
                 "successful initialize()"};
  }
  if (region_index >= impl_->regions.size()) {
    return Failure{"E-RT-REGION",
                 "region index " + std::to_string(region_index) +
                     " is out of range (" +
                     std::to_string(impl_->regions.size()) + " region(s))"};
  }
  const detail::RegionRuntime& R = impl_->regions[region_index];

  // `RegionRuntime::newton_direct_members` is built in resolve_regions()
  // ONLY when `decl.newton_config.residual_path == Direct` (this file's
  // own sibling, src/runtime/iteration.cpp -- no reason to resolve a
  // Direct member list for a region that will never use it). Calling
  // THIS diagnostic on a region configured for Advance would otherwise
  // silently run evaluate_region_residual_direct() against an empty
  // member list -- every Provides block's own Source arc left
  // unwritten, reading back whatever stale value it held from the
  // runner's own initial tick -- a wrong answer with no symptom, not a
  // refusal. Checked here, by name, rather than left to whatever the
  // empty list happens to produce.
  const IterationRegionDecl& decl = impl_->graph.iteration_region(
      RegionIdx{static_cast<std::uint32_t>(region_index)});
  if (decl.newton_config.residual_path != NewtonResidualPath::Direct) {
    return Failure{"E-RT-REGION",
                 "region_fixed_point_residual_direct() called for iteration "
                     "region '" + R.name + "', whose newton_config.residual_"
                     "path is not Direct, so it has no Direct member list to "
                     "evaluate against; region_fixed_point_residual() is the "
                     "Advance-path diagnostic every Newton region supports"};
  }

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
    return impl->evaluate_region_residual_direct(*comps, region_index, in, f_out);
  };
  return out;
}

/// TEST/DIAGNOSTIC-ONLY -- see this function's own declaration
/// (fuse/runtime/runner.hpp) for what it is for.
Result<std::vector<double>> Runner::debug_newton_scale(std::size_t region_index) const {
  if (!impl_->initialized) {
    return Failure{"E-RT-REGION",
                 "debug_newton_scale() called before a successful initialize()"};
  }
  if (region_index >= impl_->regions.size()) {
    return Failure{"E-RT-REGION",
                 "region index " + std::to_string(region_index) +
                     " is out of range (" +
                     std::to_string(impl_->regions.size()) + " region(s))"};
  }
  return impl_->regions[region_index].newton_scale;
}

Result<std::vector<std::size_t>> Runner::debug_newton_gmres_iterations(
    std::size_t region_index) const {
  if (!impl_->initialized) {
    return Failure{"E-RT-REGION",
                 "debug_newton_gmres_iterations() called before a successful "
                 "initialize()"};
  }
  if (region_index >= impl_->regions.size()) {
    return Failure{"E-RT-REGION",
                 "region index " + std::to_string(region_index) +
                     " is out of range (" +
                     std::to_string(impl_->regions.size()) + " region(s))"};
  }
  return impl_->regions[region_index].newton_gmres_iterations;
}

}  // namespace fuse
