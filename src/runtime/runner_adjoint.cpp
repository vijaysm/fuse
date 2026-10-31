// Reverse mode (adjoint) of one period of the coupling (plans/2026-10-22-
// universal-number-types.md, stage U6; docs/dev/adjoint-schedule.md).
//
// A forward period is a sequence of events: component steps (`Tick`) and
// coupling instructions. Each instruction is linear in the field values, with
// weights fixed at setup, so `y = A x` is reversed by `xbar += A^T ybar` and
// needs no value from the forward run. The reverse sweep walks the same
// sequence from the back. For a component step it calls the component's own
// `AdjointComponent::advance_adjoint()`; for an instruction it applies the
// rules below to the adjoint arrays, one per arc.
//
// An instruction that overwrites its output (every one except the accumulation
// of a later sample) is reversed as an assignment: the adjoint of the output is
// taken and set to zero, and what was taken is added to the adjoint of each
// input. That is the reverse of `y := A x` when `y` is dead before the write.

#include "fuse/runtime/runner.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <Kokkos_Core.hpp>

#include "fuse/num/adjoint_kernels.hpp"
#include "fuse/num/space_kernels.hpp"
#include "runner_impl.hpp"

namespace fuse {

using detail::AdjointState;
using detail::Resolved;
using detail::Slot;
using detail::Tick;

namespace {

/// The same test the forward setup applies to a registered field: the sweep
/// walks every adjoint array as a flat, component-fastest `double[n]`.
[[nodiscard]] bool flat_contiguous(const Layout& l) noexcept {
  if (l.components < 1) return false;
  if (l.count == 0) return true;
  const auto nc = static_cast<std::size_t>(l.components);
  if (nc == 1) return l.stride == 1;
  return l.stride == nc && l.effective_component_stride() == 1;
}

[[nodiscard]] Failure refuse(const char* code, std::string message) {
  return Failure{code, std::move(message)};
}

}  // namespace

// ---------------------------------------------------------------------------
// Public surface
// ---------------------------------------------------------------------------

Result<void> Runner::initialize_adjoint(ComponentSet& components,
                                        FieldRegistry& adjoint_registry) {
  if (!impl_->initialized) {
    return refuse("E-ADJ-INIT",
                  "initialize_adjoint() called before a successful initialize()");
  }
  try {
    return impl_->init_adjoint(components, adjoint_registry);
  } catch (const std::exception& e) {
    return refuse("E-ADJ-INIT", std::string{"initialize_adjoint failed: "} + e.what());
  }
}

Result<void> Runner::run_period_adjoint(ComponentSet& components) {
  if (!impl_->initialized || impl_->adjoint == nullptr) {
    return refuse("E-ADJ-INIT",
                  "run_period_adjoint() called before a successful "
                  "initialize_adjoint()");
  }
  try {
    return impl_->sweep_adjoint_period(components);
  } catch (const std::exception& e) {
    return refuse("E-ADJ-RUN", std::string{"adjoint period failed: "} + e.what());
  }
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

Result<void> Runner::Impl::init_adjoint(ComponentSet& components,
                                        FieldRegistry& adjoint_registry) {
  const std::vector<Instruction>& stream = plan.instructions();
  const std::vector<BufferDecl>& buffers = plan.buffers().buffers;

  // 1. What this stage does not differentiate through, refused by name.
  const bool device_exec =
      config.execution_space == RunnerConfig::RunnerExecutionSpace::Device;
  // An ordinary iteration region is reversed as the fixed point it converged to (the adjoint
  // iteration of `reverse_region`); what that cannot do is refused here, by name.
  std::vector<std::uint8_t> in_region_body(stream.size(), 0);
  for (const detail::RegionRuntime& R : regions) {
    if (R.newton) {
      return refuse("E-ADJ-REGION",
                    "iteration region '" + R.name +
                        "' is solved by Newton's method; the adjoint of a Newton solve "
                        "is not implemented (only the fixed point of an ordinary "
                        "sweep is reversed)");
    }
    if (R.window_scoped) {
      return refuse("E-ADJ-REGION", "iteration region '" + R.name +
                                        "' is scoped to a time window; the adjoint of a "
                                        "window iteration is not implemented");
    }
    if (R.participants.size() > 1) {
      return refuse("E-ADJ-REGION",
                    "iteration region '" + R.name +
                        "' spans several ranks; its adjoint iteration would need the "
                        "agreement of every rank on each pass, which is not implemented");
    }
    for (std::size_t i = R.enter; i <= R.exit && i < stream.size(); ++i) in_region_body[i] = 1;
  }
  for (std::size_t i = 0; i < slots.size(); ++i) {
    const std::string& label = buffers[i].label;
    const bool storage_typed = i < storage_binding_of_arc.size() && storage_binding_of_arc[i] >= 0;
    if (!slots[i].arithmetic.empty() || storage_typed) {
      return refuse("E-ADJ-FIELD-TYPE",
                    "arc %" + std::to_string(i) + " (" + label +
                        ") carries a field of an Arithmetic or Storage value type; "
                        "the adjoint is implemented for binary64 scalars and arrays "
                        "of them (dual numbers) only");
    }
  }

  // 2. Every instruction in the stream has an adjoint.
  std::size_t transport_in_max = 0, transport_out_max = 0;
  for (std::size_t i = 0; i < stream.size(); ++i) {
    const Instruction& ins = stream[i];
    const Resolved& r = resolved[i];
    // A time history is filed by a Sample and projected by an Interpolate; either half refuses.
    if ((ins.kind == OpKind::Sample || ins.kind == OpKind::Interpolate) && r.spacetime >= 0) {
      return refuse("E-ADJ-SPACETIME",
                    "instruction " + std::to_string(i) + " is a '" + to_cstring(ins.kind) +
                        "' of a time history (a space-time channel); its adjoint is not implemented");
    }
    switch (ins.kind) {
      case OpKind::Source:
      case OpKind::BuildRemapHalo:
      case OpKind::Sink:
      case OpKind::Merge:
      case OpKind::Scale:
      case OpKind::Accumulate:
      case OpKind::Epoch:
      case OpKind::Constrain:
      case OpKind::Diagnose:
        break;
      case OpKind::Residual:
        // The residual of an iteration region feeds only its convergence test, which the
        // adjoint iteration makes for itself: nothing to reverse.
        if (in_region_body[i] != 0) break;
        return refuse("E-ADJ-OP", "instruction " + std::to_string(i) +
                                      " is a Residual outside an iteration region, which has no adjoint");
      case OpKind::Extrapolate:
        // The forward run applies order 0 (a copy) whatever order is asked, because it keeps no
        // history; an adjoint written for that would go wrong silently the day it does.
        if (ins.attrs.order != 0) {
          return refuse("E-ADJ-OP",
                        "instruction " + std::to_string(i) + " is an Extrapolate of order " +
                            std::to_string(ins.attrs.order) +
                            "; only order 0 (a copy, all the forward run applies) has an adjoint");
        }
        break;
      case OpKind::Interpolate:
        break;
      case OpKind::Sample:
        if (r.accum == AccumOp::Min || r.accum == AccumOp::Max) {
          return refuse("E-ADJ-ACCUM",
                        "instruction " + std::to_string(i) +
                            " accumulates a minimum or maximum; the adjoint of "
                            "selecting a sample needs the forward values, which the "
                            "coupler does not keep");
        }
        break;
      case OpKind::Remap:
        if (r.nodal_dist >= 0) {
          return refuse("E-ADJ-NODAL-DIST",
                        "instruction " + std::to_string(i) +
                            " is a distributed nodal Remap; the adjoint of its MPI "
                            "exchange is not implemented");
        }
        break;
      case OpKind::Transport: {
        if (ins.n_in == 0 || !ins.out.valid()) break;
        if (r.transport < 0) {
          return refuse("E-ADJ-TRANSPORT",
                        "instruction " + std::to_string(i) + " is a Transport that resolved to no plan");
        }
        const auto pi = static_cast<std::size_t>(r.transport);
        if (!transport_plans[pi].has_graph) break;  // within one process: a copy
        // Between processes: the exchange run backwards, by the backend that ran it forward.
        const Transport* const tp = pi < graph_transports.size() ? graph_transports[pi].get() : nullptr;
        if (tp == nullptr || !tp->has_adjoint()) {
          return refuse("E-ADJ-TRANSPORT",
                        "instruction " + std::to_string(i) + " moves data between processes through the '" +
                            (tp != nullptr ? tp->name() : std::string{"(none)"}) +
                            "' transport, which has no adjoint (Transport::move_adjoint)");
        }
        transport_in_max = std::max(transport_in_max, slots[ins.in[0].value()].total());
        transport_out_max = std::max(transport_out_max, slots[ins.out.value()].total());
        break;
      }
      default:
        return refuse("E-ADJ-OP", std::string{"instruction "} + std::to_string(i) +
                                      " is a '" + to_cstring(ins.kind) +
                                      "', which has no adjoint");
    }
  }

  // 3. Every component that advances can be reversed.
  for (const Tick& t : ticks) {
    Component& c = components.at(t.slot);
    if (c.is_externally_driven()) {
      return refuse("E-ADJ-COMPONENT",
                    "component '" + c.name() + "' is driven from outside the Runner");
    }
    if (c.adjoint() == nullptr) {
      return refuse("E-ADJ-COMPONENT",
                    "component '" + c.name() +
                        "' advances but has no adjoint (Component::adjoint() is null)");
    }
  }

  // 4. One adjoint array per arc: the component's own for an arc that aliases
  //    a field, the coupler's for the rest.
  auto st = std::make_unique<AdjointState>();
  st->bar.assign(slots.size(), nullptr);
  st->bar_device.assign(slots.size(), 0);
  st->owned.assign(slots.size(), {});
  st->owned_device.assign(slots.size(), {});
  std::unordered_map<std::uint32_t, FieldId> arc_field;
  for (std::uint32_t i = 0; i < dag.node_count(); ++i) {
    const Node& n = dag.node(NodeId{i});
    if ((n.kind != OpKind::Source && n.kind != OpKind::Sink) || !n.output.valid()) continue;
    const auto dot = n.label.find('.');
    if (dot == std::string::npos) continue;
    const auto cid = graph.find_component(n.label.substr(0, dot));
    if (!cid.has_value()) continue;
    const auto fid = graph.find_field(*cid, n.label.substr(dot + 1));
    if (fid.has_value()) arc_field.emplace(n.output.value(), *fid);
  }
  std::size_t largest = 0;
  for (std::size_t i = 0; i < slots.size(); ++i) {
    const BufferDecl& b = buffers[i];
    const Slot& s = slots[i];
    largest = std::max(largest, s.total());
    if (!b.is_component_owned) {
      if (s.device) {  // an arc the forward run keeps in device memory
        st->owned_device[i] = Kokkos::View<double*, detail::AdjointDeviceMemSpace>(
            Kokkos::view_alloc(std::string("fuse::adjoint_bar")), std::max<std::size_t>(s.total(), 1));
        st->bar[i] = st->owned_device[i].data();
        st->bar_device[i] = 1;
      } else {
        st->owned[i].assign(s.total(), 0.0);
        st->bar[i] = st->owned[i].data();
      }
      continue;
    }
    const auto it = arc_field.find(b.arc.value());
    if (it == arc_field.end()) {
      return refuse("E-ADJ-UNBOUND", "component-owned arc %" + std::to_string(i) + " (" +
                                         b.label + ") does not resolve to a field");
    }
    const Result<FieldRef> ref = adjoint_registry.get_scalars(it->second);
    if (!ref.ok()) {
      return refuse("E-ADJ-UNBOUND", "field '" + b.label + "' has no adjoint array: " +
                                         ref.error().message +
                                         " (did the component call register_adjoint_fields?)");
    }
    FieldRef view = ref.value();  // by value: the registry hands back a copy
    // The adjoint array is in the memory of the field it is the adjoint of: a field in
    // device memory is reversed by kernels on the device, a field in host memory by
    // kernels on the host or, for device execution, through a staged copy.
    const MemorySpace want = s.device ? MemorySpace::Device : MemorySpace::Host;
    if (view.space() != want) {
      return refuse("E-ADJ-SPACE",
                    "the adjoint array of field '" + b.label + "' is in " +
                        to_cstring(view.space()) + " memory but the field is in " +
                        to_cstring(want) + " memory; the two must be in the same memory space");
    }
    if (!flat_contiguous(view.layout()) || view.layout().count != s.elements ||
        view.layout().components != s.components) {
      return refuse("E-ADJ-LAYOUT",
                    "the adjoint array of field '" + b.label +
                        "' must be a flat, component-fastest array of " +
                        std::to_string(s.elements) + "x" + std::to_string(s.components) +
                        " values, the layout of the field itself");
    }
    // (A rank that holds none of a field has no array: two null pointers are not an alias.)
    if (s.total() != 0 && view.data() == s.data) {
      return refuse("E-ADJ-ALIAS", "the adjoint array of field '" + b.label +
                                       "' is the field's own array");
    }
    st->bar[i] = view.data();
    st->bar_device[i] = s.device ? 1 : 0;
  }
  st->transport_in.assign(transport_in_max, 0.0);
  st->transport_out.assign(transport_out_max, 0.0);
  // Scratch lanes in the memory of the execution space.
  const std::size_t lane_size = std::max<std::size_t>(largest, 1);
  if (device_exec) {
    st->lanes_device.reserve(AdjointState::kLanes);
    for (std::size_t k = 0; k < AdjointState::kLanes; ++k) {
      st->lanes_device.emplace_back(Kokkos::view_alloc(std::string("fuse::adjoint_lane")), lane_size);
    }
  } else {
    st->lanes_host.assign(AdjointState::kLanes, std::vector<double>(lane_size, 0.0));
  }

  // 5. The transpose of every remap as a CSR matrix: the triplets with rows and columns
  //    exchanged, then the forward run's own stable sort by row, so that each output
  //    entry sums its terms in the order of the triplets, as `apply_transpose` of
  //    `WeightsOperator` does.
  st->remap_t.resize(remap_weights.size());
  if (device_exec) st->remap_t_device.resize(remap_weights.size());
  for (std::size_t ri = 0; ri < remap_weights.size(); ++ri) {
    const RemapWeights* w = remap_weights[ri];
    if (w == nullptr) continue;
    RemapWeights wt;
    wt.row = w->col;
    wt.col = w->row;
    wt.val = w->val;
    wt.n_rows = w->n_cols;
    wt.n_cols = w->n_rows;
    Result<detail::RemapCsr> csr = build_remap_csr(wt);
    if (!csr.ok()) return csr.error();
    st->remap_t[ri] = csr.value();
    if (device_exec) {
      st->remap_t_device[ri] = detail::mirror_remap_csr<detail::AdjointDeviceMemSpace>(st->remap_t[ri]);
    }
  }

  // 6. The order of one period: the merged walk of `step_period()`, replayed
  //    without executing anything. The condition below is the one it uses.
  st->events.reserve(ticks.size() + stream.size());
  {
    std::size_t ti = 0;
    for (std::size_t i = 0; i < stream.size(); ++i) {
      const Instruction& ins = stream[i];
      const bool ticks_here = ins.phase == TickPhase::AfterTicks;
      while (ti < ticks.size() &&
             (ticks[ti].time < ins.time || (ticks_here && !(ins.time < ticks[ti].time)))) {
        st->events.push_back({true, ti++});
      }
      st->events.push_back({false, i});
    }
    for (; ti < ticks.size(); ++ti) st->events.push_back({true, ti});
  }

  // 7. Accumulation windows, from the same replay: the position of each sample
  //    in its window, and the number of samples each Accumulate reduces. The
  //    forward run counts these at run time (`sample_count`); here they follow
  //    from the stream alone, and a window that is still open at the end of the
  //    period would make them depend on the previous period.
  st->window_index.assign(stream.size(), 0);
  {
    std::vector<std::size_t> count(slots.size(), 0);
    for (const AdjointState::Event& e : st->events) {
      if (e.is_tick) continue;
      const Instruction& ins = stream[e.index];
      if (ins.n_in == 0 || !ins.out.valid()) continue;
      if (ins.kind == OpKind::Sample) {
        st->window_index[e.index] = count[ins.out.value()]++;
      } else if (ins.kind == OpKind::Accumulate) {
        st->window_index[e.index] = count[ins.in[0].value()];
        count[ins.in[0].value()] = 0;
      }
    }
    for (std::size_t a = 0; a < count.size(); ++a) {
      if (count[a] != 0) {
        return refuse("E-ADJ-WINDOW",
                      "arc %" + std::to_string(a) + " (" + buffers[a].label +
                          ") still holds " + std::to_string(count[a]) +
                          " sample(s) of an accumulation window at the end of a period; "
                          "the adjoint of a window that spans periods is not implemented");
      }
    }
  }

  // 8. Iteration regions: the interval of events each is reversed over. It runs from the first
  //    component step the region re-advances to the last instruction of its body, and may hold
  //    instructions of other channels but no step of a component the region does not step (the
  //    adjoint step of such a component would be called once per pass).
  st->event_loop.assign(st->events.size(), -1);
  for (std::size_t ri = 0; ri < regions.size(); ++ri) {
    const detail::RegionRuntime& R = regions[ri];
    std::size_t first = st->events.size(), last = 0, body = 0;
    for (std::size_t e = 0; e < st->events.size(); ++e) {
      const AdjointState::Event& ev = st->events[e];
      if (ev.is_tick || ev.index < R.enter || ev.index > R.exit) continue;
      first = std::min(first, e);
      last = std::max(last, e);
      ++body;
    }
    if (body == 0 || body != R.exit - R.enter + 1) {
      return refuse("E-ADJ-REGION", "the body of iteration region '" + R.name +
                                        "' does not appear once in a period");
    }
    for (const std::size_t t : R.tick_indices) {
      for (std::size_t e = 0; e < st->events.size(); ++e) {
        if (st->events[e].is_tick && st->events[e].index == t) first = std::min(first, e);
      }
    }
    for (std::size_t e = first; e <= last; ++e) {
      const AdjointState::Event& ev = st->events[e];
      if (st->event_loop[e] >= 0) {
        return refuse("E-ADJ-REGION", "iteration region '" + R.name +
                                          "' overlaps another region in the order of a period");
      }
      if (ev.is_tick &&
          std::find(R.tick_indices.begin(), R.tick_indices.end(), ev.index) == R.tick_indices.end()) {
        return refuse("E-ADJ-REGION",
                      "component '" + components.at(ticks[ev.index].slot).name() +
                          "' steps between the steps and the body of iteration region '" + R.name +
                          "' without being part of it; its adjoint step would be called once per pass");
      }
      st->event_loop[e] = static_cast<std::int32_t>(st->loops.size());
    }
    st->loops.push_back({ri, first, last});
  }
  if (!st->loops.empty()) {
    // every adjoint array a pass can change: the arcs, and the fields of the components stepped
    const auto add_managed = [&](double* ptr, std::size_t len, bool device) {
      if (ptr == nullptr || len == 0) return;
      for (const AdjointState::ManagedArray& m : st->managed) {
        if (m.ptr == ptr) return;
      }
      st->managed.push_back({ptr, len, device});
    };
    for (std::size_t i = 0; i < slots.size(); ++i) {
      add_managed(st->bar[i], slots[i].total(), st->bar_device[i] != 0);
    }
    for (const AdjointState::RegionLoop& loop : st->loops) {
      for (const std::size_t slot : regions[loop.region].component_slots) {
        const auto cid = graph.find_component(components.at(slot).name());
        if (!cid.has_value()) continue;
        for (const FieldId f : graph.component(*cid).fields) {
          const Result<FieldRef> found = adjoint_registry.get_scalars(f);
          if (!found.ok() || !flat_contiguous(found.value().layout())) continue;
          FieldRef view = found.value();  // by value: data() is not const
          add_managed(view.data(), view.layout().count * static_cast<std::size_t>(view.layout().components),
                      view.space() == MemorySpace::Device);
        }
      }
    }
    st->snapshot.resize(st->managed.size());
    for (std::size_t i = 0; i < st->managed.size(); ++i) st->snapshot[i].assign(st->managed[i].len, 0.0);
  }

  adjoint = std::move(st);
  return Result<void>::success();
}

// ---------------------------------------------------------------------------
// The reverse sweep
// ---------------------------------------------------------------------------

Result<void> Runner::Impl::sweep_adjoint_period(ComponentSet& components) {
  AdjointState& A = *adjoint;

  // A sweep reverses the periods the forward run completed, newest first. It is
  // opened by the first call after a forward run; running forward again in the
  // middle would pair adjoint steps with the wrong forward ones.
  if (!A.open || (A.next < 0 && period_index != A.forward_periods)) {
    if (period_index < 1) {
      return refuse("E-ADJ-ORDER", "run_period_adjoint(): no forward period has run");
    }
    A.open = true;
    A.forward_periods = period_index;
    A.next = period_index - 1;
  } else if (period_index != A.forward_periods) {
    return refuse("E-ADJ-ORDER",
                  "run_period_adjoint(): a forward period ran since the reverse sweep "
                  "began (" + std::to_string(A.forward_periods) + " periods then, " +
                      std::to_string(period_index) + " now)");
  }
  if (A.next < 0) {
    return refuse("E-ADJ-ORDER",
                  "run_period_adjoint(): every forward period has already been reversed");
  }
  const std::int64_t p = A.next;

  // A failure on one rank is carried to the end of the sweep, as `step_period` carries one: the
  // rank goes on executing the collective instructions (an exchange, an agreement point), so
  // that the ranks it exchanges with are not left waiting for it, and skips the rest, which is
  // local arithmetic on a sweep being abandoned. The failing rank reports its own cause at the
  // end. Without a communicator this is a sweep that stops doing work at its first failure.
  const std::vector<Instruction>& stream = plan.instructions();
  step_failure.reset();
  for (std::size_t e = A.events.size(); e-- > 0;) {
    const AdjointState::Event& ev = A.events[e];
    // An iteration region is reversed as a whole when the sweep reaches the end of its interval.
    if (const std::int32_t loop = A.event_loop[e]; loop >= 0) {
      if (e != A.loops[static_cast<std::size_t>(loop)].last_event || step_failure.has_value()) continue;
      const Result<void> r = reverse_region(components, static_cast<std::size_t>(loop), p);
      if (!r.ok()) step_failure = r.error();
      continue;
    }
    if (!ev.is_tick) {
      if (step_failure.has_value() && !is_collective(stream[ev.index].kind)) continue;
      const Result<void> r = reverse_instruction(ev.index);
      if (!r.ok()) {
        if (!step_failure.has_value()) step_failure = r.error();
        // An Epoch that fails has already reconciled: nothing is left to tell the partners.
        if (stream[ev.index].kind == OpKind::Epoch) break;
      }
      continue;
    }
    if (step_failure.has_value()) continue;
    const Result<void> r = reverse_tick(components, ev.index, p);
    if (!r.ok()) step_failure = r.error();
  }
  Kokkos::fence("fuse::adjoint_sweep");
  if (step_failure.has_value()) {
    // A period that failed did not happen: the next call reverses it again.
    const Failure f = *step_failure;
    step_failure.reset();
    return f;
  }
  --A.next;
  return Result<void>::success();
}

Result<void> Runner::Impl::reverse_tick(ComponentSet& components, std::size_t tick_index,
                                        std::int64_t period) {
  const Tick& tick = ticks[tick_index];
  StepContext ctx;  // the context the forward step had (Impl::advance)
  ctx.t_begin = Rational{period} * plan.period() + tick.time;
  ctx.t_end = ctx.t_begin + tick.dt;
  ctx.step = period * tick.per_period + tick.local_step;
  ctx.iteration = 0;
  Component& comp = components.at(tick.slot);
  if (!ctx.valid()) {
    return refuse("E-RT-STEP", "step context [" + ctx.t_begin.to_string() + ", " +
                                   ctx.t_end.to_string() + ") for component '" + comp.name() +
                                   "' is not valid");
  }
  AdjointComponent* adj = comp.adjoint();
  if (adj == nullptr) {
    return refuse("E-ADJ-COMPONENT", "component '" + comp.name() + "' no longer has an adjoint");
  }
  // The kernels of the instructions reversed so far must have finished before the
  // component reads or writes the adjoint arrays; the component's own kernels must
  // have finished when it returns (Component's threading contract).
  Kokkos::fence("fuse::adjoint_before_component");
  return adj->advance_adjoint(ctx);
}

// An iteration region, reversed as the fixed point the forward sweeps converged to.
//
// Forward, a period with a region steps the components of the region, runs the body of the
// region (the channels, ending in the delivery of G(y) into the components' inputs), and repeats
// from a restored state with the accelerated iterate, until G(y) = y within tolerance. At the
// fixed point y* = F(y*, x), with x everything the region reads from outside (the checkpointed
// states, the drives, the values delivered before it), so the derivative of the converged values
// is (I - F_y)^-1 F_x whatever way the iteration reached y*, and its adjoint is
//
//     lambda = ybar + F_y^T lambda,        xbar += F_x^T lambda,
//
// where ybar is the adjoint of the delivered values as the sweep found it. One pass of the
// reverse of the interval (the body, the instructions between, the component steps) given a seed
// lambda on the delivered values leaves F_y^T lambda on them and F_x^T lambda in the adjoint
// arrays of everything else, so the iteration is the reverse of the interval repeated to a
// fixed point (Christianson, "Reverse accumulation and attractive fixed points", 1994): the
// arrays are restored from a copy taken before the first pass so that the contribution to xbar
// is that of the last pass only, and the iteration is held to the convergence criterion of the
// region. The initial iterate y_0 does not influence y*, so what remains on the delivered values
// is zero.
Result<void> Runner::Impl::reverse_region(ComponentSet& components, std::size_t loop_index,
                                          std::int64_t period) {
  AdjointState& A = *adjoint;
  const AdjointState::RegionLoop& loop = A.loops[loop_index];
  detail::RegionRuntime& R = regions[loop.region];

  const auto copy_out = [&](const AdjointState::ManagedArray& m, double* host) {
    if (m.device) {
      Kokkos::deep_copy(num::SpanOf<HostExecSpace>(host, m.len),
                        num::ConstSpanOf<DeviceExecSpace>(m.ptr, m.len));
    } else if (m.len != 0) {
      std::copy(m.ptr, m.ptr + m.len, host);
    }
  };
  const auto copy_in = [&](const AdjointState::ManagedArray& m, const double* host) {
    if (m.device) {
      Kokkos::deep_copy(num::SpanOf<DeviceExecSpace>(m.ptr, m.len),
                        num::ConstSpanOf<HostExecSpace>(host, m.len));
    } else if (m.len != 0) {
      std::copy(host, host + m.len, m.ptr);
    }
  };
  // the iterate lives on the value arcs of the blocks (G(y), what the body computes); what the
  // component steps leave on the delivered values (`feed_array`) is its image under F_y^T
  const auto block_array = [&](const detail::RegionBlock& b) {
    return AdjointState::ManagedArray{A.bar[b.value_arc], b.length, A.bar_device[b.value_arc] != 0};
  };
  const auto feed_array = [&](const detail::RegionBlock& b) {
    return AdjointState::ManagedArray{A.bar[b.feed_arc], b.length, A.bar_device[b.feed_arc] != 0};
  };

  Kokkos::fence("fuse::adjoint_region_begin");
  for (std::size_t i = 0; i < A.managed.size(); ++i) copy_out(A.managed[i], A.snapshot[i].data());

  // the adjoint of the values the body computes, as the sweep found it (the delivery after the
  // region has been reversed), and what stood on the delivered values
  std::vector<double> ybar(R.n, 0.0), lambda(R.n, 0.0), next(R.n, 0.0), fed(R.n, 0.0), first_residual(R.blocks.size(), 0.0);
  for (const detail::RegionBlock& b : R.blocks) {
    copy_out(block_array(b), ybar.data() + b.offset);
    copy_out(feed_array(b), fed.data() + b.offset);
  }
  lambda = ybar;

  const std::vector<AdjointState::Event>& events = A.events;
  bool converged = false;
  int passes = 0;
  for (int j = 0; j < R.max_iterations && !converged; ++j) {
    // every pass starts from the state the interval was found in, with lambda on the delivered values
    for (std::size_t i = 0; i < A.managed.size(); ++i) copy_in(A.managed[i], A.snapshot[i].data());
    for (const detail::RegionBlock& b : R.blocks) copy_in(block_array(b), lambda.data() + b.offset);
    for (std::size_t e = loop.last_event + 1; e-- > loop.first_event;) {
      const AdjointState::Event& ev = events[e];
      const Result<void> r = ev.is_tick ? reverse_tick(components, ev.index, period)
                                        : reverse_instruction(ev.index);
      if (!r.ok()) return r;
    }
    Kokkos::fence("fuse::adjoint_region_pass");
    ++passes;

    // the new iterate: what the pass added to the delivered values, plus the seed
    for (const detail::RegionBlock& b : R.blocks) {
      copy_out(feed_array(b), next.data() + b.offset);
      for (std::size_t k = 0; k < b.length; ++k) {
        next[b.offset + k] += ybar[b.offset + k] - fed[b.offset + k];
      }
    }
    // held to the convergence criterion of the region, on the change of the iterate
    std::vector<double> residual(R.blocks.size(), 0.0), value(R.blocks.size(), 0.0);
    bool finite = true;
    for (std::size_t bi = 0; bi < R.blocks.size(); ++bi) {
      const detail::RegionBlock& b = R.blocks[bi];
      double rs = 0.0, vs = 0.0;
      for (std::size_t k = 0; k < b.length; ++k) {
        const double d = next[b.offset + k] - lambda[b.offset + k];
        rs += d * d;
        vs += next[b.offset + k] * next[b.offset + k];
      }
      residual[bi] = std::sqrt(rs);
      value[bi] = std::sqrt(vs);
      finite = finite && std::isfinite(residual[bi]) && std::isfinite(value[bi]);
      if (j == 0) first_residual[bi] = residual[bi];
    }
    if (!finite) {
      return refuse("E-ADJ-REGION-DIVERGED",
                    "the adjoint iteration of region '" + R.name + "' left the finite numbers after " +
                        std::to_string(passes) + " pass(es); the fixed-point map is not a contraction, "
                        "which the forward run's accelerator may have hidden");
    }
    std::vector<double> residual_norms(R.measure_block.size(), 0.0), reference_norms(R.measure_block.size(), 0.0);
    for (std::size_t mi = 0; mi < R.measure_block.size(); ++mi) {
      const std::size_t bi = R.measure_block[mi];
      residual_norms[mi] = residual[bi];
      reference_norms[mi] = R.convergence.measures()[mi].kind == MeasureKind::ResidualRelative
                                ? first_residual[bi]
                                : value[bi];
    }
    converged = R.convergence.all_satisfied(residual_norms, reference_norms);
    lambda = next;
  }
  if (!converged) {
    return refuse("E-ADJ-REGION-NOCONVERGE",
                  "the adjoint iteration of region '" + R.name + "' did not meet the region's convergence criterion in " +
                      std::to_string(R.max_iterations) +
                      " passes; the plain fixed-point iteration of the adjoint converges only when the "
                      "forward iteration does without acceleration");
  }
  // y_0, the iterate the forward sweeps started from, does not influence the fixed point: what
  // the last pass added to the delivered values is not passed on
  for (const detail::RegionBlock& b : R.blocks) copy_in(feed_array(b), fed.data() + b.offset);
  Kokkos::fence("fuse::adjoint_region_end");
  return Result<void>::success();
}

Result<void> Runner::Impl::reverse_instruction(std::size_t index) {
  if (config.execution_space == RunnerConfig::RunnerExecutionSpace::Device) {
    return reverse_on<DeviceExecSpace>(index);
  }
  return reverse_on<HostExecSpace>(index);
}

// The rule of each instruction on execution space `Space`. The adjoint array of an arc is
// in the memory of the arc's forward slot; an operand in a memory space the kernels of
// `Space` cannot reach is staged through a lane of `Space`'s memory, read in when the rule
// reads it and written back when the rule changes it, as the forward interpreter stages
// the arcs it keeps on the host.
template <class Space>
Result<void> Runner::Impl::reverse_on(std::size_t index) {
  const Instruction& ins = plan.instructions()[index];
  const Resolved& r = resolved[index];
  AdjointState& A = *adjoint;
  const std::size_t n = r.out_total;
  // The forward cases `break` before doing anything when they have no input or
  // no output; so does the reverse.
  const bool has_io = ins.n_in > 0 && ins.out.valid();

  constexpr bool kHostExec = Kokkos::SpaceAccessibility<Space, Kokkos::HostSpace>::accessible;
  using Span = num::SpanOf<Space>;
  using CSpan = num::ConstSpanOf<Space>;
  using HostSpan = num::SpanOf<HostExecSpace>;
  using HostCSpan = num::ConstSpanOf<HostExecSpace>;

  const auto lane = [&](std::size_t k) -> double* {
    if constexpr (kHostExec) {
      return A.lanes_host[k].data();
    } else {
      return A.lanes_device[k].data();
    }
  };

  // An adjoint array as the kernels of `Space` can reach it: itself when it is in
  // that memory, else lane `lane_index` holding a copy (`read`) of its `len` values.
  struct Staged {
    double* p = nullptr;       // what the kernels use
    double* host = nullptr;    // the host array to write back to, null when not staged
    std::size_t len = 0;
  };
  const auto stage = [&](ArcId a, std::size_t len, std::size_t lane_index, bool read) -> Staged {
    double* const ptr = A.bar[a.value()];
    if constexpr (kHostExec) {
      (void)lane_index;
      (void)read;
      return Staged{ptr, nullptr, len};
    } else {
      if (A.bar_device[a.value()] != 0) return Staged{ptr, nullptr, len};
      double* const l = lane(lane_index);
      if (read) Kokkos::deep_copy(Span(l, len), HostCSpan(ptr, len));
      return Staged{l, ptr, len};
    }
  };
  const auto commit = [&](const Staged& s) {
    if constexpr (!kHostExec) {
      if (s.host != nullptr) Kokkos::deep_copy(HostSpan(s.host, s.len), CSpan(s.p, s.len));
    }
  };

  // g <- adjoint of the output; the adjoint of the output <- 0. g is lane 0.
  const auto take = [&]() -> const double* {
    Staged o = stage(ins.out, n, 1, true);
    num::take<Space>(Span(lane(0), n), Span(o.p, n));
    commit(o);
    return lane(0);
  };
  // adjoint of input `a` (`count` values) += f * g
  const auto add_n = [&](ArcId a, const double* g, double f, std::size_t count) {
    Staged s = stage(a, count, 2, true);
    num::axpy<Space>(f, CSpan(g, count), Span(s.p, count));
    commit(s);
  };
  const auto add = [&](ArcId a, const double* g, double f) { add_n(a, g, f, n); };

  const auto done = [&]() -> Result<void> {
    if constexpr (!kHostExec) Kokkos::fence("fuse::adjoint_instruction");
    return Result<void>::success();
  };

  switch (ins.kind) {
    case OpKind::Source:
    case OpKind::BuildRemapHalo:
      // The arc is the provider's own array: the adjoint arrays of its readers
      // were added to the provider's adjoint array when they were reversed.
      return Result<void>::success();

    case OpKind::Residual:
      // The residual of an iteration region only feeds its convergence test.
      return Result<void>::success();

    case OpKind::Sink:
    case OpKind::Extrapolate:  // order 0: a copy
    case OpKind::Interpolate:  // without a time history: a copy
    case OpKind::Constrain:    // reports the conservation defect, copies
    case OpKind::Diagnose: {   // checks for non-finite values, copies
      if (!has_io) return Result<void>::success();
      add(ins.in[0], take(), 1.0);
      return done();
    }

    case OpKind::Epoch: {
      // The agreement of the forward run, then the copy reversed: a rank whose partners have
      // failed does not pass on an adjoint derived from an exchange that did not complete.
      const Result<void> agreed = reconcile_epoch(ins);
      if (!agreed.ok()) return agreed;
      if (!has_io) return Result<void>::success();
      add(ins.in[0], take(), 1.0);
      return done();
    }

    case OpKind::Transport: {
      if (!has_io) return Result<void>::success();
      detail::TransportPlan& p = transport_plans[static_cast<std::size_t>(r.transport)];
      if (!p.has_graph) {  // within one process: a copy
        add(ins.in[0], take(), 1.0);
        return done();
      }
      // Between processes. The forward run copied the entries it wrote from the source arc
      // (the same-rank matches directly, the rest through the exchange of `Transport::move`)
      // and left the other entries of the destination arc alone. Reversed: the adjoint of
      // each entry written is taken from the destination arc, which is set to zero there, and
      // added to the source entry it came from; the exchange itself runs backwards
      // (`Transport::move_adjoint`), on host buffers, as the forward exchange does.
      Transport* const tp = graph_transports[static_cast<std::size_t>(r.transport)].get();
      if (tp == nullptr) {
        return refuse("E-ADJ-TRANSPORT", "instruction " + std::to_string(index) +
                                             " has no transport instance");
      }
      const std::size_t n_in = slots[ins.in[0].value()].total();
      const std::size_t n_out = slots[ins.out.value()].total();
      const std::size_t out_arc = ins.out.value();
      const std::size_t nc = static_cast<std::size_t>(p.spec.components < 1 ? 1 : p.spec.components);

      // the adjoint of the destination arc, as a host array
      double* hout = A.bar[out_arc];
      const bool out_in_device_memory = A.bar_device[out_arc] != 0;
      if (out_in_device_memory) {
        hout = A.transport_out.data();
        Kokkos::deep_copy(HostSpan(hout, n_out), CSpan(A.bar[out_arc], n_out));
      }
      double* const hin = A.transport_in.data();
      std::fill(hin, hin + n_in, 0.0);

      Result<void> result = Result<void>::success();
      if (p.from_peers) {
        // A peer plan addresses both buffers by local entity index: the entries written are
        // `peer_written`, and the adjoint of the whole source block comes back.
        std::fill(p.recv.begin(), p.recv.begin() + static_cast<std::ptrdiff_t>(p.total()), 0.0);
        for (const std::size_t d2 : p.peer_written) {
          for (std::size_t c = 0; c < nc; ++c) {
            p.recv[d2 * nc + c] = hout[d2 * nc + c];
            hout[d2 * nc + c] = 0.0;
          }
        }
        result = tp->prepare(p.spec);
        if (result.ok()) result = tp->move_adjoint(p.recv.data(), p.send.data(), p.total());
        if (result.ok()) {
          for (std::size_t k = 0; k < p.peer_src_total; ++k) hin[k] += p.send[k];
        }
      } else {
        // The matched plan, backwards: scatter -> exchange -> gather, then the local matches.
        for (std::size_t k = 0; k < p.scatter.size(); ++k) {
          const std::size_t d2 = p.scatter[k] * nc;
          for (std::size_t c = 0; c < nc; ++c) {
            p.recv[k * nc + c] = hout[d2 + c];
            hout[d2 + c] = 0.0;
          }
        }
        result = tp->prepare(p.spec);
        if (result.ok()) result = tp->move_adjoint(p.recv.data(), p.send.data(), p.total());
        if (result.ok()) {
          for (std::size_t k = 0; k < p.gather.size(); ++k) {
            const std::size_t s = p.gather[k] * nc;
            for (std::size_t c = 0; c < nc; ++c) hin[s + c] += p.send[k * nc + c];
          }
        }
        for (std::size_t k = 0; k < p.local_src.size(); ++k) {
          const std::size_t s = p.local_src[k] * nc;
          const std::size_t d2 = p.local_dst[k] * nc;
          for (std::size_t c = 0; c < nc; ++c) {
            hin[s + c] += hout[d2 + c];
            hout[d2 + c] = 0.0;
          }
        }
      }
      if (out_in_device_memory) Kokkos::deep_copy(Span(A.bar[out_arc], n_out), HostCSpan(hout, n_out));
      if (!result.ok()) return result;
      // add the adjoint of the source to the arc's adjoint array
      const double* g = hin;
      if constexpr (!kHostExec) {
        Kokkos::deep_copy(Span(lane(3), n_in), HostCSpan(hin, n_in));
        g = lane(3);
      }
      add_n(ins.in[0], g, 1.0, n_in);
      return done();
    }

    case OpKind::Remap: {
      if (!has_io) return Result<void>::success();
      const double* g = take();
      if (r.remap < 0) {  // the proved-coincident identity
        add(ins.in[0], g, 1.0);
        return done();
      }
      const auto ri = static_cast<std::size_t>(r.remap);
      const RemapWeights& w = *remap_weights[ri];
      const auto nc = static_cast<std::size_t>(slots[ins.out.value()].components);
      // W^T g by the transposed matrix, then added to the adjoint of the input (lane 3)
      num::IndexSpanOf<Space> row_begin;
      num::IndexSpanOf<Space> col;
      num::ConstSpanOf<Space> val;
      if constexpr (kHostExec) {
        const detail::RemapCsr& csr = A.remap_t[ri];
        row_begin = num::IndexSpanOf<Space>(csr.row_begin.data(), csr.row_begin.extent(0));
        col = num::IndexSpanOf<Space>(csr.col.data(), csr.col.extent(0));
        val = num::ConstSpanOf<Space>(csr.val.data(), csr.val.extent(0));
      } else {
        const auto& csr = A.remap_t_device[ri];
        row_begin = num::IndexSpanOf<Space>(csr.row_begin.data(), csr.row_begin.extent(0));
        col = num::IndexSpanOf<Space>(csr.col.data(), csr.col.extent(0));
        val = num::ConstSpanOf<Space>(csr.val.data(), csr.val.extent(0));
      }
      num::remap_apply<Space>(row_begin, col, val, CSpan(g, w.n_rows * nc),
                              Span(lane(3), w.n_cols * nc), nc);
      add_n(ins.in[0], lane(3), 1.0, w.n_cols * nc);
      return done();
    }

    case OpKind::Scale: {
      if (!has_io) return Result<void>::success();
      add(ins.in[0], take(), ins.attrs.scale.to_double());
      return done();
    }

    case OpKind::Merge: {
      if (!has_io) return Result<void>::success();
      const double* g = take();
      switch (ins.attrs.merge) {
        case MergeOp::First:
          add(ins.in[0], g, 1.0);
          break;
        case MergeOp::Sum:
          for (std::uint8_t k = 0; k < ins.n_in; ++k) add(ins.in[k], g, 1.0);
          break;
        case MergeOp::Mean: {
          const double inv = 1.0 / static_cast<double>(ins.n_in);
          for (std::uint8_t k = 0; k < ins.n_in; ++k) add(ins.in[k], g, inv);
          break;
        }
      }
      return done();
    }

    case OpKind::Sample: {
      if (!has_io) return Result<void>::success();
      // Forward: acc = sample (the first of a window, or Instant), else
      // acc = acc + sample. So the sample receives the adjoint of the
      // accumulator, and the accumulator's own adjoint is dead before the first
      // sample of the window (or before any sample, for Instant).
      Staged acc = stage(ins.out, n, 1, true);
      Staged in = stage(ins.in[0], n, 2, true);
      num::axpy<Space>(1.0, CSpan(acc.p, n), Span(in.p, n));
      if (r.accum == AccumOp::Instant || A.window_index[index] == 0) {
        num::fill<Space>(0.0, Span(acc.p, n));
      }
      commit(in);
      commit(acc);
      return done();
    }

    case OpKind::Accumulate: {
      if (!has_io) return Result<void>::success();
      // Forward: out = acc, then Mean scales by 1 / count (count != 0) and
      // Integral by the sample interval. The accumulator is read, not changed.
      const std::size_t count = A.window_index[index];
      double factor = 1.0;
      if (ins.attrs.accum == AccumOp::Mean && count != 0) {
        factor = 1.0 / static_cast<double>(count);
      } else if (ins.attrs.accum == AccumOp::Integral) {
        factor = ins.attrs.scale.to_double();
      }
      add(ins.in[0], take(), factor);
      return done();
    }

    default:
      return refuse("E-ADJ-OP", std::string{"instruction "} + std::to_string(index) +
                                    " is a '" + to_cstring(ins.kind) +
                                    "', which has no adjoint");
  }
}

}  // namespace fuse
