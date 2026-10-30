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

#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "fuse/backend/weights_operator.hpp"
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
  if (config.execution_space == RunnerConfig::RunnerExecutionSpace::Device) {
    return refuse("E-ADJ-SPACE",
                  "the adjoint is implemented for the host execution space only; "
                  "RunnerConfig::execution_space is Device");
  }
  if (!regions.empty()) {
    return refuse("E-ADJ-REGION",
                  "the plan has an iteration region; the adjoint of a fixed-point "
                  "iteration (implicit differentiation) is not implemented, and "
                  "unrolling the sweeps would need the sweep count the forward run "
                  "reached");
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
      case OpKind::Transport:
        if (r.transport < 0 || transport_plans[static_cast<std::size_t>(r.transport)].has_graph) {
          return refuse("E-ADJ-TRANSPORT",
                        "instruction " + std::to_string(i) +
                            " moves data between processes; the adjoint of an MPI "
                            "exchange is not implemented");
        }
        break;
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
  st->owned.assign(slots.size(), {});
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
      st->owned[i].assign(s.total(), 0.0);
      st->bar[i] = st->owned[i].data();
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
    if (view.space() != MemorySpace::Host) {
      return refuse("E-ADJ-SPACE", "the adjoint array of field '" + b.label +
                                       "' is not in host memory");
    }
    if (!flat_contiguous(view.layout()) || view.layout().count != s.elements ||
        view.layout().components != s.components) {
      return refuse("E-ADJ-LAYOUT",
                    "the adjoint array of field '" + b.label +
                        "' must be a flat, component-fastest array of " +
                        std::to_string(s.elements) + "x" + std::to_string(s.components) +
                        " values, the layout of the field itself");
    }
    if (view.data() == s.data) {
      return refuse("E-ADJ-ALIAS", "the adjoint array of field '" + b.label +
                                       "' is the field's own array");
    }
    st->bar[i] = view.data();
  }
  st->scratch.assign(largest, 0.0);
  st->scratch_in.assign(largest, 0.0);

  // 5. The transposable operator of every remap.
  st->remap_operator.reserve(remap_weights.size());
  for (const RemapWeights* w : remap_weights) {
    st->remap_operator.push_back(w != nullptr ? std::make_unique<WeightsOperator>(*w)
                                              : std::unique_ptr<LinearOperator>{});
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

  for (std::size_t e = A.events.size(); e-- > 0;) {
    const AdjointState::Event& ev = A.events[e];
    if (!ev.is_tick) {
      const Result<void> r = reverse_instruction(ev.index);
      if (!r.ok()) return r;
      continue;
    }
    const Tick& tick = ticks[ev.index];
    StepContext ctx;  // the context the forward step had (Impl::advance)
    ctx.t_begin = Rational{p} * plan.period() + tick.time;
    ctx.t_end = ctx.t_begin + tick.dt;
    ctx.step = p * tick.per_period + tick.local_step;
    ctx.iteration = 0;
    Component& comp = components.at(tick.slot);
    if (!ctx.valid()) {
      return refuse("E-RT-STEP", "step context [" + ctx.t_begin.to_string() + ", " +
                                     ctx.t_end.to_string() + ") for component '" +
                                     comp.name() + "' is not valid");
    }
    AdjointComponent* adj = comp.adjoint();
    if (adj == nullptr) {
      return refuse("E-ADJ-COMPONENT", "component '" + comp.name() +
                                           "' no longer has an adjoint");
    }
    const Result<void> r = adj->advance_adjoint(ctx);
    if (!r.ok()) return r;
  }
  --A.next;
  return Result<void>::success();
}

Result<void> Runner::Impl::reverse_instruction(std::size_t index) {
  const Instruction& ins = plan.instructions()[index];
  const Resolved& r = resolved[index];
  AdjointState& A = *adjoint;
  const std::size_t n = r.out_total;
  // The forward cases `break` before doing anything when they have no input or
  // no output; so does the reverse.
  const bool has_io = ins.n_in > 0 && ins.out.valid();

  // g <- adjoint of the output; the adjoint of the output <- 0.
  const auto take = [&]() -> const double* {
    double* g = A.scratch.data();
    double* b = A.bar[ins.out.value()];
    for (std::size_t k = 0; k < n; ++k) {
      g[k] = b[k];
      b[k] = 0.0;
    }
    return g;
  };
  // adjoint of input `a` (`count` values) += f * g
  const auto add_n = [&](ArcId a, const double* g, double f, std::size_t count) {
    double* b = A.bar[a.value()];
    for (std::size_t k = 0; k < count; ++k) b[k] += f * g[k];
  };
  const auto add = [&](ArcId a, const double* g, double f) { add_n(a, g, f, n); };

  switch (ins.kind) {
    case OpKind::Source:
    case OpKind::BuildRemapHalo:
      // The arc is the provider's own array: the adjoint arrays of its readers
      // were added to the provider's adjoint array when they were reversed.
      return Result<void>::success();

    case OpKind::Sink:
    case OpKind::Extrapolate:  // order 0: a copy
    case OpKind::Interpolate:  // without a time history: a copy
    case OpKind::Epoch:        // agreement, then a copy
    case OpKind::Constrain:    // reports the conservation defect, copies
    case OpKind::Diagnose:     // checks for non-finite values, copies
    case OpKind::Transport: {  // within one process: a copy
      if (!has_io) return Result<void>::success();
      add(ins.in[0], take(), 1.0);
      return Result<void>::success();
    }

    case OpKind::Remap: {
      if (!has_io) return Result<void>::success();
      const double* g = take();
      if (r.remap < 0) {  // the proved-coincident identity
        add(ins.in[0], g, 1.0);
        return Result<void>::success();
      }
      const auto ri = static_cast<std::size_t>(r.remap);
      const RemapWeights& w = *remap_weights[ri];
      const auto nc = static_cast<std::size_t>(slots[ins.out.value()].components);
      double* in_bar = A.scratch_in.data();
      const Result<void> t = A.remap_operator[ri]->apply_transpose(
          std::span<const double>(g, w.n_rows * nc), std::span<double>(in_bar, w.n_cols * nc),
          static_cast<int>(nc));
      if (!t.ok()) return t;
      add_n(ins.in[0], in_bar, 1.0, w.n_cols * nc);
      return Result<void>::success();
    }

    case OpKind::Scale: {
      if (!has_io) return Result<void>::success();
      add(ins.in[0], take(), ins.attrs.scale.to_double());
      return Result<void>::success();
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
      return Result<void>::success();
    }

    case OpKind::Sample: {
      if (!has_io) return Result<void>::success();
      // Forward: acc = sample (the first of a window, or Instant), else
      // acc = acc + sample. So the sample receives the adjoint of the
      // accumulator, and the accumulator's own adjoint is dead before the first
      // sample of the window (or before any sample, for Instant).
      double* acc = A.bar[ins.out.value()];
      double* in = A.bar[ins.in[0].value()];
      for (std::size_t k = 0; k < n; ++k) in[k] += acc[k];
      if (r.accum == AccumOp::Instant || A.window_index[index] == 0) {
        for (std::size_t k = 0; k < n; ++k) acc[k] = 0.0;
      }
      return Result<void>::success();
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
      return Result<void>::success();
    }

    default:
      return refuse("E-ADJ-OP", std::string{"instruction "} + std::to_string(index) +
                                    " is a '" + to_cstring(ins.kind) +
                                    "', which has no adjoint");
  }
}

}  // namespace fuse
