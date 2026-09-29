#include "fuse/runtime/window_coupler.hpp"

#include <algorithm>
#include <cmath>

#include "fuse/num/temporal_basis.hpp"

namespace fuse {

namespace detail {

std::pair<std::vector<WindowSubstep>, Rational> plan_substeps(
    Rational clock, Rational interval, Rational window_end, DtPolicy policy) {
  std::vector<WindowSubstep> steps;
  Rational              t = clock;
  while (t < window_end) {
    const Rational end = t + interval;
    if (end > window_end) {
      if (policy == DtPolicy::Adjustable) {
        steps.push_back({t, window_end, false});
        t = window_end;
      } else {
        steps.push_back({t, end, true});
        t = end;
      }
      break;
    }
    steps.push_back({t, end, false});
    t = end;
  }
  return {std::move(steps), t};
}

Result<double> mean_over(const Waveform& wf, Rational a, Rational b) {
  double integral = 0.0;
  const Result<void> r = wf.integrate(a, b, &integral);
  if (!r.ok()) return r.error();
  return integral / (b - a).to_double();
}

}  // namespace detail

namespace {

// Unit A, step 1: `Substep`/`plan_substeps`/`mean_over` are now
// `fuse::detail::WindowSubstep`/`plan_substeps`/`mean_over`
// (fuse/runtime/window_coupler.hpp), unchanged, so a future subcycling-
// node runtime dispatch can call the identical functions. These aliases
// keep every existing use in THIS file (`Substep`, bare `plan_substeps(...)`,
// bare `mean_over(...)`) resolving to them without a single other line
// changing -- the safest way to widen a function's visibility without
// touching the logic that calls it.
using Substep = detail::WindowSubstep;
using detail::plan_substeps;
using detail::mean_over;

// WaveformIteration::GaussSeidel only: a topological order over `channels`'
// UNLAGGED provider -> consumer edges (an unlagged provider must run
// before its consumer, within one sweep, or the consumer would see this
// sweep's stale/seeded value instead of the fresh one GaussSeidel
// promises). `std::nullopt` if that graph has a cycle -- including "no
// channel at all is lagged", which is that graph with EVERY channel in
// it, so this ALSO doubles as the no-lag-at-all refusal
// (WindowCoupler::create()'s own E-RT-WINDOW-GAUSSSEIDEL-NOLAG): one
// method (this cycle check), one meaning, whether the cycle is total or
// partial.
//
// Kahn's algorithm, but the next zero-indegree node is always the
// SMALLEST-INDEX one available, not merely "any" one: declaration order
// is the tie-break among members with no ordering constraint between
// them, so two runs of the identical graph return the identical order
// (this is what a Runner-side dispatch driving the same compiled schedule
// needs to match, not merely "some" valid topological order).
[[nodiscard]] std::optional<std::vector<std::size_t>> topological_sweep_order(
    std::size_t n, const std::vector<WindowChannel>& channels) {
  std::vector<std::vector<std::size_t>> adj(n);
  std::vector<std::size_t> indeg(n, 0);
  for (const WindowChannel& ch : channels) {
    if (ch.lagged) continue;
    if (ch.provider >= n || ch.consumer >= n) continue;  // create() already checked
    adj[ch.provider].push_back(ch.consumer);
    ++indeg[ch.consumer];
  }
  std::vector<bool> done(n, false);
  std::vector<std::size_t> order;
  order.reserve(n);
  for (std::size_t step = 0; step < n; ++step) {
    std::size_t pick = n;  // sentinel: none found
    for (std::size_t i = 0; i < n; ++i) {
      if (!done[i] && indeg[i] == 0) { pick = i; break; }
    }
    if (pick == n) return std::nullopt;  // cycle: no zero-indegree node left
    done[pick] = true;
    order.push_back(pick);
    for (const std::size_t nb : adj[pick]) --indeg[nb];
  }
  return order;
}

}  // namespace

double extrapolate_boundary_value(
    const std::vector<std::pair<Rational, double>>& history, Rational target,
    int order) {
  FUSE_ASSERT(!history.empty(),
              "extrapolate_boundary_value: history must have >= 1 point");
  const std::size_t want = static_cast<std::size_t>(std::max(order, 2));
  const std::size_t take = std::min(want, history.size());
  const std::size_t start = history.size() - take;
  std::vector<double> t(take), v(take);
  for (std::size_t i = 0; i < take; ++i) {
    t[i] = (history[start + i].first - history[start].first).to_double();
    v[i] = history[start + i].second;
  }
  const int degree = static_cast<int>(take) - 1;
  const std::vector<double> coeffs = num::fit_polynomial(t, v, degree, 0.0);
  const double x = (target - history[start].first).to_double();
  return num::eval_polynomial(coeffs, x, 0.0);
}

namespace detail {

Result<void> advance_straddle_split(
    const WindowMember& member, Rational t_a, Rational t_split, Rational t_b,
    std::size_t step, int iteration,
    const std::function<Result<double>(std::size_t, Rational, Rational)>&
        mean_in_window,
    const std::function<Result<double>(std::size_t)>& boundary_at_split,
    const std::function<double(std::size_t, Rational)>& extrapolate_tail,
    double* out_at_split) {
  for (std::size_t e = 0; e < member.length; ++e) {
    const Result<double> mv = mean_in_window(e, t_a, t_split);
    if (!mv.ok()) return mv.error();
    member.required[e] = mv.value();
  }
  StepContext ctx_in;
  ctx_in.t_begin = t_a;
  ctx_in.t_end = t_split;
  ctx_in.step = step;
  ctx_in.iteration = iteration;
  const Result<void> ar_in = member.component->advance(ctx_in);
  if (!ar_in.ok()) return ar_in;

  for (std::size_t e = 0; e < member.length; ++e) {
    out_at_split[e] = member.provided[e];
  }

  if (t_split < t_b) {
    for (std::size_t e = 0; e < member.length; ++e) {
      const Result<double> bv = boundary_at_split(e);
      if (!bv.ok()) return bv.error();
      const double v_tail = extrapolate_tail(e, t_b);
      member.required[e] = 0.5 * (bv.value() + v_tail);
    }
    StepContext ctx_tail;
    ctx_tail.t_begin = t_split;
    ctx_tail.t_end = t_b;
    ctx_tail.step = step;
    ctx_tail.iteration = iteration;
    const Result<void> ar_tail = member.component->advance(ctx_tail);
    if (!ar_tail.ok()) return ar_tail;
  }
  return Result<void>::success();
}

}  // namespace detail

Result<double> WindowChannelIO::mean(
    const WindowChannel& /*ch*/, std::size_t e,
    const std::vector<std::unique_ptr<Waveform>>& provider, Rational a,
    Rational b) const {
  const std::size_t pe = std::min(e, provider.size() - 1);
  return mean_over(*provider[pe], a, b);
}

Result<double> WindowChannelIO::boundary(
    const WindowChannel& /*ch*/, std::size_t e,
    const std::vector<std::unique_ptr<Waveform>>& provider,
    Rational t) const {
  const std::size_t pe = std::min(e, provider.size() - 1);
  double v = 0.0;
  const Result<void> r = provider[pe]->evaluate(t, &v);
  if (!r.ok()) return r.error();
  return v;
}

double WindowChannelIO::extrapolate(
    const WindowChannel& /*ch*/, std::size_t e,
    const std::vector<std::vector<std::pair<Rational, double>>>&
        provider_history,
    Rational target, int order) const {
  const std::size_t pe = std::min(e, provider_history.size() - 1);
  return extrapolate_boundary_value(provider_history[pe], target, order);
}

WindowCoupler::WindowCoupler(std::vector<WindowMember> members,
                             std::vector<WindowChannel> channels,
                             WindowCouplerConfig cfg)
    : members_(std::move(members)),
      channels_(std::move(channels)),
      cfg_(std::move(cfg)) {
  channel_io_ = cfg_.channel_io ? cfg_.channel_io
                                : std::make_shared<WindowChannelIO>();
  // Jacobi: declaration order, unchanged from before this member existed
  // (every member reads the same sweep-start snapshot regardless of when
  // it runs, so no order is actually forced -- declaration order is as
  // good as any, and byte-identical to today's plain `for (i)` loop).
  // GaussSeidel: `create()` above already proved `topological_sweep_
  // order()` succeeds for this exact (members, channels) pair before
  // this constructor is ever reached (it is private; `create()` is the
  // only caller), so recomputing it here -- the SAME function, not a
  // second one that could disagree -- cannot fail.
  if (cfg_.iteration == WaveformIteration::GaussSeidel) {
    sweep_order_ = *topological_sweep_order(members_.size(), channels_);
  } else {
    sweep_order_.resize(members_.size());
    for (std::size_t i = 0; i < members_.size(); ++i) sweep_order_[i] = i;
  }
  state_.resize(members_.size());
  for (std::size_t i = 0; i < members_.size(); ++i) {
    MemberState& st = state_[i];
    st.history.resize(members_[i].length);
    for (std::size_t e = 0; e < members_[i].length; ++e) {
      st.history[e].push_back({Rational{0}, members_[i].provided[e]});
    }
  }
}

Result<std::unique_ptr<WindowCoupler>> WindowCoupler::create(
    std::vector<WindowMember> members, std::vector<WindowChannel> channels,
    WindowCouplerConfig cfg) {
  if (members.empty() || channels.empty()) {
    return Failure{"E-RT-WINDOW-EMPTY",
                 "WindowCoupler::create: at least one member and one "
                 "channel are required"};
  }
  if (cfg.order < 2) {
    return Failure{"E-RT-WINDOW-ORDER",
                 "WindowCoupler::create: cfg.order (" +
                     std::to_string(cfg.order) + ") must be >= 2"};
  }
  if (!(cfg.window_length > Rational{0})) {
    return Failure{"E-RT-WINDOW-INTERVAL",
                 "WindowCoupler::create: cfg.window_length (" +
                     cfg.window_length.to_string() + ") must be positive"};
  }
  for (std::size_t i = 0; i < members.size(); ++i) {
    const WindowMember& m = members[i];
    if (m.component == nullptr) {
      return Failure{"E-RT-WINDOW-NULL",
                   "WindowCoupler::create: member " + std::to_string(i) +
                       " ('" + m.name + "') has a null component"};
    }
    if (m.length > 0 && (m.provided == nullptr || m.required == nullptr)) {
      return Failure{"E-RT-WINDOW-NULL",
                   "WindowCoupler::create: member '" + m.name +
                       "' has length " + std::to_string(m.length) +
                       " but a null provided/required pointer"};
    }
    if (!(m.interval > Rational{0})) {
      return Failure{"E-RT-WINDOW-INTERVAL",
                   "WindowCoupler::create: member '" + m.name +
                       "' has a non-positive interval (" +
                       m.interval.to_string() + ")"};
    }
    if (m.order != 0 && m.order < 2) {
      return Failure{"E-RT-WINDOW-ORDER",
                   "WindowCoupler::create: member '" + m.name +
                       "' has order " + std::to_string(m.order) +
                       ", which must be 0 (use cfg.order) or >= 2"};
    }
    if (m.policy == DtPolicy::Fixed && !m.component->supports_rollback()) {
      return Failure{"E-RT-WINDOW-ROLLBACK",
                   "WindowCoupler::create: member '" + m.name +
                       "' is Fixed, which requires rolling back and "
                       "recomputing a straddling step every window; its "
                       "component's supports_rollback() is false"};
    }
  }

  // The verified envelope (KNOWN_ISSUES.md D7). A member straddles a window
  // end only if it is Fixed and its step does not divide the window. The
  // replay handles ONE such member per window: two replay against each
  // other's unresolved trajectories, and a Fixed step longer than the window
  // spans a whole window it never samples. Outside the envelope the result
  // was measured to be wrong without any error, so it is refused here rather
  // than run.
  {
    const std::string* first_straddler = nullptr;
    for (const WindowMember& m : members) {
      if (m.policy != DtPolicy::Fixed) continue;
      if (m.interval > cfg.window_length) {
        return Failure{"E-RT-WINDOW-LONGSTEP",
                       "WindowCoupler::create: member '" + m.name +
                           "' is Fixed with interval " + m.interval.to_string() +
                           ", longer than the window (" +
                           cfg.window_length.to_string() +
                           "); a Fixed step may not span a window end it cannot "
                           "replay. Make the member Adjustable, or define the "
                           "window by this member's step"};
      }
      if ((cfg.window_length / m.interval).is_integer()) continue;
      if (first_straddler != nullptr) {
        return Failure{"E-RT-WINDOW-STRADDLE",
                       "WindowCoupler::create: members '" + *first_straddler +
                           "' and '" + m.name +
                           "' are both Fixed with steps that do not divide the "
                           "window (" + cfg.window_length.to_string() +
                           "), so both can straddle one window end; only one "
                           "straddling member per window is supported. Make one "
                           "of them Adjustable, or choose a window both steps "
                           "divide"};
      }
      first_straddler = &m.name;
    }
  }
  for (std::size_t c = 0; c < channels.size(); ++c) {
    if (channels[c].provider >= members.size() ||
        channels[c].consumer >= members.size()) {
      return Failure{"E-RT-WINDOW-CHANNEL",
                   "WindowCoupler::create: channel " + std::to_string(c) +
                       " names a member index outside [0, " +
                       std::to_string(members.size()) + ")"};
    }
  }

  if (cfg.iteration == WaveformIteration::GaussSeidel) {
    // A member providing both a lagged and an unlagged outgoing channel:
    // `delivered[provider]` is one waveform per provider, and cannot be
    // simultaneously this sweep's fresh value (what the unlagged
    // consumer needs) and last sweep's (what the lagged one needs).
    for (std::size_t i = 0; i < members.size(); ++i) {
      bool has_lagged = false, has_unlagged = false;
      for (const WindowChannel& ch : channels) {
        if (ch.provider != i) continue;
        (ch.lagged ? has_lagged : has_unlagged) = true;
      }
      if (has_lagged && has_unlagged) {
        return Failure{"E-RT-WINDOW-GAUSSSEIDEL-MIXED-LAG",
                     "WindowCoupler::create: member '" + members[i].name +
                         "' provides both a lagged and an unlagged "
                         "outgoing channel; WaveformIteration::GaussSeidel "
                         "needs one delivered waveform per provider, which "
                         "cannot represent both at once"};
      }
    }
    if (!topological_sweep_order(members.size(), channels).has_value()) {
      return Failure{"E-RT-WINDOW-GAUSSSEIDEL-NOLAG",
                   "WindowCoupler::create: the directed graph of UNLAGGED "
                       "channels has a cycle (every channel unlagged is "
                       "the sharpest case: no lag at all breaks it). No "
                       "sweep order can run every unlagged provider before "
                       "its consumer. Lag at least one direction, or use "
                       "WaveformIteration::Jacobi instead, which needs no "
                       ".lag() at all"};
    }
  }

  return std::unique_ptr<WindowCoupler>(new WindowCoupler(
      std::move(members), std::move(channels), std::move(cfg)));
}

Result<void> WindowCoupler::resolve_pending_straddles(Rational /*T0*/) {
  // A `Fixed` member that overshot the previous window replays EVERY
  // substep it took since its last checkpoint (still `restore_state()`-able:
  // nothing has saved over it since -- see the class comment on why this,
  // not a second checkpoint, is how the single-checkpoint contract is kept),
  // now reading the previous window's CONVERGED data for the in-window part
  // and an extrapolation for whatever ran past it.
  for (std::size_t i = 0; i < members_.size(); ++i) {
    MemberState& st = state_[i];
    if (!st.straddle_pending) continue;
    const WindowMember& m = members_[i];
    const Rational prev_end = st.straddle_window_end;

    const Result<void> r0 = m.component->restore_state();
    if (!r0.ok()) return r0;

    // `sweep_window()` already committed `history[e]`'s last entry --
    // {prev_end, v_at_t1} -- as the EXACT value at prev_end from a genuine
    // advance() split at that boundary (not an interpolant; see the class
    // comment on why the split, not a blended mean_val plus interpolation,
    // is what makes a straddling substep's boundary value as accurate as
    // an ordinary substep's own endpoint). This replay must reproduce that
    // SAME split at `prev_end` for its own crossing chunk, for the same
    // reason -- and once it does, it recomputes the SAME accurate value
    // sweep_window already committed (both use the same converged
    // in-window data for [chunk_start, prev_end)), so `history` needs no
    // separate correction here.
    Rational clock = st.straddle_window_start;
    while (clock < st.clock) {
      const Rational end = std::min(clock + m.interval, st.clock);
      if (clock < prev_end && prev_end < end) {
        // This chunk straddles prev_end: split it exactly there, same as
        // sweep_window's straddling substep, and via the SAME shared
        // choreography (`detail::advance_straddle_split()`, Unit A, step
        // 3) -- here reading the previous window's converged
        // `last_window_waveform`/`history` instead of this sweep's live
        // `delivered[]`. The split value itself is never used here (the
        // replay's whole point is to reproduce the component's state, not
        // to record it anywhere new), so `out_at_split` is a throwaway
        // scratch buffer.
        std::vector<double> scratch(m.length);
        const Result<void> split_r = detail::advance_straddle_split(
            m, clock, prev_end, end, window_index_, 0,
            [&](std::size_t e, Rational a, Rational b) -> Result<double> {
              double mean_in = m.required[e];
              for (const WindowChannel& ch : channels_) {
                if (ch.consumer != i) continue;
                const MemberState& pst = state_[ch.provider];
                const Result<double> mv =
                    channel_io_->mean(ch, e, pst.last_window_waveform, a, b);
                if (!mv.ok()) return mv.error();
                mean_in = mv.value();
              }
              return mean_in;
            },
            [&](std::size_t e) -> Result<double> {
              double v_bnd = m.required[e];
              for (const WindowChannel& ch : channels_) {
                if (ch.consumer != i) continue;
                const MemberState& pst = state_[ch.provider];
                const Result<double> bv = channel_io_->boundary(
                    ch, e, pst.last_window_waveform, prev_end);
                if (!bv.ok()) return bv.error();
                v_bnd = bv.value();
              }
              return v_bnd;
            },
            [&](std::size_t e, Rational target) -> double {
              double v_tail = m.required[e];
              for (const WindowChannel& ch : channels_) {
                if (ch.consumer != i) continue;
                const MemberState& pst = state_[ch.provider];
                v_tail = channel_io_->extrapolate(ch, e, pst.history, target,
                                                  order_for(ch.provider));
              }
              return v_tail;
            },
            scratch.data());
        if (!split_r.ok()) return split_r;
        clock = end;
        continue;
      }
      for (std::size_t e = 0; e < m.length; ++e) {
        double mean_val = 0.0;
        bool have = false;
        for (const WindowChannel& ch : channels_) {
          if (ch.consumer != i) continue;
          const MemberState& pst = state_[ch.provider];
          const int ord = order_for(ch.provider);
          if (end <= prev_end) {
            const Result<double> mv =
                channel_io_->mean(ch, e, pst.last_window_waveform, clock, end);
            if (!mv.ok()) return mv.error();
            mean_val = mv.value();
          } else {
            const double v0 =
                channel_io_->extrapolate(ch, e, pst.history, clock, ord);
            const double v1 =
                channel_io_->extrapolate(ch, e, pst.history, end, ord);
            mean_val = 0.5 * (v0 + v1);
          }
          have = true;
        }
        if (have) m.required[e] = mean_val;
      }
      StepContext ctx;
      ctx.t_begin = clock;
      ctx.t_end = end;
      ctx.step = window_index_;
      ctx.iteration = 0;
      const Result<void> ar = m.component->advance(ctx);
      if (!ar.ok()) return ar;
      clock = end;
    }
    st.straddle_pending = false;
  }
  return Result<void>::success();
}

Result<void> WindowCoupler::sweep_window(Rational T0, Rational T1) {
  const std::size_t n = members_.size();

  // Captured ONCE, right after save_state(), before any advance() this
  // window can mutate `provided`. `build_delivered()` is called again at
  // the end of every sweep (with the just-accelerated y), and by then
  // `members_[i].provided[e]` holds whatever the LAST advance() call left
  // there -- the value at the end of this member's last substep, not at
  // start_clock[i]. Reading the live pointer for the leading knot there
  // would silently mislabel that end-of-substep value as the window-start
  // value, corrupting every later sweep's delivered trajectory. This
  // capture is the fix: `start_value[i][e]` never changes for the
  // lifetime of this window.
  std::vector<std::vector<double>> start_value(n);
  for (std::size_t i = 0; i < n; ++i) {
    const Result<void> s = members_[i].component->save_state();
    if (!s.ok()) return s;
    start_value[i].resize(members_[i].length);
    for (std::size_t e = 0; e < members_[i].length; ++e) {
      start_value[i][e] = members_[i].provided[e];
    }
  }

  // The substep grid is fixed for the whole window (it depends only on
  // each member's own clock/interval/policy, never on sweep results), so it
  // is planned once, outside the sweep loop.
  std::vector<std::vector<Substep>> steps(n);
  std::vector<Rational>             new_clock(n);
  std::vector<Rational>             start_clock(n);
  for (std::size_t i = 0; i < n; ++i) {
    start_clock[i] = state_[i].clock;
    auto [sp, nc] = plan_substeps(state_[i].clock, members_[i].interval, T1,
                                  members_[i].policy);
    steps[i] = std::move(sp);
    new_clock[i] = nc;
  }

  // Canonical flat layout of the exchanged/accelerated vector: one entry per
  // (member, element, substep), in that nesting order. `clipped_time(i,s)`
  // is the time the exchanged sample stands for -- the substep's own end,
  // clipped to the window (a straddling substep's provisional true end lies
  // past T1; consumers only ever need the value AT T1).
  std::size_t total = 0;
  for (std::size_t i = 0; i < n; ++i) total += members_[i].length * steps[i].size();

  std::vector<double> y_fed(total), y_new(total);

  // Seed y_fed (sweep 0's "what was fed") from each member's committed
  // history, extrapolated to every one of its own clipped sample times.
  {
    std::size_t idx = 0;
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t e = 0; e < members_[i].length; ++e) {
        for (const Substep& sub : steps[i]) {
          const Rational tc = std::min(sub.t1, T1);
          y_fed[idx++] = extrapolate_boundary_value(state_[i].history[e], tc,
                                                     order_for(i));
        }
      }
    }
  }

  // Member i's own flat (element, substep) sub-range always starts at the
  // same offset within a canonical-layout `y`, whatever sweep or mode is
  // running: the SAME nesting order (member, element, substep) that both
  // `build_delivered_member()` below and the main sweep loop's own `base`
  // walk. Precomputed once so a GaussSeidel mid-sweep partial rebuild
  // (member i alone, from `y_new`, the moment i's own substep loop
  // finishes) reads exactly the slice the next FULL rebuild would have
  // read for i -- one offset table, so the two cannot disagree.
  std::vector<std::size_t> member_base(n, 0);
  {
    std::size_t acc = 0;
    for (std::size_t i = 0; i < n; ++i) {
      member_base[i] = acc;
      acc += members_[i].length * steps[i].size();
    }
  }

  // Waveform built from y_fed/y_new for ONE member's (element, substep)
  // slice, spanning [start_clock[i], T1] -- what a consumer reads this
  // sweep. Shared by build_delivered() (called once per member, for
  // every member, from y_fed or the just-accelerated y_out) and
  // WaveformIteration::GaussSeidel's own mid-sweep refresh (called once,
  // for the one member that just finished its substep loop, from y_new).
  const auto build_delivered_member = [&](std::size_t i,
                                          const std::vector<double>& y)
      -> std::vector<std::unique_ptr<Waveform>> {
    std::vector<std::unique_ptr<Waveform>> wf_i(members_[i].length);
    std::size_t idx = member_base[i];
    for (std::size_t e = 0; e < members_[i].length; ++e) {
      wf_i[e] = std::make_unique<Waveform>(
          1, cfg_.waveform_capacity, TemporalBasis::Linear, 1);
      // A Fixed member's start_clock[i] can already be past T0 here (it
      // committed past the previous window's boundary via a straddling
      // step whose overshoot resolve_pending_straddles() has since
      // replayed and corrected). A consumer whose own substep starts at
      // T0 -- the window-definer's single full-window substep, in
      // particular -- still needs this member's value there. The value
      // this member committed at the end of the PREVIOUS window is
      // exactly that: history's last entry sits at T1(prev) == T0(this
      // window) by construction, and needs no extrapolation.
      const Rational lo = std::min(start_clock[i], T0);
      wf_i[e]->reset(lo, T1);
      if (start_clock[i] > T0) {
        double v_t0 = state_[i].history[e].back().second;
        (void)wf_i[e]->push(T0, &v_t0);
      }
      double v0 = start_value[i][e];
      (void)wf_i[e]->push(start_clock[i], &v0);
      for (const Substep& sub : steps[i]) {
        const Rational tc = std::min(sub.t1, T1);
        double v = y[idx++];
        if (tc > wf_i[e]->window_end()) continue;  // defensive; cannot happen
        (void)wf_i[e]->push(tc, &v);
      }
    }
    return wf_i;
  };

  const auto build_delivered = [&](const std::vector<double>& y)
      -> std::vector<std::vector<std::unique_ptr<Waveform>>> {
    std::vector<std::vector<std::unique_ptr<Waveform>>> wf(n);
    for (std::size_t i = 0; i < n; ++i) wf[i] = build_delivered_member(i, y);
    return wf;
  };

  // WaveformIteration::GaussSeidel only: whether member i provides at
  // least one UNLAGGED channel. Such a member's own outgoing waveform is
  // refreshed the moment its substep loop finishes THIS sweep (so a
  // LATER member in the SAME sweep sees it fresh, not last sweep's
  // value) and is excluded from the accelerated vector (its value is
  // already exact -- Gauss-Seidel substitution, not a relaxed iterate).
  // A member with no outgoing channels, or only LAGGED ones, is
  // accelerated exactly as every member is under Jacobi.
  const auto provides_unlagged_channel = [&](std::size_t i) {
    for (const WindowChannel& ch : channels_) {
      if (ch.provider == i && !ch.lagged) return true;
    }
    return false;
  };

  // The members whose own outgoing waveform is part of the accelerated
  // vector: every member under Jacobi (today's behaviour, unchanged);
  // under GaussSeidel, only those that do not provide an unlagged
  // channel -- "only the lagged direction's waveform" (this track's own
  // design note on acceleration). `accel_member_offset[i]` is member i's
  // own slice's start within the PROJECTED vector, valid only for `i` in
  // `accel_members`; `member_base[i]` above is its start within the
  // FULL one -- the projection below copies between the two.
  std::vector<std::size_t> accel_members;
  std::vector<std::size_t> accel_member_offset(n, 0);
  std::size_t accel_total = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (cfg_.iteration == WaveformIteration::Jacobi ||
        !provides_unlagged_channel(i)) {
      accel_members.push_back(i);
      accel_member_offset[i] = accel_total;
      accel_total += members_[i].length * steps[i].size();
    }
  }

  std::vector<std::vector<std::unique_ptr<Waveform>>> real_traj(n);  // this sweep's OWN advance() output
  stats_ = WindowIterationStats{};

  auto delivered = build_delivered(y_fed);

  int k = 0;
  for (; k < cfg_.max_iterations; ++k) {
    for (std::size_t i = 0; i < n; ++i) {
      const Result<void> rs = members_[i].component->restore_state();
      if (!rs.ok()) return rs;
    }

    real_traj.clear();
    real_traj.resize(n);
    // Every write below goes to the canonical (member, element, substep)
    // index, computed, never to a running counter. The substep loop visits
    // (substep, element); a counter there wrote the transposed layout, which
    // agrees with the canonical one only for a scalar member or a single
    // substep -- the only cases tested until a field-valued multirate
    // member (Gray-Scott) delivered a "mean" outside the partner's range.
    //
    // `sweep_order_[pos]`, not `pos` itself: under Jacobi this IS the
    // identity permutation (declaration order, unchanged from before this
    // member existed); under GaussSeidel it is the topological order the
    // unlagged channels force (an unlagged provider before its consumer),
    // computed once at construction (WindowCoupler::WindowCoupler(),
    // via the SAME topological_sweep_order() WindowCoupler::create() uses
    // to validate the graph is acyclic in the first place). `base` reads
    // `member_base[i]` -- member i's own flat offset, precomputed above in
    // DECLARATION-index order, independent of sweep order -- rather than
    // accumulating incrementally in loop order, which would silently
    // transpose the layout the moment sweep order differs from declaration
    // order.
    for (std::size_t pos = 0; pos < n; ++pos) {
      const std::size_t i = sweep_order_[pos];
      const std::size_t base = member_base[i];
      const WindowMember& m = members_[i];
      const std::size_t n_sub = steps[i].size();
      const auto at = [&](std::size_t e, std::size_t s) { return base + e * n_sub + s; };
      std::size_t s_idx = 0;
      real_traj[i].resize(m.length);
      for (std::size_t e = 0; e < m.length; ++e) {
        real_traj[i][e] = std::make_unique<Waveform>(
            1, cfg_.waveform_capacity, TemporalBasis::Linear, 1);
        real_traj[i][e]->reset(start_clock[i], steps[i].empty() ? T1 : steps[i].back().t1);
        double v0 = m.provided[e];
        (void)real_traj[i][e]->push(start_clock[i], &v0);
      }

      for (const Substep& sub : steps[i]) {
        if (sub.straddle && T1 > sub.t0) {
          // Split the advance() exactly AT T1 instead of taking one
          // advance() over the whole [sub.t0, sub.t1) with a single
          // BLENDED mean and then interpolating the value AT T1 between
          // the two endpoint STATES. That interpolation does not recover
          // x(T1) to the substep's own order: x(T1) depends on the TRUE
          // mean over [sub.t0, T1) specifically, and a state computed by
          // advancing with a mean valid for the LONGER, blended interval
          // does not sit on the same trajectory between t0 and T1 that a
          // direct advance() with mean_in would produce. Splitting the
          // call -- advance to T1 with mean_in (exactly the non-straddle
          // formula, restricted to [sub.t0, T1)), then continue to
          // sub.t1 with mean_tail -- makes the committed T1 value as
          // accurate as an ordinary substep's own endpoint, which is what
          // this member's Adjustable counterpart already gets for free
          // (its last substep simply ends at T1). This is also what the
          // validated Python reference implements (run_substeps3's two
          // advance() calls at the straddle boundary); the single-call
          // interpolation was a divergence from it introduced when this
          // was ported to C++.
          // Unit A, step 3: the split-advance choreography itself (advance
          // to the split point with the in-window mean, record the exact
          // split value, then -- unless the split point IS the chunk's own
          // end -- advance the tail with the blended boundary/extrapolate
          // mean) is shared with `resolve_pending_straddles()`'s replay via
          // `detail::advance_straddle_split()`; only the callbacks below
          // (which waveform/history they read) are specific to this sweep.
          std::vector<double> v_at_t1(m.length);
          const Result<void> split_r = detail::advance_straddle_split(
              members_[i], sub.t0, T1, sub.t1, window_index_, k,
              [&](std::size_t e, Rational a, Rational b) -> Result<double> {
                double mean_in = 0.0;
                for (const WindowChannel& ch : channels_) {
                  if (ch.consumer != i) continue;
                  const Result<double> mv =
                      channel_io_->mean(ch, e, delivered[ch.provider], a, b);
                  if (!mv.ok()) return mv.error();
                  mean_in = mv.value();
                }
                return mean_in;
              },
              [&](std::size_t e) -> Result<double> {
                double v_bnd = 0.0;
                // (void): this call site never checked evaluate()'s Result
                // before channel_io_ existed either -- preserved exactly,
                // not a new omission.
                for (const WindowChannel& ch : channels_) {
                  if (ch.consumer != i) continue;
                  const Result<double> bv =
                      channel_io_->boundary(ch, e, delivered[ch.provider], T1);
                  if (bv.ok()) v_bnd = bv.value();
                }
                return v_bnd;
              },
              [&](std::size_t e, Rational target) -> double {
                double v_tail = 0.0;
                for (const WindowChannel& ch : channels_) {
                  if (ch.consumer != i) continue;
                  v_tail = channel_io_->extrapolate(
                      ch, e, state_[ch.provider].history, target,
                      order_for(ch.provider));
                }
                return v_tail;
              },
              v_at_t1.data());
          if (!split_r.ok()) return split_r;
          for (std::size_t e = 0; e < m.length; ++e) {
            const Result<void> pr = real_traj[i][e]->push(T1, &v_at_t1[e]);
            if (!pr.ok()) return pr.error();
          }
          if (T1 < sub.t1) {
            for (std::size_t e = 0; e < m.length; ++e) {
              double v = m.provided[e];
              const Result<void> pr = real_traj[i][e]->push(sub.t1, &v);
              if (!pr.ok()) return pr.error();
            }
          }
          for (std::size_t e = 0; e < m.length; ++e) {
            // The committed/exchanged value for this substep is the EXACT
            // value at T1 from the first advance() call above, never an
            // interpolant -- this is the whole point of the split.
            y_new[at(e, s_idx)] = v_at_t1[e];
          }
          ++s_idx;
          continue;
        }
        for (std::size_t e = 0; e < m.length; ++e) {
          double mean_val = m.required[e];
          for (const WindowChannel& ch : channels_) {
            if (ch.consumer != i) continue;
            const Result<double> mv = channel_io_->mean(
                ch, e, delivered[ch.provider], sub.t0, sub.t1);
            if (!mv.ok()) return mv.error();
            mean_val = mv.value();
          }
          m.required[e] = mean_val;
        }
        StepContext ctx;
        ctx.t_begin = sub.t0;
        ctx.t_end = sub.t1;
        ctx.step = window_index_;
        ctx.iteration = k;
        const Result<void> ar = members_[i].component->advance(ctx);
        if (!ar.ok()) return ar;
        // This substep is never a straddle here (the straddling case
        // `continue`s past this point above): tc == sub.t1 always, so the
        // committed value is simply this substep's own endpoint, no
        // interpolation involved.
        for (std::size_t e = 0; e < m.length; ++e) {
          double v = m.provided[e];
          const Result<void> pr = real_traj[i][e]->push(sub.t1, &v);
          if (!pr.ok()) return pr.error();
          y_new[at(e, s_idx)] = v;
        }
        ++s_idx;
      }

      // WaveformIteration::GaussSeidel: member i's own outgoing waveform,
      // refreshed from what it JUST computed (y_new's now-fully-written
      // slice for i), so a member swept LATER in this SAME sweep sees it
      // -- the one thing that distinguishes this from Jacobi, whose
      // `delivered` never changes within a sweep. `y_new`'s slice for
      // every OTHER member may still hold an earlier sweep's values at
      // this point; build_delivered_member(i, y_new) reads only member
      // i's own slice, so that is never read here.
      if (cfg_.iteration == WaveformIteration::GaussSeidel &&
          provides_unlagged_channel(i)) {
        delivered[i] = build_delivered_member(i, y_new);
      }
    }

    double resid_sq = 0.0, ref_sq = 0.0;
    for (std::size_t j = 0; j < total; ++j) {
      const double d = y_new[j] - y_fed[j];
      resid_sq += d * d;
      ref_sq += y_new[j] * y_new[j];
    }
    const double resid_norm = std::sqrt(resid_sq);
    const double ref_norm = std::sqrt(ref_sq);
    stats_.residual_history.push_back(resid_norm);
    stats_.iterations = k + 1;

    bool converged;
    if (cfg_.convergence.empty()) {
      converged = resid_norm <= 1e-10 * std::max(1.0, ref_norm);
    } else {
      const std::vector<double> res(cfg_.convergence.size(), resid_norm);
      const std::vector<double> ref(cfg_.convergence.size(), ref_norm);
      converged = cfg_.convergence.all_satisfied(res, ref);
    }
    if (converged) {
      stats_.converged = true;
      break;
    }
    if (k + 1 == cfg_.max_iterations) break;

    std::vector<double> y_out(total);
    if (cfg_.iteration == WaveformIteration::GaussSeidel) {
      // Project y_fed/y_new down to the accelerated members' own
      // slices, accelerate exactly that (never the unlagged-providing
      // members' entries), then scatter back. An excluded member's own
      // slice of y_out is a straight copy from y_new -- Gauss-Seidel
      // substitution, not a relaxed iterate: its value is already
      // exact given what every other member just computed, the same
      // reasoning that makes the mid-sweep `delivered` refresh above
      // read from y_new directly rather than through the accelerator.
      std::vector<double> y_fed_proj(accel_total), y_new_proj(accel_total),
          y_out_proj(accel_total);
      for (const std::size_t i : accel_members) {
        const std::size_t len = members_[i].length * steps[i].size();
        std::copy(y_fed.begin() + static_cast<std::ptrdiff_t>(member_base[i]),
                  y_fed.begin() + static_cast<std::ptrdiff_t>(member_base[i] + len),
                  y_fed_proj.begin() + static_cast<std::ptrdiff_t>(accel_member_offset[i]));
        std::copy(y_new.begin() + static_cast<std::ptrdiff_t>(member_base[i]),
                  y_new.begin() + static_cast<std::ptrdiff_t>(member_base[i] + len),
                  y_new_proj.begin() + static_cast<std::ptrdiff_t>(accel_member_offset[i]));
      }
      const Result<void> up = accel_->update(y_fed_proj.data(), y_new_proj.data(),
                                             y_out_proj.data(), accel_total, k);
      if (!up.ok()) return up.error();
      y_out = y_new;  // default: excluded members pass straight through
      for (const std::size_t i : accel_members) {
        const std::size_t len = members_[i].length * steps[i].size();
        std::copy(y_out_proj.begin() + static_cast<std::ptrdiff_t>(accel_member_offset[i]),
                  y_out_proj.begin() + static_cast<std::ptrdiff_t>(accel_member_offset[i] + len),
                  y_out.begin() + static_cast<std::ptrdiff_t>(member_base[i]));
      }
    } else {
      const Result<void> up = accel_->update(y_fed.data(), y_new.data(),
                                             y_out.data(), total, k);
      if (!up.ok()) return up.error();
    }
    y_fed = y_out;
    delivered = build_delivered(y_fed);
  }

  if (!stats_.converged) {
    return Failure{"E-RT-WINDOW-NOCONVERGE",
                 "coupling window [" + T0.to_string() + ", " + T1.to_string() +
                     ") ran its full " + std::to_string(cfg_.max_iterations) +
                     " iterations without meeting its convergence criterion"};
  }

  // Commit: the component state IS this sweep's real advance() output
  // already; only the driver's own bookkeeping needs updating.
  std::size_t idx = 0;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t e = 0; e < members_[i].length; ++e) {
      // The LAST substep's clipped (window-end) sample is this window's
      // committed boundary value. A member can have ZERO planned
      // substeps this window (reachable when a prior straddle already
      // carried its clock at or past this window's own end) -- in that
      // case there is no new y_new entry to read, and the value this
      // window commits is simply whatever it committed last: carry
      // forward history[e]'s current last entry rather than an
      // uninitialized 0.0, which would otherwise silently overwrite a
      // real carried-forward value with an artificial zero that
      // extrapolate_boundary_value() would then read as if it were
      // genuine data.
      double last_val = state_[i].history[e].empty()
                             ? members_[i].provided[e]
                             : state_[i].history[e].back().second;
      for (std::size_t s = 0; s < steps[i].size(); ++s) last_val = y_new[idx++];
      state_[i].history[e].push_back({T1, last_val});
    }
    state_[i].last_window_waveform = std::move(real_traj[i]);
    state_[i].clock = new_clock[i];
    if (!steps[i].empty() && steps[i].back().straddle) {
      state_[i].straddle_pending = true;
      // The replay's restore_state() returns to whatever save_state()
      // above actually checkpointed -- this member's clock at the TOP of
      // THIS window, i.e. start_clock[i], which is T0 only the first time
      // this member ever straddles. A member that is ALREADY overshooting
      // every window (the 7:13 case) has start_clock[i] > T0 by the
      // previous window's own overshoot; recording T0 here made the next
      // window's resolve_pending_straddles() replay substeps as if they
      // began at T0, one component's worth of true elapsed time short of
      // where restore_state() actually put it -- a per-window
      // mislabelling whose size never shrinks with the window length
      // (both T0 and start_clock[i] shrink together, but their GAP is a
      // roughly fixed fraction of the window), which is exactly why the
      // resulting error was O(1) rather than O(h^2).
      state_[i].straddle_window_start = start_clock[i];
      state_[i].straddle_window_end = T1;
    } else {
      state_[i].straddle_pending = false;
    }
  }
  return Result<void>::success();
}

Result<void> WindowCoupler::run_window() {
  const Rational T0 = cfg_.window_length * Rational{window_index_};
  const Rational T1 = T0 + cfg_.window_length;

  if (accel_ == nullptr) {
    Result<std::unique_ptr<Accelerator>> a = make_accelerator(cfg_.acceleration);
    if (!a.ok()) return a.error();
    accel_ = std::move(a.value());
  }
  accel_->reset();

  const Result<void> f = resolve_pending_straddles(T0);
  if (!f.ok()) return f;

  const Result<void> s = sweep_window(T0, T1);
  if (!s.ok()) return s;

  ++window_index_;
  return Result<void>::success();
}

}  // namespace fuse
