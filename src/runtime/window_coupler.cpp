#include "fuse/runtime/window_coupler.hpp"

#include <algorithm>
#include <cmath>

#include "fuse/num/temporal_basis.hpp"

namespace fuse {

namespace {

/// One substep of a member's own grid within a window.
struct Substep {
  Rational t0, t1;
  bool     straddle = false;  ///< true only for the (at most one) last step
};

/// Spec: choose the window so it never waits for the LCM of two intervals.
/// A member's own substeps march on ITS OWN grid, `Adjustable` truncating
/// the last one onto `window_end`, `Fixed` taking it in full even past
/// `window_end` (the straddling step; see the class comment in the header).
[[nodiscard]] std::pair<std::vector<Substep>, Rational> plan_substeps(
    Rational clock, Rational interval, Rational window_end, DtPolicy policy) {
  std::vector<Substep> steps;
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

/// The mean of `wf` over `[a, b)`, `a < b` required. `wf` must already
/// cover `[a, b]` (its support, per `Waveform`'s own contract).
[[nodiscard]] Result<double> mean_over(const Waveform& wf, Rational a,
                                       Rational b) {
  double integral = 0.0;
  const Result<void> r = wf.integrate(a, b, &integral);
  if (!r.ok()) return r.error();
  return integral / (b - a).to_double();
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

WindowCoupler::WindowCoupler(std::vector<WindowMember> members,
                             std::vector<WindowChannel> channels,
                             WindowCouplerConfig cfg)
    : members_(std::move(members)),
      channels_(std::move(channels)),
      cfg_(std::move(cfg)) {
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
        // sweep_window's straddling substep.
        for (std::size_t e = 0; e < m.length; ++e) {
          double mean_in = 0.0;
          bool have = false;
          for (const WindowChannel& ch : channels_) {
            if (ch.consumer != i) continue;
            const MemberState& pst = state_[ch.provider];
            const std::size_t pe = std::min(e, members_[ch.provider].length - 1);
            const Result<double> mv =
                mean_over(*pst.last_window_waveform[pe], clock, prev_end);
            if (!mv.ok()) return mv.error();
            mean_in = mv.value();
            have = true;
          }
          if (have) m.required[e] = mean_in;
        }
        StepContext ctx_in;
        ctx_in.t_begin = clock;
        ctx_in.t_end = prev_end;
        ctx_in.step = window_index_;
        ctx_in.iteration = 0;
        const Result<void> ar_in = m.component->advance(ctx_in);
        if (!ar_in.ok()) return ar_in;

        for (std::size_t e = 0; e < m.length; ++e) {
          double mean_tail = 0.0;
          bool have = false;
          for (const WindowChannel& ch : channels_) {
            if (ch.consumer != i) continue;
            const MemberState& pst = state_[ch.provider];
            const std::size_t pe = std::min(e, members_[ch.provider].length - 1);
            const int ord = order_for(ch.provider);
            double v_bnd = 0.0;
            const Result<void> ev =
                pst.last_window_waveform[pe]->evaluate(prev_end, &v_bnd);
            if (!ev.ok()) return ev.error();
            const double v_tail = extrapolate_boundary_value(pst.history[pe], end, ord);
            mean_tail = 0.5 * (v_bnd + v_tail);
            have = true;
          }
          if (have) m.required[e] = mean_tail;
        }
        StepContext ctx_tail;
        ctx_tail.t_begin = prev_end;
        ctx_tail.t_end = end;
        ctx_tail.step = window_index_;
        ctx_tail.iteration = 0;
        const Result<void> ar_tail = m.component->advance(ctx_tail);
        if (!ar_tail.ok()) return ar_tail;
        clock = end;
        continue;
      }
      for (std::size_t e = 0; e < m.length; ++e) {
        double mean_val = 0.0;
        bool have = false;
        for (const WindowChannel& ch : channels_) {
          if (ch.consumer != i) continue;
          const MemberState& pst = state_[ch.provider];
          const std::size_t pe = std::min(e, members_[ch.provider].length - 1);
          const int ord = order_for(ch.provider);
          if (end <= prev_end) {
            const Result<double> mv =
                mean_over(*pst.last_window_waveform[pe], clock, end);
            if (!mv.ok()) return mv.error();
            mean_val = mv.value();
          } else {
            const double v0 = extrapolate_boundary_value(pst.history[pe], clock, ord);
            const double v1 = extrapolate_boundary_value(pst.history[pe], end, ord);
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

  // Waveform built from y_fed/y_new for each (member, element), spanning
  // [start_clock[i], T1] -- what a consumer reads this sweep.
  const auto build_delivered = [&](const std::vector<double>& y)
      -> std::vector<std::vector<std::unique_ptr<Waveform>>> {
    std::vector<std::vector<std::unique_ptr<Waveform>>> wf(n);
    std::size_t idx = 0;
    for (std::size_t i = 0; i < n; ++i) {
      wf[i].resize(members_[i].length);
      for (std::size_t e = 0; e < members_[i].length; ++e) {
        wf[i][e] = std::make_unique<Waveform>(
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
        wf[i][e]->reset(lo, T1);
        if (start_clock[i] > T0) {
          double v_t0 = state_[i].history[e].back().second;
          (void)wf[i][e]->push(T0, &v_t0);
        }
        double v0 = start_value[i][e];
        (void)wf[i][e]->push(start_clock[i], &v0);
        for (const Substep& sub : steps[i]) {
          const Rational tc = std::min(sub.t1, T1);
          double v = y[idx++];
          if (tc > wf[i][e]->window_end()) continue;  // defensive; cannot happen
          (void)wf[i][e]->push(tc, &v);
        }
      }
    }
    return wf;
  };

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
    std::size_t base = 0;
    for (std::size_t i = 0; i < n; ++i) {
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
          for (std::size_t e = 0; e < m.length; ++e) {
            double mean_in = 0.0;
            for (const WindowChannel& ch : channels_) {
              if (ch.consumer != i) continue;
              const std::size_t pe = std::min(e, members_[ch.provider].length - 1);
              const Waveform& pw = *delivered[ch.provider][pe];
              const Result<double> mv = mean_over(pw, sub.t0, T1);
              if (!mv.ok()) return mv.error();
              mean_in = mv.value();
            }
            m.required[e] = mean_in;
          }
          StepContext ctx_in;
          ctx_in.t_begin = sub.t0;
          ctx_in.t_end = T1;
          ctx_in.step = window_index_;
          ctx_in.iteration = k;
          const Result<void> ar_in = members_[i].component->advance(ctx_in);
          if (!ar_in.ok()) return ar_in;
          std::vector<double> v_at_t1(m.length);
          for (std::size_t e = 0; e < m.length; ++e) {
            v_at_t1[e] = m.provided[e];
            const Result<void> pr = real_traj[i][e]->push(T1, &v_at_t1[e]);
            if (!pr.ok()) return pr.error();
          }
          if (T1 < sub.t1) {
            for (std::size_t e = 0; e < m.length; ++e) {
              double mean_tail = 0.0;
              for (const WindowChannel& ch : channels_) {
                if (ch.consumer != i) continue;
                const std::size_t pe = std::min(e, members_[ch.provider].length - 1);
                const Waveform& pw = *delivered[ch.provider][pe];
                double v_bnd = 0.0;
                (void)pw.evaluate(T1, &v_bnd);
                const double v_tail = extrapolate_boundary_value(
                    state_[ch.provider].history[pe], sub.t1, order_for(ch.provider));
                mean_tail = 0.5 * (v_bnd + v_tail);
              }
              m.required[e] = mean_tail;
            }
            StepContext ctx_tail;
            ctx_tail.t_begin = T1;
            ctx_tail.t_end = sub.t1;
            ctx_tail.step = window_index_;
            ctx_tail.iteration = k;
            const Result<void> ar_tail = members_[i].component->advance(ctx_tail);
            if (!ar_tail.ok()) return ar_tail;
          }
          for (std::size_t e = 0; e < m.length; ++e) {
            if (T1 < sub.t1) {
              double v = m.provided[e];
              const Result<void> pr = real_traj[i][e]->push(sub.t1, &v);
              if (!pr.ok()) return pr.error();
            }
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
            const std::size_t pe = std::min(e, members_[ch.provider].length - 1);
            const Waveform& pw = *delivered[ch.provider][pe];
            const Result<double> mv = mean_over(pw, sub.t0, sub.t1);
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
      base += m.length * n_sub;
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
    const Result<void> up = accel_->update(y_fed.data(), y_new.data(),
                                           y_out.data(), total, k);
    if (!up.ok()) return up.error();
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
