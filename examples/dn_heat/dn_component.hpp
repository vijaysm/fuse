#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include <fuse/runtime/component.hpp>

/// The two participants of Demo A (spec §11) at the Phase 7 milestone: a
/// Dirichlet-Neumann pair of heat conduction domains, coupled implicitly.
///
/// THE PROBLEM, and why its error is the COUPLER'S and nobody else's.
///
/// Two slabs meet at a common interface. Slab 1 occupies x in [-L1, 0] and
/// slab 2 occupies x in [0, L2]; the interface x = 0 is discretised as a line
/// of `n` stations in y. Conduction is in x only -- the slabs are laterally
/// insulated between stations, the standard thermal-resistance-network model
/// of a composite wall -- so each station is an independent 1-D two-layer wall
/// and every station can be written down in closed form.
///
/// The outer faces are held at prescribed, time-varying temperatures T1(t) and
/// T2(t). Forcing is slow compared with the diffusion time, so each slab is in
/// steady state at every instant (the quasi-steady approximation): the profile
/// through each slab is a straight line and both participants evaluate it
/// EXACTLY. Neither has a mesh, a time integrator, or a linear solve, and
/// therefore neither has a discretisation error. Whatever the coupled answer
/// misses the analytic two-layer-wall solution by came from the coupling, and
/// only from the coupling. That is the whole point: Task 4.7's example
/// measured a coupler against a component carrying its own O(dx^2) + O(dt)
/// error, which is a measurement of the component.
///
/// THE SPLIT (classical Dirichlet-Neumann):
///
///   DirichletSlab  gets the interface temperature u, imposes it as a
///                  Dirichlet condition at x = 0, and reports the heat flux it
///                  takes to hold it:
///                      q(y) = k1(y) * (T1(t) - u(y)) / L1      [W/m^2, +x]
///
///   NeumannSlab    gets that flux, imposes it as a Neumann condition at
///                  x = 0, and reports the interface temperature it implies:
///                      u(y) = T2(t) + q(y) * L2 / k2           [K]
///
/// Composing the two gives the fixed-point map u <- T2 + r(y) (T1 - u) with
///
///   r(y) = k1(y) L2 / (k2 L1)
///
/// the classical Dirichlet-Neumann contraction factor: Picard converges at
/// rate r and diverges for r > 1. Its unique fixed point is the series
/// resistance answer,
///
///   u*(y,t) = (T2(t) + r(y) T1(t)) / (1 + r(y))
///   q*(y,t) = (T1(t) - T2(t)) / (L1/k1(y) + L2/k2)
///
/// which is exactly the solution of the whole two-slab problem with continuous
/// temperature and continuous flux across x = 0. Those two functions are the
/// yardstick the demo measures against.
///
/// THE SWEEP IS PARALLEL, NOT SERIAL, and it changes the rate. The textbook
/// Dirichlet-Neumann iteration is a Gauss-Seidel: solve the Dirichlet problem,
/// hand the FRESH flux to the Neumann problem, solve that, repeat. A FUSE
/// implicit coupling group restores every checkpointed participant and re-advances
/// them all from the SAME delivered iterate (src/runtime/iteration.cpp:
/// deliver every block, then run every tick), so both blocks see the previous
/// sweep's values -- a Jacobi iteration on the pair (q, u). Writing
/// dq_k = q_k - q_{k-1} and du_k = u_k - u_{k-1}, one sweep gives
/// dq_{k+1} = -r du_k and du_{k+1} = dq_k, so a block's norm contracts by r
/// only every SECOND sweep and the per-sweep rate is sqrt(r), not r. That is
/// why the printed Picard residual history falls in near-equal pairs, and why
/// Picard costs roughly twice the sweeps a serial Dirichlet-Neumann iteration
/// would. It is a property of the implicit coupling group's execution model, not a defect: the
/// two participants can be advanced concurrently, which a Gauss-Seidel sweep
/// cannot.
///
/// WHY THE CONDUCTIVITY VARIES ALONG THE INTERFACE. With one r the whole
/// coupled operator is a scalar affine map and *any* secant method solves it
/// in two steps, which demonstrates nothing about a vector quasi-Newton
/// scheme. Four conductivity strips (a composite wall built from four
/// materials) make the coupled operator a diagonal map with four distinct
/// eigenvalues, spread from 0.50 to 0.96. Picard then crawls at 0.96, Aitken's
/// single scalar omega cannot annihilate four modes at once, and IQN-ILS --
/// a least-squares fit over the whole concatenated vector -- can, in about as
/// many sweeps as there are distinct modes. The demo prints all three.
///
/// STORAGE. Every array these components register with the coupler is
/// allocated once in the constructor and never resized, reassigned or swapped.
/// `fuse/runtime/component.hpp` requires the registered address to stay fixed
/// for the whole run, and both idioms that break it are silent: `swap()`
/// exchanges heap buffers so the coupler keeps reading the buffer that is now
/// the scratch one, and `v = saved_` may reallocate (and, with an empty
/// `saved_`, resizes the field to zero). save_state()/restore_state() below
/// therefore use std::copy into pre-sized storage, never assignment.
namespace dn {

inline constexpr double kPi = 3.14159265358979323846;

// --- geometry and materials, SI throughout ---------------------------------
// metres, kelvin, W/(m K) for conductivity, W/m^2 for flux.
inline constexpr double kL1 = 1.0;  // slab 1 thickness            [m]
inline constexpr double kL2 = 1.0;  // slab 2 thickness            [m]
inline constexpr double kK2 = 1.0;  // slab 2 conductivity         [W/(m K)]

/// Four material strips along the interface, given directly as the
/// Dirichlet-Neumann contraction factor r = k1 L2 / (k2 L1) they produce.
/// All below 1, so plain Picard converges and its iteration count is a number
/// the demo can quote rather than a divergence.
inline constexpr std::size_t kStrips = 4;
inline constexpr double kStripRatio[kStrips] = {0.50, 0.75, 0.90, 0.96};

/// Which strip station `i` of `n` lies in. Any n >= kStrips works; n a
/// multiple of kStrips gives equal strips.
[[nodiscard]] inline double strip_ratio(std::size_t i, std::size_t n) {
  const std::size_t s = (i * kStrips) / (n == 0 ? 1 : n);
  return kStripRatio[s < kStrips ? s : kStrips - 1];
}

/// Slab 1's conductivity at station i, chosen so that r(y) is exactly
/// kStripRatio[...].
[[nodiscard]] inline double k1_at(std::size_t i, std::size_t n) {
  return strip_ratio(i, n) * kK2 * kL1 / kL2;
}

/// Outer face temperatures [K], on incommensurate coupling cycles so the pair does not
/// simply repeat. T1 lies in [300, 380] and T2 in [265, 295], so
///
///     T1(t) - T2(t) >= 300 - 295 = 5 K > 0   for every t,
///
/// which is deliberate and load-bearing rather than cosmetic: the interface
/// flux is proportional to T1 - T2, and a flux that passed through zero would
/// give the RELATIVE convergence measure on the flux field transfer a vanishing
/// reference norm. convergence.hpp then falls back to comparing the residual
/// absolutely against the same number, silently reinterpreting a dimensionless
/// tolerance as one in W/m^2. Keeping the flux bounded away from zero keeps
/// both measures meaning what they say.
[[nodiscard]] inline double T_outer_1(double t) {
  return 340.0 + 40.0 * std::sin(2.0 * kPi * t);
}
[[nodiscard]] inline double T_outer_2(double t) {
  return 280.0 + 15.0 * std::sin(kPi * t + 0.7);
}

/// The analytic coupled solution: the two-layer wall's interface temperature
/// and interface flux, with temperature and flux both continuous at x = 0.
[[nodiscard]] inline double exact_u(std::size_t i, std::size_t n, double t) {
  const double r = strip_ratio(i, n);
  return (T_outer_2(t) + r * T_outer_1(t)) / (1.0 + r);
}
[[nodiscard]] inline double exact_q(std::size_t i, std::size_t n, double t) {
  return (T_outer_1(t) - T_outer_2(t)) /
         (kL1 / k1_at(i, n) + kL2 / kK2);
}

/// Discrete L2 norm over the interface, which is the unit line split into
/// `v.size()` equal stations.
[[nodiscard]] inline double l2_norm(const std::vector<double>& v) {
  const double dy = 1.0 / static_cast<double>(v.size());
  double       s = 0.0;
  for (const double x : v) s += x * x * dy;
  return std::sqrt(s);
}

/// Discrete L2 error of `v` against `f(i)`.
template <class F>
[[nodiscard]] double l2_error(const std::vector<double>& v, F&& f) {
  const double dy = 1.0 / static_cast<double>(v.size());
  double       s = 0.0;
  for (std::size_t i = 0; i < v.size(); ++i) {
    const double e = v[i] - f(i);
    s += e * e * dy;
  }
  return std::sqrt(s);
}

template <class F>
[[nodiscard]] double max_error(const std::vector<double>& v, F&& f) {
  double m = 0.0;
  for (std::size_t i = 0; i < v.size(); ++i) {
    m = std::max(m, std::abs(v[i] - f(i)));
  }
  return m;
}

/// How a participant's `Requires` storage is wired up.
///
/// `Connected` binds the incoming field to the array advance() reads, which is
/// the only correct wiring. `Severed` binds it to a decoy array of the same
/// size that nothing reads, so the coupler transports, remaps, diagnoses and
/// delivers exactly as before and the value never reaches the physics.
///
/// This is not a hypothetical. It is the bug Task 4.7 shipped -- a participant
/// that was accidentally uncoupled while the run, the plan and the headline
/// error all looked healthy -- reproduced on purpose so the demo can prove its
/// gate detects it. A severed run still CONVERGES, monotonically and in two
/// sweeps, because a constant map has a trivial fixed point; only the
/// comparison against the analytic solution can tell the two apart.
enum class Wiring { Connected, Severed };

/// Slab 1: the Dirichlet solver. Requires the interface temperature, provides
/// the interface heat flux.
class DirichletSlab final : public fuse::Component {
 public:
  DirichletSlab(std::string name, std::size_t n, Wiring wiring)
      : name_(std::move(name)),
        wiring_(wiring),
        n_(n),
        q_(n, 0.0),
        u_in_(n, 0.0),
        decoy_(n, 0.0),
        saved_q_(n, 0.0) {}

  [[nodiscard]] std::string name() const override { return name_; }

  fuse::Result<void> register_fields(fuse::FieldRegistry& reg,
                                     const fuse::CouplingGraph& g) override {
    const auto cid = g.find_component(name_);
    if (!cid.has_value()) {
      return fuse::Failure{"E-EX-COMP", "component '" + name_ + "' not in graph"};
    }
    for (const fuse::FieldId f : g.component(*cid).fields) {
      const bool provides = g.field(f).intent == fuse::Intent::Provides;
      // q_ for the output; the input goes to u_in_ (Connected) or to the
      // decoy (Severed). Never both to q_: the coupler would overwrite the
      // solution with the incoming boundary data on every exchange.
      double* base = provides
                         ? q_.data()
                         : (wiring_ == Wiring::Connected ? u_in_.data()
                                                         : decoy_.data());
      auto r = reg.bind(f, fuse::FieldRef{base,
                                          fuse::Layout{.stride = 1,
                                                       .count = n_},
                                          fuse::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }

  /// A pure function of (restored state, inputs, ctx), as an iterated
  /// component must be. `history_` is the one thing it mutates, and it is
  /// checkpointed.
  fuse::Result<void> advance(const fuse::StepContext& ctx) override {
    ++advance_count;
    history_ += 1.0;
    const double t = ctx.t_end.to_double();
    const double T1 = T_outer_1(t);
    for (std::size_t i = 0; i < n_; ++i) {
      q_[i] = k1_at(i, n_) * (T1 - u_in_[i]) / kL1;
    }
    time_ = t;
    return fuse::Result<void>::success();
  }

  fuse::Result<void> save_state() override {
    std::copy(q_.begin(), q_.end(), saved_q_.begin());  // element-wise
    saved_time_ = time_;
    saved_history_ = history_;
    ++save_count;
    return fuse::Result<void>::success();
  }
  fuse::Result<void> restore_state() override {
    // std::copy into the pre-sized registered array. `q_ = saved_q_` is
    // permitted to reallocate, which would move the address the coupler
    // resolved at setup.
    std::copy(saved_q_.begin(), saved_q_.end(), q_.begin());
    time_ = saved_time_;
    history_ = saved_history_;
    ++restore_count;
    return fuse::Result<void>::success();
  }
  /// Rollback: save_state()/restore_state() are real. Bitwise: a hand-written
  /// solver with no RNG and no unordered reduction (elementwise Kokkos
  /// kernels only), and example.*.omp_identical measures it bit for bit at
  /// one thread and at every core. Hidden state: YES -- advance() mutates
  /// state that is not bound through register_fields() (the time grid, the
  /// history, the solver's internals), which the runtime's byte-for-byte
  /// rollback verification cannot see; save_state() is what carries it.
  [[nodiscard]] std::uint64_t capabilities() const override {
    return fuse::make_capabilities(/*rollback=*/true, fuse::DeterminismLevel::Bitwise,
                                   /*hidden_state=*/true);
  }

  [[nodiscard]] const std::vector<double>& q() const { return q_; }
  [[nodiscard]] const std::vector<double>& u_in() const { return u_in_; }
  [[nodiscard]] const double* registered_provides() const { return q_.data(); }
  [[nodiscard]] double time() const { return time_; }
  /// Private state that advance() ACCUMULATES rather than overwrites, and the
  /// only thing in this component that a missing rollback can be seen in: q_
  /// and u_ are written from scratch every sweep, so a runner that never
  /// restored anything would still reach the same fixed point. The checkpoint
  /// is taken once per coupling cycle, so after P coupling cycles a correct run has
  /// history() == P exactly, however many thousand sweeps it took. It is not
  /// visible to the runtime's byte comparison of registered storage either --
  /// that checks q_/u_, this is private.
  [[nodiscard]] double history() const { return history_; }

  int advance_count = 0;
  int save_count = 0;
  int restore_count = 0;

 private:
  std::string         name_;
  Wiring              wiring_;
  std::size_t         n_;
  std::vector<double> q_;      // Provides: interface flux [W/m^2]
  std::vector<double> u_in_;   // Requires: interface temperature [K]
  std::vector<double> decoy_;  // where a severed input lands
  std::vector<double> saved_q_;
  double              time_ = 0.0;
  double              saved_time_ = 0.0;
  double              history_ = 0.0;
  double              saved_history_ = 0.0;
};

/// Slab 2: the Neumann solver. Requires the interface heat flux, provides the
/// interface temperature.
class NeumannSlab final : public fuse::Component {
 public:
  NeumannSlab(std::string name, std::size_t n, Wiring wiring)
      : name_(std::move(name)),
        wiring_(wiring),
        n_(n),
        u_(n, 0.0),
        q_in_(n, 0.0),
        decoy_(n, 0.0),
        saved_u_(n, 0.0) {
    // A physically sensible first guess, and the value a severed run is stuck
    // with: the interface at the outer temperature of this slab.
    std::fill(u_.begin(), u_.end(), T_outer_2(0.0));
  }

  [[nodiscard]] std::string name() const override { return name_; }

  fuse::Result<void> register_fields(fuse::FieldRegistry& reg,
                                     const fuse::CouplingGraph& g) override {
    const auto cid = g.find_component(name_);
    if (!cid.has_value()) {
      return fuse::Failure{"E-EX-COMP", "component '" + name_ + "' not in graph"};
    }
    for (const fuse::FieldId f : g.component(*cid).fields) {
      const bool provides = g.field(f).intent == fuse::Intent::Provides;
      double* base = provides
                         ? u_.data()
                         : (wiring_ == Wiring::Connected ? q_in_.data()
                                                         : decoy_.data());
      auto r = reg.bind(f, fuse::FieldRef{base,
                                          fuse::Layout{.stride = 1,
                                                       .count = n_},
                                          fuse::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }

  fuse::Result<void> advance(const fuse::StepContext& ctx) override {
    ++advance_count;
    history_ += 1.0;
    const double t = ctx.t_end.to_double();
    const double T2 = T_outer_2(t);
    for (std::size_t i = 0; i < n_; ++i) {
      u_[i] = T2 + q_in_[i] * kL2 / kK2;
    }
    time_ = t;
    return fuse::Result<void>::success();
  }

  fuse::Result<void> save_state() override {
    std::copy(u_.begin(), u_.end(), saved_u_.begin());
    saved_time_ = time_;
    saved_history_ = history_;
    ++save_count;
    return fuse::Result<void>::success();
  }
  fuse::Result<void> restore_state() override {
    std::copy(saved_u_.begin(), saved_u_.end(), u_.begin());
    time_ = saved_time_;
    history_ = saved_history_;
    ++restore_count;
    return fuse::Result<void>::success();
  }
  /// Rollback: save_state()/restore_state() are real. Bitwise: a hand-written
  /// solver with no RNG and no unordered reduction (elementwise Kokkos
  /// kernels only), and example.*.omp_identical measures it bit for bit at
  /// one thread and at every core. Hidden state: YES -- advance() mutates
  /// state that is not bound through register_fields() (the time grid, the
  /// history, the solver's internals), which the runtime's byte-for-byte
  /// rollback verification cannot see; save_state() is what carries it.
  [[nodiscard]] std::uint64_t capabilities() const override {
    return fuse::make_capabilities(/*rollback=*/true, fuse::DeterminismLevel::Bitwise,
                                   /*hidden_state=*/true);
  }

  [[nodiscard]] const std::vector<double>& u() const { return u_; }
  [[nodiscard]] const std::vector<double>& q_in() const { return q_in_; }
  [[nodiscard]] const double* registered_provides() const { return u_.data(); }
  [[nodiscard]] double time() const { return time_; }
  [[nodiscard]] double history() const { return history_; }

  int advance_count = 0;
  int save_count = 0;
  int restore_count = 0;

 private:
  std::string         name_;
  Wiring              wiring_;
  std::size_t         n_;
  std::vector<double> u_;      // Provides: interface temperature [K]
  std::vector<double> q_in_;   // Requires: interface flux [W/m^2]
  std::vector<double> decoy_;
  std::vector<double> saved_u_;
  double              time_ = 0.0;
  double              saved_time_ = 0.0;
  double              history_ = 0.0;
  double              saved_history_ = 0.0;
};

}  // namespace dn
