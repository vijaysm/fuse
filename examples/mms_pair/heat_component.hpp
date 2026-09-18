#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include <fuse/runtime/component.hpp>

namespace mms {

inline constexpr double kPi = 3.14159265358979323846;

/// Cell centre of cell `i` of an `n`-cell uniform mesh on [0, 1] -- the layout
/// SerialMeshBackend's "generated:line:n" produces.
[[nodiscard]] inline double cell_x(std::size_t i, std::size_t n) {
  return (static_cast<double>(i) + 0.5) / static_cast<double>(n);
}

/// Discrete L2 norm of `a - b` on that mesh.
[[nodiscard]] inline double l2_diff(const std::vector<double>& a,
                                    const std::vector<double>& b) {
  const double dx = 1.0 / static_cast<double>(a.size());
  double       sum = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double e = a[i] - b[i];
    sum += e * e * dx;
  }
  return std::sqrt(sum);
}

/// 1D heat equation, explicit Euler, cell-centred finite volume on [0,1].
///   T_t = alpha * T_xx,  T(0,t) = T(1,t) = 0
/// Exact solution for T(x,0) = sin(pi x):
///   T(x,t) = exp(-pi^2 alpha t) sin(pi x)
///
/// STORAGE. `t_` is registered with the coupler and its ADDRESS MUST NOT MOVE
/// (soma/runtime/component.hpp, register_fields). The update below therefore
/// writes the new level into `scratch_` and copies it back element-wise.
/// Swapping the two vectors -- the obvious idiom -- swaps their heap buffers,
/// so the coupler would keep reading the address it resolved at setup, which
/// after the swap is `scratch_`: the exchange silently alternates between the
/// current and the previous time level with no diagnostic.
///
/// Only `Intent::Provides` and `Intent::Bidirectional` fields are bound to
/// `t_`. A `Requires` field bound to the state array would have the coupler
/// overwrite the solution with the incoming boundary data every period.
class HeatComponent final : public fuse::Component {
 public:
  HeatComponent(std::string name, std::size_t cells, double alpha)
      : name_(std::move(name)),
        alpha_(alpha),
        dx_(1.0 / static_cast<double>(cells)),
        t_(cells, 0.0),
        scratch_(cells, 0.0),
        saved_(cells, 0.0),
        incoming_(cells, 0.0) {
    for (std::size_t i = 0; i < cells; ++i) {
      t_[i] = std::sin(kPi * x(i));
    }
  }

  [[nodiscard]] std::string name() const override { return name_; }

  fuse::Result<void> register_fields(fuse::FieldRegistry& reg,
                                     const fuse::CouplingGraph& g) override {
    const auto cid = g.find_component(name_);
    if (!cid.has_value()) {
      return fuse::Failure{"E-EX-COMP", "component '" + name_ + "' not in graph"};
    }
    for (const fuse::FieldId f : g.component(*cid).fields) {
      double* base = g.field(f).intent == fuse::Intent::Requires
                         ? incoming_.data()
                         : t_.data();
      const std::size_t count = g.field(f).intent == fuse::Intent::Requires
                                    ? incoming_.size()
                                    : t_.size();
      auto r = reg.bind(f, fuse::FieldRef{base,
                                          fuse::Layout{.stride = 1,
                                                       .count = count},
                                          fuse::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }

  fuse::Result<void> advance(const fuse::StepContext& ctx) override {
    const double dt = ctx.dt().to_double();
    const double r = alpha_ * dt / (dx_ * dx_);
    // Checked with a Result, not an assert: -DNDEBUG deletes an assert and an
    // unstable explicit Euler does not crash, it produces growing garbage.
    if (r > 0.5) {
      return fuse::Failure{"E-EX-CFL",
                         "explicit Euler needs alpha*dt/dx^2 <= 1/2, got " +
                             std::to_string(r)};
    }
    const std::size_t n = t_.size();
    for (std::size_t i = 0; i < n; ++i) {
      const double left  = (i == 0) ? -t_[0] : t_[i - 1];          // T(0) = 0
      const double right = (i + 1 == n) ? -t_[n - 1] : t_[i + 1];  // T(1) = 0
      scratch_[i] = t_[i] + r * (left - 2.0 * t_[i] + right);
    }
    std::copy(scratch_.begin(), scratch_.end(), t_.begin());  // never swap()
    time_ = ctx.t_end.to_double();
    return fuse::Result<void>::success();
  }

  fuse::Result<void> save_state() override {
    std::copy(t_.begin(), t_.end(), saved_.begin());
    saved_time_ = time_;
    return fuse::Result<void>::success();
  }
  fuse::Result<void> restore_state() override {
    // Element-wise: `t_ = saved_` is permitted to reallocate, which would move
    // the registered address.
    std::copy(saved_.begin(), saved_.end(), t_.begin());
    time_ = saved_time_;
    return fuse::Result<void>::success();
  }
  [[nodiscard]] bool supports_rollback() const override { return true; }

  [[nodiscard]] double x(std::size_t i) const {
    return (static_cast<double>(i) + 0.5) * dx_;
  }

  [[nodiscard]] double exact(std::size_t i, double t) const {
    return std::exp(-kPi * kPi * alpha_ * t) * std::sin(kPi * x(i));
  }

  [[nodiscard]] double l2_error() const {
    double sum = 0.0;
    for (std::size_t i = 0; i < t_.size(); ++i) {
      const double e = t_[i] - exact(i, time_);
      sum += e * e * dx_;
    }
    return std::sqrt(sum);
  }

  [[nodiscard]] const std::vector<double>& values() const { return t_; }
  [[nodiscard]] double time() const { return time_; }
  [[nodiscard]] double alpha() const { return alpha_; }

 private:
  std::string         name_;
  double              alpha_;
  double              dx_;
  std::vector<double> t_;
  std::vector<double> scratch_;
  std::vector<double> saved_;
  /// Storage for any Requires field. Never aliased onto `t_`.
  std::vector<double> incoming_;
  double              time_ = 0.0;
  double              saved_time_ = 0.0;
};

/// The coupled partner, and the reason this example measures the COUPLER.
///
///   dR/dt = kappa (T_in(x, t) - R(x, t)),   R(x, 0) = 0
///
/// It is integrated in closed form over each step -- for an input the coupler
/// holds fixed across the step, R(t0 + dt) = c + (R(t0) - c) exp(-kappa dt) is
/// exact -- so this component contributes NO discretisation error. Driven by
/// the heat equation's analytic solution T = exp(-mu t) sin(pi x), with
/// mu = pi^2 alpha, the exact response is
///
///   R(x, t) = sin(pi x) * kappa/(kappa - mu) * (exp(-mu t) - exp(-kappa t))
///
/// so the L2 error this component reports is the coupler's: the window mean
/// the plan forms of T, plus the one coupling period by which it is late.
class RelaxComponent final : public fuse::Component {
 public:
  RelaxComponent(std::string name, std::size_t cells, double kappa, double mu)
      : name_(std::move(name)),
        kappa_(kappa),
        mu_(mu),
        dx_(1.0 / static_cast<double>(cells)),
        r_(cells, 0.0),
        in_(cells, 0.0),
        saved_(cells, 0.0) {}

  [[nodiscard]] std::string name() const override { return name_; }

  fuse::Result<void> register_fields(fuse::FieldRegistry& reg,
                                     const fuse::CouplingGraph& g) override {
    const auto cid = g.find_component(name_);
    if (!cid.has_value()) {
      return fuse::Failure{"E-EX-COMP", "component '" + name_ + "' not in graph"};
    }
    for (const fuse::FieldId f : g.component(*cid).fields) {
      // in_, not r_: r_ is the solution and the coupler must never write it.
      auto r = reg.bind(f, fuse::FieldRef{in_.data(),
                                          fuse::Layout{.stride = 1,
                                                       .count = in_.size()},
                                          fuse::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }

  fuse::Result<void> advance(const fuse::StepContext& ctx) override {
    const double decay = std::exp(-kappa_ * ctx.dt().to_double());
    for (std::size_t i = 0; i < r_.size(); ++i) {
      const double c = in_[i];
      r_[i] = c + (r_[i] - c) * decay;
    }
    time_ = ctx.t_end.to_double();
    return fuse::Result<void>::success();
  }

  fuse::Result<void> save_state() override {
    std::copy(r_.begin(), r_.end(), saved_.begin());
    saved_time_ = time_;
    return fuse::Result<void>::success();
  }
  fuse::Result<void> restore_state() override {
    std::copy(saved_.begin(), saved_.end(), r_.begin());
    time_ = saved_time_;
    return fuse::Result<void>::success();
  }
  [[nodiscard]] bool supports_rollback() const override { return true; }

  [[nodiscard]] double exact(std::size_t i, double t) const {
    return std::sin(kPi * cell_x(i, r_.size())) * (kappa_ / (kappa_ - mu_)) *
           (std::exp(-mu_ * t) - std::exp(-kappa_ * t));
  }

  [[nodiscard]] double l2_error() const {
    double sum = 0.0;
    for (std::size_t i = 0; i < r_.size(); ++i) {
      const double e = r_[i] - exact(i, time_);
      sum += e * e * dx_;
    }
    return std::sqrt(sum);
  }

  /// ||R_exact(T)||: the error this component would report if the exchange
  /// delivered nothing at all, since R then stays identically zero. The
  /// coupled error has to be a small fraction of this or the run proves
  /// nothing about the coupling.
  [[nodiscard]] double uncoupled_error() const {
    double sum = 0.0;
    for (std::size_t i = 0; i < r_.size(); ++i) {
      const double e = exact(i, time_);
      sum += e * e * dx_;
    }
    return std::sqrt(sum);
  }

  [[nodiscard]] const std::vector<double>& values() const { return r_; }
  [[nodiscard]] const std::vector<double>& received() const { return in_; }
  [[nodiscard]] double time() const { return time_; }

 private:
  std::string         name_;
  double              kappa_;
  double              mu_;
  double              dx_;
  std::vector<double> r_;
  std::vector<double> in_;
  std::vector<double> saved_;
  double              time_ = 0.0;
  double              saved_time_ = 0.0;
};

}  // namespace mms
