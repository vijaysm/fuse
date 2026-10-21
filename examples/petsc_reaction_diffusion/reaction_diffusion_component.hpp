#pragma once

/// S7.4 (owner decision 2026-10-21, Option C). Two `fuse::ImplicitComponent`
/// halves of the reaction-diffusion problem `reaction_diffusion_physics.hpp`
/// documents: `DirichletHalf` (always native) and `NeumannHalf<Substrate>`
/// (native OR petsc, selected at construction) -- the SAME physics, so the
/// native-substrate run of `NeumannHalf` is both "the existing native
/// ImplicitComponent partner" this track pairs its PETSc component against
/// AND the "native twin" the acceptance criteria compare the PETSc
/// substrate to (design_notes.md Section 2 explains why these are one
/// requirement, not two, for a region with exactly one PETSc-eligible
/// participant).
///
/// n = 1 STATION (one shared interface, no lateral parallelism): this
/// track's own region exchanges a single scalar Dirichlet/Neumann pair,
/// exactly dn_heat's own per-station convention collapsed to one station
/// -- the region-level `LinearizationToken` is therefore a REAL but SMALL
/// (1x1) PETSc `Mat`; the genuinely multi-point sparse `Mat`/`Vec`/`KSP`
/// PETSc machinery this track demonstrates is the INNER Newton solve
/// (reaction_diffusion_physics.hpp's own `right_residual_jacobian`, m
/// interior points, tridiagonal), not the outer token.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <fuse/runtime/component.hpp>

#include "reaction_diffusion_physics.hpp"

namespace petsc_reaction_diffusion {

/// Slab-1 analogue (examples/dn_heat/dn_component.hpp's own DirichletSlab):
/// receives the interface value, provides the flux. ALWAYS native (plain
/// `std::vector`, `thomas_solve`) -- this is the "existing native
/// ImplicitComponent" half of the pair; nothing in this track makes it
/// PETSc-selectable, because the region needs exactly one PETSc-eligible
/// participant to demonstrate the seam this track is about.
class DirichletHalf final : public fuse::Component, public fuse::ImplicitComponent {
 public:
  DirichletHalf(std::string name, int m, PhysicsParams params)
      : name_(std::move(name)),
        m_(m),
        params_(params),
        dx_(params.L_left / static_cast<double>(m + 1)),
        q_(1, 0.0),
        u_in_(1, 0.0),
        profile_(static_cast<std::size_t>(m), 0.0),
        saved_profile_(static_cast<std::size_t>(m), 0.0) {}

  [[nodiscard]] std::string name() const override { return name_; }

  fuse::Result<void> register_fields(fuse::FieldRegistry& reg,
                                     const fuse::CouplingGraph& g) override {
    const auto cid = g.find_component(name_);
    if (!cid.has_value()) {
      return fuse::Failure{"E-EX-COMP", "component '" + name_ + "' not in graph"};
    }
    for (const fuse::FieldId f : g.component(*cid).fields) {
      const bool provides = g.field(f).intent == fuse::Intent::Provides;
      double* base = provides ? q_.data() : u_in_.data();
      auto r = reg.bind(f, fuse::FieldRef{base, fuse::Layout{.stride = 1, .count = 1},
                                          fuse::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }

  /// Seeds `profile_` to the manufactured solution at t=0 -- this
  /// component's own initial condition, called once by main.cpp before
  /// the first period (there is no graph-level "initial condition" hook
  /// for a private, unregistered array; dn_heat has no analogue because
  /// its own profile is closed-form and needs no seed).
  void seed_initial_condition() {
    for (int i = 0; i < m_; ++i) {
      profile_[static_cast<std::size_t>(i)] =
          manufactured_u(static_cast<double>(i + 1) * dx_, 0.0, params_);
    }
  }

  fuse::Result<void> advance(const fuse::StepContext& ctx) override {
    const double dt = ctx.dt().to_double();
    const double t_new = ctx.t_end.to_double();
    const LeftSolveResult r =
        left_newton_solve(profile_, u_in_[0], dt, dx_, t_new, params_);
    profile_ = r.u;
    q_[0] = r.q_I;
    last_sensitivity_ = r.dG_du_I;
    ++advance_count;
    ++signature_;
    return fuse::Result<void>::success();
  }

  fuse::Result<void> save_state() override {
    saved_profile_ = profile_;
    ++save_count;
    return fuse::Result<void>::success();
  }
  fuse::Result<void> restore_state() override {
    profile_ = saved_profile_;
    ++restore_count;
    return fuse::Result<void>::success();
  }

  [[nodiscard]] std::uint64_t capabilities() const override {
    return fuse::make_capabilities(/*rollback=*/true, fuse::DeterminismLevel::Bitwise,
                                   /*hidden_state=*/true) |
           fuse::kCapabilityResidual | fuse::kCapabilityLinearization;
  }

  /// Pure: uses the CURRENT (checkpointed) `profile_` as the backward-Euler
  /// "old" level, same convention as DirichletSlab's own
  /// `evaluate_residual` (component.hpp's own contract: this must compute
  /// the SAME VALUE `advance()` would at this `ctx`/`interface_in`, with
  /// no mutation of anything `save_state()` would need).
  fuse::Result<void> evaluate_residual(const fuse::StepContext& ctx,
                                       std::span<const double> interface_in,
                                       std::span<double> residual_out) override {
    if (interface_in.size() != 1 || residual_out.size() != 1) {
      return fuse::Failure{"E-EX-COMP", "DirichletHalf::evaluate_residual: expected 1 in/out"};
    }
    const double dt = ctx.dt().to_double();
    const double t_new = ctx.t_end.to_double();
    const LeftSolveResult r =
        left_newton_solve(profile_, interface_in[0], dt, dx_, t_new, params_);
    residual_out[0] = r.q_I;
    last_sensitivity_ = r.dG_du_I;
    ++signature_;
    return fuse::Result<void>::success();
  }

  class Token final : public fuse::LinearizationToken {
   public:
    explicit Token(double slope, std::uint64_t sig) : slope_(slope), sig_(sig) {}
    fuse::Result<void> apply_jacobian(std::span<const double> x,
                                      std::span<double> y) override {
      y[0] = slope_ * x[0];
      return fuse::Result<void>::success();
    }
    fuse::Result<void> apply_jacobian_transpose(std::span<const double> x,
                                                std::span<double> y) override {
      y[0] = slope_ * x[0];  // 1x1: transpose is itself
      return fuse::Result<void>::success();
    }
    fuse::Result<void> apply_preconditioner(std::span<const double> r,
                                            std::span<double> z) override {
      z[0] = r[0];  // identity: the always-legal fallback (component.hpp)
      return fuse::Result<void>::success();
    }
    [[nodiscard]] std::size_t rows() const override { return 1; }
    [[nodiscard]] std::size_t cols() const override { return 1; }
    [[nodiscard]] std::uint64_t signature() const override { return sig_; }

   private:
    double slope_;
    std::uint64_t sig_;
  };

  [[nodiscard]] fuse::LinearizationToken* linearization() override {
    token_ = std::make_unique<Token>(last_sensitivity_, signature_);
    return token_.get();
  }
  [[nodiscard]] fuse::ImplicitComponent* implicit() override { return this; }

  [[nodiscard]] const std::vector<double>& profile() const { return profile_; }
  [[nodiscard]] double q() const { return q_[0]; }

  int advance_count = 0;
  int save_count = 0;
  int restore_count = 0;

 private:
  std::string name_;
  int m_;
  PhysicsParams params_;
  double dx_;
  std::vector<double> q_;       // Provides: interface flux (size 1)
  std::vector<double> u_in_;    // Requires: interface value (size 1)
  std::vector<double> profile_;
  std::vector<double> saved_profile_;
  double last_sensitivity_ = 0.0;
  std::uint64_t signature_ = 0;
  std::unique_ptr<Token> token_;
};

/// Slab-2 analogue (NeumannSlab): receives the interface flux, provides the
/// value its own solution implies. NATIVE substrate: `thomas_solve` over
/// `reaction_diffusion_physics.hpp`'s own plain-array Jacobian -- the
/// "native twin" of `NeumannHalfPetsc` below (IDENTICAL discrete equations,
/// different linear-algebra substrate for the inner solve).
class NeumannHalfNative final : public fuse::Component, public fuse::ImplicitComponent {
 public:
  NeumannHalfNative(std::string name, int m, PhysicsParams params)
      : name_(std::move(name)),
        m_(m),
        params_(params),
        dx_(params.L_right / static_cast<double>(m + 1)),
        q_in_(1, 0.0),
        u_out_(1, 0.0),
        profile_(static_cast<std::size_t>(m), 0.0),
        saved_profile_(static_cast<std::size_t>(m), 0.0) {}

  [[nodiscard]] std::string name() const override { return name_; }

  fuse::Result<void> register_fields(fuse::FieldRegistry& reg,
                                     const fuse::CouplingGraph& g) override {
    const auto cid = g.find_component(name_);
    if (!cid.has_value()) {
      return fuse::Failure{"E-EX-COMP", "component '" + name_ + "' not in graph"};
    }
    for (const fuse::FieldId f : g.component(*cid).fields) {
      const bool provides = g.field(f).intent == fuse::Intent::Provides;
      double* base = provides ? u_out_.data() : q_in_.data();
      auto r = reg.bind(f, fuse::FieldRef{base, fuse::Layout{.stride = 1, .count = 1},
                                          fuse::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }

  void seed_initial_condition() {
    for (int i = 0; i < m_; ++i) {
      profile_[static_cast<std::size_t>(i)] = manufactured_u(
          params_.L_left + static_cast<double>(i + 1) * dx_, 0.0, params_);
    }
  }

  fuse::Result<void> advance(const fuse::StepContext& ctx) override {
    const double dt = ctx.dt().to_double();
    const double t_new = ctx.t_end.to_double();
    const RightSolveResult r =
        right_newton_solve(profile_, q_in_[0], dt, dx_, t_new, params_);
    profile_ = r.u;
    u_out_[0] = r.u_interface;
    last_sensitivity_ = r.dG_dq_I;
    ++advance_count;
    ++signature_;
    return fuse::Result<void>::success();
  }

  fuse::Result<void> save_state() override {
    saved_profile_ = profile_;
    ++save_count;
    return fuse::Result<void>::success();
  }
  fuse::Result<void> restore_state() override {
    profile_ = saved_profile_;
    ++restore_count;
    return fuse::Result<void>::success();
  }

  [[nodiscard]] std::uint64_t capabilities() const override {
    return fuse::make_capabilities(/*rollback=*/true, fuse::DeterminismLevel::Bitwise,
                                   /*hidden_state=*/true) |
           fuse::kCapabilityResidual | fuse::kCapabilityLinearization;
  }

  fuse::Result<void> evaluate_residual(const fuse::StepContext& ctx,
                                       std::span<const double> interface_in,
                                       std::span<double> residual_out) override {
    if (interface_in.size() != 1 || residual_out.size() != 1) {
      return fuse::Failure{"E-EX-COMP", "NeumannHalfNative::evaluate_residual: expected 1 in/out"};
    }
    const double dt = ctx.dt().to_double();
    const double t_new = ctx.t_end.to_double();
    const RightSolveResult r =
        right_newton_solve(profile_, interface_in[0], dt, dx_, t_new, params_);
    residual_out[0] = r.u_interface;
    last_sensitivity_ = r.dG_dq_I;
    ++signature_;
    return fuse::Result<void>::success();
  }

  class Token final : public fuse::LinearizationToken {
   public:
    explicit Token(double slope, std::uint64_t sig) : slope_(slope), sig_(sig) {}
    fuse::Result<void> apply_jacobian(std::span<const double> x,
                                      std::span<double> y) override {
      y[0] = slope_ * x[0];
      return fuse::Result<void>::success();
    }
    fuse::Result<void> apply_jacobian_transpose(std::span<const double> x,
                                                std::span<double> y) override {
      y[0] = slope_ * x[0];
      return fuse::Result<void>::success();
    }
    fuse::Result<void> apply_preconditioner(std::span<const double> r,
                                            std::span<double> z) override {
      z[0] = r[0];
      return fuse::Result<void>::success();
    }
    [[nodiscard]] std::size_t rows() const override { return 1; }
    [[nodiscard]] std::size_t cols() const override { return 1; }
    [[nodiscard]] std::uint64_t signature() const override { return sig_; }

   private:
    double slope_;
    std::uint64_t sig_;
  };

  [[nodiscard]] fuse::LinearizationToken* linearization() override {
    token_ = std::make_unique<Token>(last_sensitivity_, signature_);
    return token_.get();
  }
  [[nodiscard]] fuse::ImplicitComponent* implicit() override { return this; }

  [[nodiscard]] const std::vector<double>& profile() const { return profile_; }
  [[nodiscard]] double u_out() const { return u_out_[0]; }

  int advance_count = 0;
  int save_count = 0;
  int restore_count = 0;

 private:
  std::string name_;
  int m_;
  PhysicsParams params_;
  double dx_;
  std::vector<double> q_in_;   // Requires: interface flux (size 1)
  std::vector<double> u_out_;  // Provides: interface value (size 1)
  std::vector<double> profile_;
  std::vector<double> saved_profile_;
  double last_sensitivity_ = 0.0;
  std::uint64_t signature_ = 0;
  std::unique_ptr<Token> token_;
};

}  // namespace petsc_reaction_diffusion
