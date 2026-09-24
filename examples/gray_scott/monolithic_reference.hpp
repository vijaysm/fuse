#pragma once

/// The single-rate reference against which the multirate windowed coupling
/// (species_component.hpp driven through fuse::WindowCoupler) is checked in
/// test/verification/gray_scott_reference_test.cpp: one Crank-Nicolson step
/// advances BOTH species together, Newton-solved as one coupled 2*N^2 system
/// (unlike GrayScottSpeciesComponent, which treats the partner as frozen
/// input and therefore never needs the cross derivative
/// d(React_u)/d(v) or d(React_v)/d(u) -- this solver needs both, because
/// here neither species is frozen).
///
/// Unknowns are laid out as [u_0..u_{n2-1}, v_0..v_{n2-1}]. The Jacobian's
/// diffusion blocks are diagonal-in-species (u's Laplacian never touches v's
/// unknowns and vice versa); the reaction term is what couples the two
/// diagonal blocks through the OFF-diagonal n2 x n2 blocks
/// d(React_u)/dv and d(React_v)/du, each of which is itself diagonal in
/// space (the reaction is local to a cell, not a stencil).

#include <Eigen/SparseLU>
#include <cmath>
#include <functional>
#include <vector>

#include "gray_scott_physics.hpp"

namespace gray_scott {

/// Manufactured-solution forcing for each species, S_u(x,y,t)/S_v(x,y,t).
using MonolithicSourceFn = std::function<double(double, double, double)>;

class MonolithicGrayScott {
 public:
  MonolithicGrayScott(Grid grid, double Du, double Dv, Kinetics kinetics,
                      std::vector<double> u0, std::vector<double> v0,
                      double newton_tol, int newton_max_iter = 50)
      : grid_(grid),
        Du_(Du),
        Dv_(Dv),
        kin_(kinetics),
        lap_(periodic_laplacian(grid)),
        u_(std::move(u0)),
        v_(std::move(v0)),
        newton_tol_(newton_tol),
        newton_max_iter_(newton_max_iter) {}

  void set_sources(MonolithicSourceFn su, MonolithicSourceFn sv) {
    su_ = std::move(su);
    sv_ = std::move(sv);
  }

  [[nodiscard]] const std::vector<double>& u() const { return u_; }
  [[nodiscard]] const std::vector<double>& v() const { return v_; }
  void set_state(const std::vector<double>& u, const std::vector<double>& v) {
    u_ = u;
    v_ = v;
  }

  /// Advances one Crank-Nicolson step over [t_begin, t_begin + dt). Returns
  /// false (and leaves state unchanged) if Newton fails to converge.
  [[nodiscard]] bool step(double dt, double t_begin) {
    const std::size_t n2 = grid_.size();
    const double t_end = t_begin + dt;

    std::vector<double> su_sum(n2, 0.0), sv_sum(n2, 0.0);
    if (su_ || sv_) {
      for (int i = 0; i < grid_.n; ++i) {
        const double x = grid_.x_center(i);
        for (int j = 0; j < grid_.n; ++j) {
          const double y = grid_.y_center(j);
          const std::size_t m = grid_.index(i, j);
          if (su_) su_sum[m] = su_(x, y, t_begin) + su_(x, y, t_end);
          if (sv_) sv_sum[m] = sv_(x, y, t_begin) + sv_(x, y, t_end);
        }
      }
    }

    const Eigen::VectorXd u_old = Eigen::Map<const Eigen::VectorXd>(u_.data(), static_cast<int>(n2));
    const Eigen::VectorXd v_old = Eigen::Map<const Eigen::VectorXd>(v_.data(), static_cast<int>(n2));
    const Eigen::VectorXd lap_u_old = lap_ * u_old;
    const Eigen::VectorXd lap_v_old = lap_ * v_old;

    Eigen::VectorXd u_new = u_old, v_new = v_old;

    // Diffusion-only blocks (constant across Newton iterations).
    Eigen::SparseMatrix<double> base_u = (-0.5 * dt * Du_) * lap_;
    Eigen::SparseMatrix<double> base_v = (-0.5 * dt * Dv_) * lap_;
    for (int m = 0; m < static_cast<int>(n2); ++m) {
      base_u.coeffRef(m, m) += 1.0;
      base_v.coeffRef(m, m) += 1.0;
    }
    base_u.makeCompressed();
    base_v.makeCompressed();

    bool converged = false;
    for (int iter = 0; iter < newton_max_iter_; ++iter) {
      const Eigen::VectorXd lap_u_new = lap_ * u_new;
      const Eigen::VectorXd lap_v_new = lap_ * v_new;
      Eigen::VectorXd Ru(static_cast<int>(n2)), Rv(static_cast<int>(n2));
      for (std::size_t m = 0; m < n2; ++m) {
        const int mi = static_cast<int>(m);
        const double ru_new = kin_.reaction_u(u_new[mi], v_new[mi]);
        const double ru_old = kin_.reaction_u(u_old[mi], v_old[mi]);
        const double rv_new = kin_.reaction_v(u_new[mi], v_new[mi]);
        const double rv_old = kin_.reaction_v(u_old[mi], v_old[mi]);
        Ru[mi] = u_new[mi] - u_old[mi] -
                 0.5 * dt * (Du_ * (lap_u_new[mi] + lap_u_old[mi]) + ru_new + ru_old + su_sum[m]);
        Rv[mi] = v_new[mi] - v_old[mi] -
                 0.5 * dt * (Dv_ * (lap_v_new[mi] + lap_v_old[mi]) + rv_new + rv_old + sv_sum[m]);
      }
      const double resid_norm = std::sqrt(Ru.squaredNorm() + Rv.squaredNorm());
      if (resid_norm <= newton_tol_) {
        converged = true;
        break;
      }
      if (iter + 1 == newton_max_iter_) break;

      // Assemble the 2n2 x 2n2 block Jacobian:
      //   [ base_u - dt/2*diag(dRu/du)     -dt/2*diag(dRu/dv)        ]
      //   [ -dt/2*diag(dRv/du)             base_v - dt/2*diag(dRv/dv)]
      const int N = static_cast<int>(n2);
      std::vector<Eigen::Triplet<double>> trips;
      trips.reserve(static_cast<std::size_t>(base_u.nonZeros() + base_v.nonZeros()) + 4 * n2);
      for (int col = 0; col < N; ++col) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(base_u, col); it; ++it) {
          double val = it.value();
          if (it.row() == col) {
            val -= 0.5 * dt * kin_.d_reaction_u_du(u_new[col], v_new[col]);
          }
          trips.emplace_back(it.row(), col, val);
        }
      }
      for (int col = 0; col < N; ++col) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(base_v, col); it; ++it) {
          double val = it.value();
          if (it.row() == col) {
            val -= 0.5 * dt * kin_.d_reaction_v_dv(u_new[col], v_new[col]);
          }
          trips.emplace_back(N + it.row(), N + col, val);
        }
      }
      for (int m = 0; m < N; ++m) {
        trips.emplace_back(m, N + m, -0.5 * dt * kin_.d_reaction_u_dv(u_new[m], v_new[m]));
        trips.emplace_back(N + m, m, -0.5 * dt * kin_.d_reaction_v_du(u_new[m], v_new[m]));
      }
      Eigen::SparseMatrix<double> jac(2 * N, 2 * N);
      jac.setFromTriplets(trips.begin(), trips.end());
      jac.makeCompressed();

      Eigen::VectorXd rhs(2 * N);
      rhs.head(N) = -Ru;
      rhs.tail(N) = -Rv;

      Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
      solver.compute(jac);
      if (solver.info() != Eigen::Success) return false;
      const Eigen::VectorXd delta = solver.solve(rhs);
      if (solver.info() != Eigen::Success) return false;

      u_new += delta.head(N);
      v_new += delta.tail(N);
    }
    if (!converged) return false;

    for (std::size_t m = 0; m < n2; ++m) {
      u_[m] = u_new[static_cast<int>(m)];
      v_[m] = v_new[static_cast<int>(m)];
    }
    return true;
  }

 private:
  Grid     grid_;
  double   Du_, Dv_;
  Kinetics kin_;
  Eigen::SparseMatrix<double> lap_;
  std::vector<double> u_, v_;
  double newton_tol_;
  int    newton_max_iter_;
  MonolithicSourceFn su_, sv_;
};

}  // namespace gray_scott
