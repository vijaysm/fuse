#pragma once

/// Euler-Bernoulli beam, clamped at both ends, on its own structured FD grid:
///
///   rho_s h_s w_tt + EI w_xxxx = q(x, t),   x in [0, L],   w(0)=w_x(0)=0,
///                                                          w(L)=w_x(L)=0.
///
/// SPACE: 2nd-order central finite differences on n+1 nodes (n elements),
/// x_i = i * dx, i = 0..n. The two boundary nodes are fixed at w=0 and are
/// eliminated from the unknown vector; the slope conditions w_x(0)=w_x(L)=0
/// are imposed through the classical MIRROR ghost nodes
///
///   w_{-1}  = w_{1},      w_{n+1} = w_{n-1}
///
/// (the ghost value that makes a centred first difference across the wall
/// vanish). Substituting the mirror into the standard 5-point stencil for
/// w_xxxx,
///
///   w_xxxx_i ~= (w_{i-2} - 4 w_{i-1} + 6 w_i - 4 w_{i+1} + w_{i+2}) / dx^4,
///
/// folds the ghost into the diagonal at the two nodes adjacent to a clamped
/// end (coefficient 7, not 6) and drops one off-diagonal entry there; at
/// every other node the real boundary value w_0 = w_n = 0 simply zeroes the
/// term that would reference it, with no special case needed. `assemble_
/// stiffness` below applies exactly this rule via index reflection, so it
/// needs no separate boundary branch.
///
/// TIME: Newmark-beta, beta=1/4, gamma=1/2 ("average acceleration" /
/// trapezoidal rule) -- unconditionally stable, second order in time.
///
/// MASS: direct nodal collocation, not a lumped FE mass -- the PDE is
/// evaluated pointwise at each node, so M is the constant diagonal
/// rho_s * h_s with no dx factor, and the forcing on the right-hand side of
/// the ODE system is q(x_i, t), the DISTRIBUTED line load itself, not an
/// already-integrated nodal amount. `docs/user/fsi-channel.md` derives why a
/// FUSE-delivered NODAL LOAD (an already-integrated per-node force, in
/// newtons -- what `consistent-transpose`'s H^T D delivers, see
/// `fuse/backend/consistent_interp.hpp`) must be divided by this node's own
/// TRIBUTARY LENGTH before it can be used as this q(x_i, t): `tributary_
/// length()` below is exactly that length, so the FSI coupling code (`fsi_
/// components.hpp`) never has to re-derive it.

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace fsi {

struct BeamParams {
  std::size_t n = 16;      // elements; n+1 nodes, nodes 0 and n are clamped
  double      L = 1.0;     // length [m]
  double      EI = 1.0;    // flexural rigidity [N m^2]
  double      rho_s = 1.0; // material density [kg/m^3]
  double      h_s = 1.0;   // cross-section "thickness" so that rho_s*h_s is
                           // the beam's mass per unit length [kg/m]
};

/// Clamped-clamped Euler-Bernoulli beam, 2nd-order FD in space, Newmark-beta
/// (average acceleration) in time. All state (displacement, velocity,
/// acceleration) is nodal, size n+1, with entries 0 and n identically zero
/// (the clamped ends) at all times.
class BeamSolver {
 public:
  explicit BeamSolver(BeamParams p) : p_(p), dx_(p.L / static_cast<double>(p.n)) {
    if (p_.n < 4) {
      throw std::invalid_argument(
          "BeamSolver: need at least 4 elements for the clamped-clamped "
          "stencil to have a genuine interior node");
    }
    assemble_stiffness();
    w_.assign(p_.n + 1, 0.0);
    wdot_.assign(p_.n + 1, 0.0);
    wddot_.assign(p_.n + 1, 0.0);
  }

  [[nodiscard]] std::size_t n_nodes() const { return p_.n + 1; }
  [[nodiscard]] double dx() const { return dx_; }
  [[nodiscard]] double node_x(std::size_t i) const {
    return static_cast<double>(i) * dx_;
  }
  [[nodiscard]] const BeamParams& params() const { return p_; }

  /// The length of interface this node "owns" for the purpose of converting
  /// an already-integrated point load [N] into this solver's distributed
  /// line-load convention q(x_i) [N/m]: dx for every interior node, dx/2 at
  /// the clamped ends (unused physically -- the FSI coupling never applies a
  /// load there because the beam's response is Dirichlet-fixed regardless --
  /// but defined for a complete, honest API rather than left to divide by
  /// zero).
  [[nodiscard]] double tributary_length(std::size_t i) const {
    return (i == 0 || i == p_.n) ? 0.5 * dx_ : dx_;
  }

  /// Static solve K w = q, q given pointwise (distributed line load, N/m) at
  /// every node; entries 0 and n are ignored (the clamped ends contribute no
  /// equation). Returns the full n+1 nodal displacement, with w[0]=w[n]=0.
  [[nodiscard]] std::vector<double> solve_static(
      const std::vector<double>& q_at_nodes) const {
    if (q_at_nodes.size() != p_.n + 1) {
      throw std::invalid_argument("BeamSolver::solve_static: expected n+1 "
                                   "pointwise load values");
    }
    const std::size_t m = p_.n - 1;  // interior unknowns
    Eigen::VectorXd rhs(m);
    for (std::size_t idx = 0; idx < m; ++idx) rhs[idx] = q_at_nodes[idx + 1];

    Eigen::SparseLU<Eigen::SparseMatrix<double>> lu;
    lu.compute(K_);
    if (lu.info() != Eigen::Success) {
      throw std::runtime_error(
          "BeamSolver::solve_static: SparseLU factorization of K failed");
    }
    const Eigen::VectorXd sol = lu.solve(rhs);
    std::vector<double> w(p_.n + 1, 0.0);
    for (std::size_t idx = 0; idx < m; ++idx) w[idx + 1] = sol[idx];
    return w;
  }

  /// One Newmark-beta step of size dt, driven by the DISTRIBUTED line load
  /// q(x_i, t_new) [N/m] at the new time level (implicit: Newmark's average-
  /// acceleration form is unconditionally stable exactly because the load
  /// and stiffness are evaluated at t_new). Advances (w_, wdot_, wddot_) in
  /// place.
  void step(double dt, const std::vector<double>& q_at_nodes_new) {
    if (q_at_nodes_new.size() != p_.n + 1) {
      throw std::invalid_argument("BeamSolver::step: expected n+1 pointwise "
                                   "load values");
    }
    if (!(dt > 0.0)) {
      throw std::invalid_argument("BeamSolver::step: dt must be positive");
    }
    const std::size_t m = p_.n - 1;
    const double beta = 0.25, gamma = 0.5;
    const double a0 = 1.0 / (beta * dt * dt);
    const double a1 = 1.0 / (beta * dt);
    const double a2 = 1.0 / (2.0 * beta) - 1.0;
    const double mconst = p_.rho_s * p_.h_s;

    rebuild_effective(dt, beta);

    Eigen::VectorXd rhs(m);
    for (std::size_t idx = 0; idx < m; ++idx) {
      const std::size_t node = idx + 1;
      const double wn = w_[node], wdn = wdot_[node], wddn = wddot_[node];
      rhs[idx] = q_at_nodes_new[node] +
                 mconst * (a0 * wn + a1 * wdn + a2 * wddn);
    }
    const Eigen::VectorXd sol = keff_lu_.solve(rhs);
    if (keff_lu_.info() != Eigen::Success) {
      throw std::runtime_error("BeamSolver::step: SparseLU solve failed");
    }

    std::vector<double> w_new(p_.n + 1, 0.0), wddot_new(p_.n + 1, 0.0),
        wdot_new(p_.n + 1, 0.0);
    for (std::size_t idx = 0; idx < m; ++idx) {
      const std::size_t node = idx + 1;
      const double wn = w_[node], wdn = wdot_[node], wddn = wddot_[node];
      const double wnew = sol[idx];
      const double wddnew = a0 * (wnew - wn) - a1 * wdn - a2 * wddn;
      const double wdnew = wdn + dt * ((1.0 - gamma) * wddn + gamma * wddnew);
      w_new[node] = wnew;
      wddot_new[node] = wddnew;
      wdot_new[node] = wdnew;
    }
    w_ = std::move(w_new);
    wdot_ = std::move(wdot_new);
    wddot_ = std::move(wddot_new);
  }

  struct State {
    std::vector<double> w, wdot, wddot;
  };
  [[nodiscard]] State save() const { return State{w_, wdot_, wddot_}; }
  void restore(const State& s) {
    w_ = s.w;
    wdot_ = s.wdot;
    wddot_ = s.wddot;
  }
  void set_state(std::vector<double> w, std::vector<double> wdot,
                 std::vector<double> wddot) {
    w_ = std::move(w);
    wdot_ = std::move(wdot);
    wddot_ = std::move(wddot);
  }

  [[nodiscard]] const std::vector<double>& displacement() const { return w_; }
  [[nodiscard]] const std::vector<double>& velocity() const { return wdot_; }
  [[nodiscard]] const std::vector<double>& acceleration() const {
    return wddot_;
  }

  /// Discrete kinetic energy [J, per unit out-of-plane depth], consistent
  /// with this scheme's own mass convention (a CONSTANT rho_s*h_s at every
  /// interior node, no dx weight in the ODE's own mass term -- see the
  /// file header). The natural quadrature that pairs with a pointwise
  /// collocation scheme is a uniform dx weight at each of the m=n-1
  /// interior unknowns (the clamped ends contribute zero velocity always,
  /// so they need no weight of their own): KE = 0.5 * rho_s*h_s*dx *
  /// sum_{i=1}^{n-1} wdot_i^2. Used by the transient-energy diagnostics in
  /// test/verification/fsi_transient_test.cpp; not on any solve path.
  [[nodiscard]] double kinetic_energy() const {
    const double mconst = p_.rho_s * p_.h_s;
    double sum = 0.0;
    for (std::size_t i = 1; i < p_.n; ++i) sum += wdot_[i] * wdot_[i];
    return 0.5 * mconst * dx_ * sum;
  }

  /// Discrete strain energy [J, per unit out-of-plane depth]: the SAME
  /// quadratic form the stiffness assembly (`assemble_stiffness`) and the
  /// static/dynamic solves already use, 0.5 * w_interior^T K w_interior --
  /// evaluated here rather than re-derived, so this is guaranteed
  /// consistent with whatever K_ actually is (including the mutation in
  /// build-evidence/mutations/fsi.toml, should it ever be applied).
  [[nodiscard]] double strain_energy() const {
    const std::size_t m = p_.n - 1;
    Eigen::VectorXd wi(m);
    for (std::size_t idx = 0; idx < m; ++idx) wi[idx] = w_[idx + 1];
    return 0.5 * wi.dot(K_ * wi);
  }

  /// The lowest `k` circular natural frequencies [rad/s] of the discrete
  /// (M, K) pencil, via a dense symmetric eigensolve of the (n-1)x(n-1)
  /// interior system -- small by construction (n is an interface resolution,
  /// not a 2D mesh), so a dense solve is the robust choice here. M is a
  /// constant multiple of the identity, so the generalized problem
  /// K w = omega^2 M w reduces to a plain symmetric eigenproblem on
  /// K / (rho_s h_s).
  [[nodiscard]] std::vector<double> natural_frequencies(int k) const {
    const std::size_t m = p_.n - 1;
    Eigen::MatrixXd Kd = Eigen::MatrixXd(K_) / (p_.rho_s * p_.h_s);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Kd);
    if (es.info() != Eigen::Success) {
      throw std::runtime_error("BeamSolver::natural_frequencies: eigensolve "
                                "failed");
    }
    const Eigen::VectorXd ev = es.eigenvalues();  // ascending, = omega^2
    std::vector<double> out;
    const std::size_t take = std::min<std::size_t>(static_cast<std::size_t>(k), m);
    out.reserve(take);
    for (std::size_t i = 0; i < take; ++i) {
      out.push_back(std::sqrt(std::max(0.0, ev[static_cast<Eigen::Index>(i)])));
    }
    return out;
  }

 private:
  /// Reflects a stencil index j (possibly -1 or n+1, the only two ghost
  /// cases the 5-point stencil can ever produce for a clamped end) back into
  /// [0, n]: -1 -> 1 (the slope-BC mirror), n+1 -> n-1. Any other out-of-
  /// range value would be a genuine bug in the caller, not a boundary case,
  /// so it is asserted rather than silently reflected.
  [[nodiscard]] static long long reflect(long long j, long long n) {
    if (j == -1) return 1;
    if (j == n + 1) return n - 1;
    if (j < -1 || j > n + 1) {
      throw std::logic_error("BeamSolver::reflect: index out of the "
                              "5-point stencil's reach");
    }
    return j;
  }

  void assemble_stiffness() {
    const std::size_t m = p_.n - 1;  // interior unknowns: nodes 1..n-1
    const double factor = p_.EI / (dx_ * dx_ * dx_ * dx_);
    const double coeff[5] = {1.0, -4.0, 6.0, -4.0, 1.0};  // offsets -2..2
    std::vector<Eigen::Triplet<double>> trip;
    trip.reserve(m * 5);
    const auto n = static_cast<long long>(p_.n);
    for (std::size_t idx = 0; idx < m; ++idx) {
      const long long node = static_cast<long long>(idx) + 1;
      for (int off = -2; off <= 2; ++off) {
        const long long j = node + off;
        if (j < -1 || j > n + 1) continue;  // never happens for n>=4, m>=3
        const long long j_ref = reflect(j, n);
        if (j_ref == 0 || j_ref == n) continue;  // fixed zero: drop the term
        const auto col = static_cast<std::size_t>(j_ref - 1);
        trip.emplace_back(static_cast<int>(idx), static_cast<int>(col),
                          factor * coeff[off + 2]);
      }
    }
    K_.resize(static_cast<int>(m), static_cast<int>(m));
    K_.setFromTriplets(trip.begin(), trip.end());
    K_.makeCompressed();
  }

  void rebuild_effective(double dt, double beta) {
    if (keff_valid_ && dt == cached_dt_) return;
    const std::size_t m = p_.n - 1;
    const double a0 = 1.0 / (beta * dt * dt);
    const double mconst = p_.rho_s * p_.h_s;
    Eigen::SparseMatrix<double> Keff = K_;
    for (std::size_t idx = 0; idx < m; ++idx) {
      Keff.coeffRef(static_cast<int>(idx), static_cast<int>(idx)) += a0 * mconst;
    }
    Keff.makeCompressed();
    keff_lu_.compute(Keff);
    if (keff_lu_.info() != Eigen::Success) {
      throw std::runtime_error(
          "BeamSolver::rebuild_effective: SparseLU factorization of the "
          "effective Newmark stiffness failed");
    }
    cached_dt_ = dt;
    keff_valid_ = true;
  }

  BeamParams p_;
  double     dx_;
  Eigen::SparseMatrix<double> K_;  // (n-1) x (n-1), interior stiffness

  Eigen::SparseLU<Eigen::SparseMatrix<double>> keff_lu_;
  double cached_dt_ = -1.0;
  bool   keff_valid_ = false;

  std::vector<double> w_, wdot_, wddot_;
};

/// Analytic clamped-clamped static deflection under an AFFINE distributed
/// load q(x) = q0 + q1 * x, derived by direct integration of
/// EI y'''' = q0 + q1 x with y(0)=y'(0)=y(L)=y'(L)=0 (docs/user/fsi-
/// channel.md carries the same derivation for the record). Reduces to the
/// textbook uniform-load case q0 L^4/(384 EI) at x=L/2 when q1=0.
[[nodiscard]] inline double clamped_beam_affine_deflection(double x, double L,
                                                           double EI,
                                                           double q0,
                                                           double q1) {
  const double C1 = -q0 * L / 2.0 - (3.0 / 20.0) * q1 * L * L;
  const double C2 = q0 * L * L / 12.0 + q1 * L * L * L / 30.0;
  const double EIy = q0 * std::pow(x, 4) / 24.0 + q1 * std::pow(x, 5) / 120.0 +
                     C1 * std::pow(x, 3) / 6.0 + C2 * x * x / 2.0;
  return EIy / EI;
}

/// First circular natural frequency of a clamped-clamped Euler-Bernoulli
/// beam, closed form. beta_1 L = 4.730040744862704 is the first positive
/// root of cos(beta L) cosh(beta L) = 1, the clamped-clamped/free-free
/// characteristic equation (Blevins, "Formulas for Natural Frequency and
/// Mode Shape", Table 8-1).
[[nodiscard]] inline double clamped_beam_first_frequency(double L, double EI,
                                                          double rho_s,
                                                          double h_s) {
  constexpr double kBeta1L = 4.730040744862704;
  const double bL2 = kBeta1L * kBeta1L;
  return bL2 * std::sqrt(EI / (rho_s * h_s * L * L * L * L));
}

}  // namespace fsi
