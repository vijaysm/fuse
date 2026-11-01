// fuse_physics::fe::element_mass_matrix / element_stiffness_matrix --
// adapted from KARMA's include/physics/common/generic_mass_operator.h.
// See generic_operators.hpp's own header comment for what is and is not
// a literal port here.

#include "fuse_physics/fe/generic_operators.hpp"

namespace fuse::physics::fe {

std::vector<double> element_mass_matrix(FEBase& fe, spatial::EntityHandle element) {
  fe.reinit(element);
  const auto& phi = fe.get_phi();
  const auto& jxw = fe.get_JxW();
  const std::size_t n = phi.size();
  std::vector<double> mass(n * n, 0.0);
  for (std::size_t q = 0; q < fe.n_quadrature_points(); ++q) {
    const double weight = jxw[q];
    for (std::size_t i = 0; i < n; ++i) {
      const double weighted_phi_i = weight * phi[i][q];
      for (std::size_t j = 0; j < n; ++j) mass[i * n + j] += weighted_phi_i * phi[j][q];
    }
  }
  return mass;
}

std::vector<double> element_stiffness_matrix(FEBase& fe, spatial::EntityHandle element) {
  fe.reinit(element);
  const auto& dphi = fe.get_dphi();
  const auto& jxw = fe.get_JxW();
  const std::size_t n = dphi.size();
  std::vector<double> stiffness(n * n, 0.0);
  for (std::size_t q = 0; q < fe.n_quadrature_points(); ++q) {
    const double weight = jxw[q];
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) stiffness[i * n + j] += weight * dot(dphi[i][q], dphi[j][q]);
    }
  }
  return stiffness;
}

}  // namespace fuse::physics::fe
