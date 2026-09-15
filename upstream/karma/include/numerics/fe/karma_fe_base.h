// KARMA finite-element basis evaluation.

#ifndef __karma_fe_base_h__
#define __karma_fe_base_h__

#include "karma_basis.h"
#include "karma_geometry_map.h"
#include "karma_quadrature.h"

#include "moab/Types.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <span>
#include <vector>

namespace KARMA
{
class KARMAMesh;

enum class FEIntegrationDomain
{
    Volume,
    Face
};

/**
 * Geometry-independent scalar field evaluation on a fixed reference cell and
 * geometry variant. Output storage is allocated at construction and remains
 * address-stable across successful and failed reinitialization. This API requires
 * KARMA's exception-enabled build; exception-off operation is not supported.
 */
class KARMAFEBase
{
  public:
    KARMAFEBase( const KARMAMesh& mesh, GeometryVariant geometry_variant, KARMABasis basis, int quadrature_degree,
                 FEIntegrationDomain domain = FEIntegrationDomain::Volume );
    ~KARMAFEBase() noexcept = default;

    KARMAFEBase( const KARMAFEBase& )            = delete;
    KARMAFEBase& operator=( const KARMAFEBase& ) = delete;
    KARMAFEBase( KARMAFEBase&& )                 = delete;
    KARMAFEBase& operator=( KARMAFEBase&& )      = delete;

    void reinit( moab::EntityHandle element );

    /**
     * Evaluate a parent basis on a side. All getters, including JxW and normals, are valid.
     * Getter vector sizes are the active native quadrature count. Resizing stays within reserved
     * construction-time capacity, preserving vector objects and data pointers without allocation.
     */
    void reinit( moab::EntityHandle element, unsigned side );

    /**
     * Evaluate a parent basis at any nonempty caller-provided physical point range up to
     * construction-time capacity. Getter sizes become that active count; phi, dphi, and xyz
     * are valid while JxW and normals are NaN because arbitrary points have no side geometry.
     */
    void reinit_at_physical_points( moab::EntityHandle element, std::span< const Point > points );

    [[nodiscard]] bool is_valid() const noexcept
    { return _valid; }
    [[nodiscard]] std::size_t n_quadrature_points() const noexcept
    { return _active_qpoints; }

    [[nodiscard]] const std::vector< Real >& get_JxW() const noexcept
    { return _jxw; }
    [[nodiscard]] const std::vector< std::vector< Real > >& get_phi() const noexcept
    { return _phi; }
    [[nodiscard]] const std::vector< std::vector< Gradient > >& get_dphi() const noexcept
    { return _dphi; }
    [[nodiscard]] const std::vector< Point >& get_xyz() const noexcept
    { return _xyz; }
    [[nodiscard]] const std::vector< Vector3 >& get_normals() const noexcept
    { return _normals; }

    [[nodiscard]] moab::EntityType topology() const noexcept
    { return _topology; }
    [[nodiscard]] GeometryVariant geometry_variant() const noexcept
    { return _geometry_variant; }
    [[nodiscard]] const KARMABasis& basis() const noexcept
    { return _basis; }

  private:
    void invalidate() noexcept;
    void set_active_size( std::size_t count );
    void evaluate_reference_points( const KARMAGeometryMap& geometry, PetscQuadrature quadrature,
                                    bool compute_weights );
    void evaluate_reference_points( const KARMAGeometryMap& geometry, std::span< const Point > points );
    void evaluate_physical_points( const KARMAGeometryMap& geometry, std::span< const Point > points,
                                   bool preserve_face_geometry );

    const KARMAMesh& _mesh;
    GeometryVariant _geometry_variant;
    KARMAReferenceTopology _geometry_reference;
    KARMABasis _basis;
    KARMAGeometryMap _geometry;
    moab::EntityType _topology;
    KARMAQuadrature _quadrature;
    std::unique_ptr< KARMAQuadrature > _triangle_quadrature;
    FEIntegrationDomain _domain;
    std::size_t _n_basis;
    std::size_t _capacity;
    std::size_t _active_qpoints;
    bool _valid = false;

    std::vector< Real > _jxw;
    std::vector< std::vector< Real > > _phi;
    std::vector< std::vector< Gradient > > _dphi;
    std::vector< Point > _xyz;
    std::vector< Vector3 > _normals;

    std::vector< Real > _coordinates;
    std::vector< Real > _reference_phi;
    std::vector< Gradient > _reference_gradient;
    std::array< Point, 27 > _geometry_nodes{};
    std::vector< Point > _side_parent_points;
    std::vector< Matrix3 > _side_parent_jacobians;
    std::vector< Point > _physical_point_scratch;
    std::vector< Point > _reference_point_scratch;
};
}  // namespace KARMA

#endif  // __karma_fe_base_h__
