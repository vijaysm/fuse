// Mesh-backed geometric element mappings, independent of field bases.
#ifndef __karma_geometry_map_h__
#define __karma_geometry_map_h__

#include "karma_reference_topology.h"

#include <array>
#include <cstddef>
#include <span>
#include <vector>

namespace KARMA
{
class KARMAMesh;

/**
 * Validation is adaptive sampling, not a formal global injectivity certificate.
 * Polynomial cells use nested reference grids; pyramids additionally use
 * collapsed-coordinate levels approaching the apex.
 */
enum class GeometryValidityPolicy
{
    PolynomialAdaptiveSampling,
    RationalPyramidAdaptiveSampling
};

enum class InverseMapStatus
{
    Located,
    NotContained,
    InvalidGeometry
};

struct InverseMapResult
{
    InverseMapStatus status = InverseMapStatus::NotContained;
    Point reference_point = Point::Zero();
    Real residual = 0.0;
};

struct SideGeometry
{
    Point point = Point::Zero();
    Matrix3 tangents = Matrix3::Zero();
    Vector3 normal = Vector3::Zero();
    Real measure = 0.0;
};

/**
 * Reusable geometry evaluator. Const evaluation updates internal fixed scratch,
 * so one instance must not be shared concurrently; use one instance per thread.
 */
class KARMAGeometryMap
{
  public:
    KARMAGeometryMap( KARMAReferenceTopology reference, std::span< const Point > nodes );
    explicit KARMAGeometryMap( KARMAReferenceTopology reference );
    KARMAGeometryMap( const KARMAMesh& mesh, moab::EntityHandle element );
    void reinit( std::span< const Point > nodes );

    [[nodiscard]] GeometryVariant variant() const noexcept { return _reference.variant(); }
    [[nodiscard]] moab::EntityType topology() const noexcept { return _reference.topology(); }
    [[nodiscard]] unsigned order() const noexcept { return _reference.order(); }
    [[nodiscard]] unsigned reference_dimension() const noexcept { return _reference.dimension(); }
    [[nodiscard]] std::size_t node_count() const noexcept { return _node_count; }
    [[nodiscard]] const KARMAReferenceTopology& reference_topology() const noexcept { return _reference; }
    [[nodiscard]] std::span< const Point > nodes() const noexcept { return { _nodes.data(), _node_count }; }
    [[nodiscard]] GeometryValidityPolicy validity_policy() const noexcept
    { return topology() == moab::MBPYRAMID ? GeometryValidityPolicy::RationalPyramidAdaptiveSampling
                                           : GeometryValidityPolicy::PolynomialAdaptiveSampling; }

    [[nodiscard]] Point map( const Point& reference_point ) const;
    /** Undefined at the exact apex of a pyramid because rational derivatives are path-dependent. */
    [[nodiscard]] Matrix3 jacobian( const Point& reference_point ) const;
    [[nodiscard]] Matrix3 metric( const Point& reference_point ) const;
    [[nodiscard]] Real measure( const Point& reference_point ) const;
    [[nodiscard]] Point inverse_map( const Point& physical_point ) const;
    [[nodiscard]] InverseMapResult try_inverse_map( const Point& physical_point, Real reference_tolerance,
                                                    Real residual_tolerance ) const noexcept;
    [[nodiscard]] SideGeometry side( unsigned side, const Point& side_point ) const;
    [[nodiscard]] SideGeometry side_from_parent( unsigned side, const Point& parent_point,
                                                 const Matrix3& parent_side_jacobian ) const;

  private:
    void update_affine_quad4_cache();
    void validate() const;

    KARMAReferenceTopology _reference;
    std::array< Point, 27 > _nodes{};
    std::size_t _node_count{};
    bool _has_affine_quad4_cache{};
    Point _affine_quad4_origin = Point::Zero();
    Vector3 _affine_quad4_r = Vector3::Zero();
    Vector3 _affine_quad4_s = Vector3::Zero();
    Matrix3 _affine_quad4_jacobian = Matrix3::Zero();
    Matrix3 _affine_quad4_metric = Matrix3::Zero();
    Real _affine_quad4_measure{};
    mutable std::array< Real, 27 > _values{};
    mutable std::array< Point, 27 > _derivatives{};
};
}  // namespace KARMA

#endif
