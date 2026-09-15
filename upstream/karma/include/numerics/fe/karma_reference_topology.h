// Reference cells and geometry interpolation functions.
#ifndef __karma_reference_topology_h__
#define __karma_reference_topology_h__

#include "karma_geometry.h"
#include "moab/Types.hpp"

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

namespace KARMA
{
enum class GeometryVariant
{
    Edge2,
    Edge3,
    Tri3,
    Tri6,
    Quad4,
    Quad8,
    Quad9,
    Tet4,
    Tet10,
    Hex8,
    Hex27,
    Prism6,
    Prism15,
    Prism18,
    Pyramid5,
    Pyramid13,
    Pyramid14
};

struct ReferenceValues
{
    std::vector< Real > values;
    std::vector< Point > derivatives;
};

class KARMAReferenceTopology
{
  public:
    explicit KARMAReferenceTopology( GeometryVariant variant );

    [[nodiscard]] static GeometryVariant from_moab( moab::EntityType topology, std::size_t node_count );
    [[nodiscard]] GeometryVariant variant() const noexcept { return _variant; }
    [[nodiscard]] moab::EntityType topology() const noexcept;
    [[nodiscard]] unsigned dimension() const noexcept;
    [[nodiscard]] unsigned order() const noexcept;
    [[nodiscard]] std::size_t node_count() const noexcept;
    [[nodiscard]] const std::vector< Point >& reference_nodes() const noexcept { return _nodes; }
    /** Map each supplied nodal point to its unique geometry connectivity ordinal. */
    [[nodiscard]] std::optional< std::vector< std::size_t > >
    node_bijection( std::span< const Point > points ) const;
    [[nodiscard]] Point center() const noexcept;
    /** Shape values only. This is well-defined at a pyramid apex. */
    [[nodiscard]] std::vector< Real > evaluate_values( const Point& point ) const;
    void evaluate_values( const Point& point, std::span< Real > values ) const;
    /** Shape values and derivatives. Rejects the exact pyramid apex. */
    [[nodiscard]] ReferenceValues evaluate( const Point& point ) const;
    void evaluate( const Point& point, std::span< Real > values, std::span< Point > derivatives ) const;

    [[nodiscard]] unsigned side_count() const noexcept;
    [[nodiscard]] KARMAReferenceTopology side_topology( unsigned side ) const;
    [[nodiscard]] std::vector< std::size_t > side_nodes( unsigned side ) const;
    [[nodiscard]] Point side_reference_map( unsigned side, const Point& point ) const;
    [[nodiscard]] Matrix3 side_reference_jacobian( unsigned side, const Point& point ) const;

  private:
    GeometryVariant _variant;
    std::vector< Point > _nodes;
};
}  // namespace KARMA

#endif
