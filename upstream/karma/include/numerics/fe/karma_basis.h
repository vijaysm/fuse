// Geometry-independent scalar finite-element bases.
#ifndef __karma_basis_h__
#define __karma_basis_h__

#include "karma_exceptions.h"
#include "karma_geometry.h"
#include "karma_reference_topology.h"

#include <array>
#include <compare>
#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace KARMA
{
enum class ReferenceCell { Edge, Triangle, Quadrilateral, Tetrahedron, Hexahedron, Prism, Pyramid };
enum class FieldBasisFamily { LagrangeNodal, Hierarchical, Legendre, SpectralElement, UserDefined };
// P0 is cell-local by definition, so built-in P0 keys are available only for dG and SE.
enum class BasisContinuity { cG, dG, SE };
enum class CoefficientRepresentation { Nodal, Modal };
enum class EdgeOrientation : unsigned { Direct, Reversed };
enum class TriangleOrientation : unsigned { Rotate0, Rotate1, Rotate2, ReflectRotate0, ReflectRotate1, ReflectRotate2 };
enum class QuadrilateralOrientation : unsigned
{
    Rotate0, Rotate1, Rotate2, Rotate3, ReflectRotate0, ReflectRotate1, ReflectRotate2, ReflectRotate3
};

// These permutations map local vertex indices to canonical indices. RotateK maps i to
// (i + K) mod N; ReflectRotateK maps i to (K - i) mod N for cyclic vertex order.
[[nodiscard]] constexpr std::array< unsigned, 2 > local_to_canonical_vertices( EdgeOrientation orientation )
{
    constexpr std::array< std::array< unsigned, 2 >, 2 > permutations{{ {0,1}, {1,0} }};
    return permutations.at( static_cast< std::size_t >( orientation ) );
}
[[nodiscard]] constexpr std::array< unsigned, 3 > local_to_canonical_vertices( TriangleOrientation orientation )
{
    constexpr std::array< std::array< unsigned, 3 >, 6 > permutations{{
        {0,1,2}, {1,2,0}, {2,0,1}, {0,2,1}, {1,0,2}, {2,1,0}
    }};
    return permutations.at( static_cast< std::size_t >( orientation ) );
}
[[nodiscard]] constexpr std::array< unsigned, 4 > local_to_canonical_vertices( QuadrilateralOrientation orientation )
{
    constexpr std::array< std::array< unsigned, 4 >, 8 > permutations{{
        {0,1,2,3}, {1,2,3,0}, {2,3,0,1}, {3,0,1,2},
        {0,3,2,1}, {1,0,3,2}, {2,1,0,3}, {3,2,1,0}
    }};
    return permutations.at( static_cast< std::size_t >( orientation ) );
}

[[nodiscard]] ReferenceCell reference_cell( GeometryVariant variant ) noexcept;

inline constexpr unsigned maximum_basis_order = 1024;

struct BasisKey
{
    ReferenceCell cell;
    FieldBasisFamily family;
    unsigned order;
    BasisContinuity continuity;

    auto operator<=>( const BasisKey& ) const = default;
};

struct DofAssociation
{
    unsigned entity_dimension;
    unsigned local_entity_index;
    unsigned slot;

    bool operator==( const DofAssociation& ) const = default;
};

struct OrientationTransform
{
    unsigned entity_dimension;
    unsigned local_entity_index;
    unsigned orientation_code;
    std::vector< std::size_t > local_to_canonical;
    std::vector< int > signs;

    bool operator==( const OrientationTransform& ) const = default;
};

class KARMABasis
{
  public:
    template < class Basis > explicit KARMABasis( Basis basis )
        : KARMABasis( std::make_shared< Model< Basis > >( std::move( basis ) ), 0 )
    {
    }

    [[nodiscard]] std::string name() const { return _self->name(); }
    [[nodiscard]] std::string compatibility_id() const { return _self->compatibility_id(); }
    [[nodiscard]] BasisKey key() const noexcept { return _key; }
    [[nodiscard]] CoefficientRepresentation coefficient_representation() const noexcept
    { return _self->coefficient_representation(); }
    [[nodiscard]] bool supports( const BasisKey& candidate ) const noexcept { return key() == candidate; }
    [[nodiscard]] std::size_t size() const noexcept { return _size; }
    void evaluate_values( const Point& point, std::span< Real > values ) const
    {
        if( values.size() != size() ) KARMA_THROW( InvalidArgument( "basis output storage must equal basis size" ) );
        _self->evaluate_values( point, values );
    }
    void evaluate( const Point& point, std::span< Real > values, std::span< Gradient > gradients ) const
    {
        if( values.size() != size() || gradients.size() != size() )
            KARMA_THROW( InvalidArgument( "basis output storage must equal basis size" ) );
        _self->evaluate( point, values, gradients );
    }
    [[nodiscard]] std::vector< Point > interpolation_points() const { return _interpolation_points; }
    [[nodiscard]] std::vector< DofAssociation > dof_associations() const { return _dof_associations; }
    [[nodiscard]] std::vector< OrientationTransform > orientation_transforms() const { return _orientation_transforms; }
    void apply_orientation( unsigned entity_dimension, unsigned local_entity_index, unsigned orientation_code,
                             std::span< const Real > local, std::span< Real > canonical ) const;
    void apply_orientation_inverse( unsigned entity_dimension, unsigned local_entity_index, unsigned orientation_code,
                                    std::span< const Real > canonical, std::span< Real > local ) const;


  private:
    struct Concept
    {
        virtual ~Concept() = default;
        [[nodiscard]] virtual std::string name() const = 0;
        [[nodiscard]] virtual std::string compatibility_id() const = 0;
        [[nodiscard]] virtual BasisKey key() const noexcept = 0;
        [[nodiscard]] virtual CoefficientRepresentation coefficient_representation() const noexcept = 0;
        [[nodiscard]] virtual std::size_t size() const noexcept = 0;
        virtual void evaluate_values( const Point&, std::span< Real > ) const = 0;
        virtual void evaluate( const Point&, std::span< Real >, std::span< Gradient > ) const = 0;
        [[nodiscard]] virtual std::vector< Point > interpolation_points() const = 0;
        [[nodiscard]] virtual std::vector< DofAssociation > dof_associations() const = 0;
        [[nodiscard]] virtual std::vector< OrientationTransform > orientation_transforms() const = 0;
    };

    template < class Basis > struct Model final : Concept
    {
        explicit Model( Basis value ) : basis( std::move( value ) ) {}
        [[nodiscard]] std::string name() const override { return basis.name(); }
        [[nodiscard]] std::string compatibility_id() const override { return basis.compatibility_id(); }
        [[nodiscard]] BasisKey key() const noexcept override { return basis.key(); }
        [[nodiscard]] CoefficientRepresentation coefficient_representation() const noexcept override
        { return basis.coefficient_representation(); }
        [[nodiscard]] std::size_t size() const noexcept override { return basis.size(); }
        void evaluate_values( const Point& point, std::span< Real > values ) const override
        {
            basis.evaluate_values( point, values );
        }
        void evaluate( const Point& point, std::span< Real > values,
                       std::span< Gradient > gradients ) const override
        {
            basis.evaluate( point, values, gradients );
        }
        [[nodiscard]] std::vector< Point > interpolation_points() const override
        {
            return basis.interpolation_points();
        }
        [[nodiscard]] std::vector< DofAssociation > dof_associations() const override
        {
            return basis.dof_associations();
        }
        [[nodiscard]] std::vector< OrientationTransform > orientation_transforms() const override
        {
            return basis.orientation_transforms();
        }
        Basis basis;
    };

    explicit KARMABasis( std::shared_ptr< const Concept > self, int );

    std::shared_ptr< const Concept > _self;
    BasisKey _key;
    std::size_t _size{};
    std::vector< Point > _interpolation_points;
    std::vector< DofAssociation > _dof_associations;
    std::vector< OrientationTransform > _orientation_transforms;
};

enum class DuplicatePolicy { Reject, Replace };

class KARMABasisRegistry
{
  public:
    using Factory = std::function< KARMABasis() >;

    KARMABasisRegistry();
    void register_factory( const BasisKey& key, Factory factory,
                           DuplicatePolicy policy = DuplicatePolicy::Reject );
    [[nodiscard]] bool supports( const BasisKey& key ) const noexcept;
    [[nodiscard]] KARMABasis create( const BasisKey& key ) const;

  private:
    std::map< BasisKey, Factory > _factories;
};
}  // namespace KARMA

#endif
