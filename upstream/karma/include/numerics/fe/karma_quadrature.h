// KARMA finite-element quadrature.

#ifndef __karma_quadrature_h__
#define __karma_quadrature_h__

#include "karma_exceptions.h"

#include "moab/Types.hpp"
#include <petscdt.h>

#include <algorithm>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace KARMA
{
/** RAII quadrature wrapper. Requires KARMA's exception-enabled build. */
class KARMAQuadrature
{
  public:
    KARMAQuadrature( moab::EntityType topology, int degree )
        : _topology( topology ), _degree( degree ), _dimension( topology_dimension( topology ) )
    {
        constexpr int maximum_degree = 64;
        if( degree < 0 || degree > maximum_degree )
            KARMA_THROW( InvalidArgument( "quadrature degree must be in [0, 64]" ) );

        PetscQuadrature candidate = nullptr;
        PetscReal* pending_points = nullptr;
        PetscReal* pending_weights = nullptr;
        try
        {
            if( topology == moab::MBVERTEX )
            {
                check_petsc( PetscMalloc1( 1, &pending_points ), "allocate point quadrature point" );
                check_petsc( PetscMalloc1( 1, &pending_weights ), "allocate point quadrature weight" );
                pending_points[0] = 0.0;
                pending_weights[0] = 1.0;
                check_petsc( PetscQuadratureCreate( PETSC_COMM_SELF, &candidate ), "create point quadrature" );
                check_petsc( PetscQuadratureSetCellType( candidate, DM_POLYTOPE_POINT ), "set point quadrature cell type" );
                check_petsc( PetscQuadratureSetData( candidate, 0, 1, 1, pending_points, pending_weights ),
                             "set point quadrature data" );
                pending_points = nullptr;
                pending_weights = nullptr;
            }
            else if( topology == moab::MBEDGE || topology == moab::MBQUAD || topology == moab::MBHEX )
            {
                const PetscInt points_per_axis = std::max( 1, ( degree + 2 ) / 2 );
                check_petsc( PetscDTGaussTensorQuadrature( static_cast< PetscInt >( _dimension ), 1, points_per_axis,
                                                           0.0, 1.0, &candidate ),
                             "create tensor quadrature" );
            }
            else if( topology == moab::MBTRI || topology == moab::MBTET )
            {
                check_petsc( PetscDTSimplexQuadrature( static_cast< PetscInt >( _dimension ), degree,
                                                       PETSCDTSIMPLEXQUAD_DEFAULT, &candidate ),
                             "create simplex quadrature" );
                candidate = normalize_simplex( candidate );
            }
            else
                candidate = create_product_quadrature();

            PetscInt dim = 0;
            check_petsc( PetscQuadratureGetData( candidate, &dim, nullptr, &_n_points, nullptr, nullptr ),
                         "query quadrature" );
            if( dim != static_cast< PetscInt >( _dimension ) || _n_points <= 0 ) KARMA_THROW( LogicError() );
            _quadrature = std::exchange( candidate, nullptr );
        }
        catch( ... )
        {
            PetscFree( pending_points );
            PetscFree( pending_weights );
            if( candidate ) PetscQuadratureDestroy( &candidate );
            throw;
        }
    }

    /** Owns a PETSc object and must be destroyed before PetscFinalize(); copying and moving are forbidden. */
    ~KARMAQuadrature() noexcept
    {
        if( !_quadrature ) return;
        const PetscErrorCode ierr = PetscQuadratureDestroy( &_quadrature );
#ifndef NDEBUG
        if( ierr != PETSC_SUCCESS )
            std::fprintf( stderr, "KARMAQuadrature: PetscQuadratureDestroy failed (%d)\n", ierr );
#else
        (void)ierr;
#endif
    }

    KARMAQuadrature( const KARMAQuadrature& )            = delete;
    KARMAQuadrature& operator=( const KARMAQuadrature& ) = delete;
    KARMAQuadrature( KARMAQuadrature&& )                 = delete;
    KARMAQuadrature& operator=( KARMAQuadrature&& )      = delete;

    [[nodiscard]] unsigned dimension() const noexcept
    { return _dimension; }
    [[nodiscard]] int degree() const noexcept
    { return _degree; }
    [[nodiscard]] moab::EntityType topology() const noexcept
    { return _topology; }
    [[nodiscard]] std::size_t n_points() const noexcept
    { return static_cast< std::size_t >( _n_points ); }
    [[nodiscard]] PetscQuadrature petsc_quadrature() const noexcept
    { return _quadrature; }

  private:
    static unsigned topology_dimension( moab::EntityType topology )
    {
        switch( topology )
        {
            case moab::MBVERTEX:
                return 0;
            case moab::MBEDGE:
                return 1;
            case moab::MBTRI:
            case moab::MBQUAD:
                return 2;
            case moab::MBTET:
            case moab::MBHEX:
            case moab::MBPRISM:
            case moab::MBPYRAMID:
                return 3;
            default:
                KARMA_THROW( InvalidArgument( "quadrature requires edge, tri, quad, tet, hex, prism, or pyramid topology" ) );
        }
    }

    /** Prism and pyramid rules.
     *
     * Both axes are sized `(degree + 2) / 2`, the same as edge, quad and hex,
     * and neither topology needs more than that.
     *
     * The prism never did: it is a genuine triangle-by-line tensor product
     * whose weights are `triangle_weight * line_weight` with no Jacobian
     * factor, so the line axis integrates a degree-`degree` polynomial with
     * the standard count.
     *
     * The pyramid is the interesting one. Its collapsed (Duffy) map
     * `x = (2u-1)h, y = (2v-1)h, h = 1-z` contributes an `h^2` Jacobian. That
     * factor used to be carried numerically in the weight, which raised the
     * integrand in `z` by two degrees and so required an extra point on
     * *every* axis - `(degree + 4) / 2` cubed, 64 points at degree 4 where a
     * hex uses 27. Gauss-Jacobi with `alpha = 2` integrates `(1-z)^2` exactly
     * by construction, so absorbing the Jacobian into the weight function
     * recovers the standard count with a degree of headroom to spare.
     *
     * PETSc's Gauss-Jacobi weights already include the `2^alpha (1-z)^alpha`
     * factor - for `alpha = 2` on [0,1] they sum to 4/3, not 1/3 - which is
     * precisely the `4 * h * h` this function used to apply by hand. That is
     * why the pyramid weight below is a plain product of the three axis
     * weights: multiplying by `4 * h * h` again would apply the Jacobian
     * twice. Verified against the monomial basis: exact to 2e-16 through
     * degree 6. See docs/developer-guide/pyramid-quadrature-cost.md.
     */
    PetscQuadrature create_product_quadrature()
    {
        const PetscInt points_per_axis = std::max( 1, ( _degree + 2 ) / 2 );
        PetscQuadrature line = nullptr;
        PetscQuadrature triangle = nullptr;
        PetscQuadrature result = nullptr;
        PetscReal* points = nullptr;
        PetscReal* weights = nullptr;
        PetscReal* collapsed_points = nullptr;
        PetscReal* collapsed_weights = nullptr;
        try
        {
            check_petsc( PetscDTGaussTensorQuadrature( 1, 1, points_per_axis, 0.0, 1.0, &line ),
                         "create product line quadrature" );
            PetscInt line_count = 0;
            const PetscReal* line_points = nullptr;
            const PetscReal* line_weights = nullptr;
            check_petsc( PetscQuadratureGetData( line, nullptr, nullptr, &line_count, &line_points, &line_weights ),
                         "query product line quadrature" );

            PetscInt triangle_count = 1;
            const PetscReal* triangle_points = nullptr;
            const PetscReal* triangle_weights = nullptr;
            if( _topology == moab::MBPRISM )
            {
                check_petsc( PetscDTSimplexQuadrature( 2, _degree, PETSCDTSIMPLEXQUAD_DEFAULT, &triangle ),
                             "create prism triangle quadrature" );
                triangle = normalize_simplex( triangle );
                check_petsc( PetscQuadratureGetData( triangle, nullptr, nullptr, &triangle_count, &triangle_points,
                                                     &triangle_weights ),
                             "query prism triangle quadrature" );
            }

            // The pyramid's collapsed axis. Allocated before the point and
            // weight buffers so the catch block has a single cleanup path.
            if( _topology != moab::MBPRISM )
            {
                check_petsc( PetscMalloc1( points_per_axis, &collapsed_points ),
                             "allocate pyramid collapsed-axis points" );
                const PetscErrorCode collapsed_error = PetscMalloc1( points_per_axis, &collapsed_weights );
                if( collapsed_error != PETSC_SUCCESS )
                {
                    PetscFree( collapsed_points );
                    collapsed_points = nullptr;
                    check_petsc( collapsed_error, "allocate pyramid collapsed-axis weights" );
                }
                check_petsc( PetscDTGaussJacobiQuadrature( points_per_axis, 0.0, 1.0, 2.0, 0.0, collapsed_points,
                                                           collapsed_weights ),
                             "create pyramid collapsed-axis quadrature" );
            }

            const PetscInt count = _topology == moab::MBPRISM ? triangle_count * line_count
                                                               : line_count * line_count * points_per_axis;
            check_petsc( PetscMalloc1( 3 * count, &points ), "allocate product quadrature points" );
            const PetscErrorCode weight_error = PetscMalloc1( count, &weights );
            if( weight_error != PETSC_SUCCESS )
            {
                PetscFree( points );
                points = nullptr;
                check_petsc( weight_error, "allocate product quadrature weights" );
            }
            PetscInt q = 0;
            if( _topology == moab::MBPRISM )
                for( PetscInt k = 0; k < line_count; ++k )
                    for( PetscInt i = 0; i < triangle_count; ++i, ++q )
                    {
                        points[3 * q] = triangle_points[2 * i];
                        points[3 * q + 1] = triangle_points[2 * i + 1];
                        points[3 * q + 2] = line_points[k];
                        weights[q] = triangle_weights[i] * line_weights[k];
                    }
            else
                for( PetscInt k = 0; k < points_per_axis; ++k )
                    for( PetscInt j = 0; j < line_count; ++j )
                        for( PetscInt i = 0; i < line_count; ++i, ++q )
                        {
                            const PetscReal height = 1.0 - collapsed_points[k];
                            points[3 * q] = ( 2.0 * line_points[i] - 1.0 ) * height;
                            points[3 * q + 1] = ( 2.0 * line_points[j] - 1.0 ) * height;
                            points[3 * q + 2] = collapsed_points[k];
                            // No 4 * height * height here: the Gauss-Jacobi
                            // weight already carries it, as described above.
                            weights[q] = line_weights[i] * line_weights[j] * collapsed_weights[k];
                        }
            check_petsc( PetscQuadratureCreate( PETSC_COMM_SELF, &result ), "create product quadrature" );
            check_petsc( PetscQuadratureSetCellType( result, _topology == moab::MBPRISM ? DM_POLYTOPE_TRI_PRISM
                                                                                       : DM_POLYTOPE_PYRAMID ),
                         "set product quadrature cell type" );
            check_petsc( PetscQuadratureSetOrder( result, _degree ), "set product quadrature order" );
            check_petsc( PetscQuadratureSetData( result, 3, 1, count, points, weights ),
                         "set product quadrature data" );
            points = nullptr;
            weights = nullptr;
            PetscFree( collapsed_points );
            PetscFree( collapsed_weights );
        }
        catch( ... )
        {
            PetscFree( points );
            PetscFree( weights );
            PetscFree( collapsed_points );
            PetscFree( collapsed_weights );
            if( result ) PetscQuadratureDestroy( &result );
            if( triangle ) PetscQuadratureDestroy( &triangle );
            if( line ) PetscQuadratureDestroy( &line );
            throw;
        }
        PetscQuadratureDestroy( &triangle );
        PetscQuadratureDestroy( &line );
        return result;
    }

    PetscQuadrature normalize_simplex( PetscQuadrature& source )
    {
        PetscInt dim                 = 0;
        PetscInt components          = 0;
        PetscInt count               = 0;
        const PetscReal* old_points  = nullptr;
        const PetscReal* old_weights = nullptr;
        check_petsc( PetscQuadratureGetData( source, &dim, &components, &count, &old_points, &old_weights ),
                     "query simplex quadrature" );

        PetscReal* points          = nullptr;
        PetscReal* weights         = nullptr;
        PetscQuadrature normalized = nullptr;
        try
        {
            check_petsc( PetscMalloc1( dim * count, &points ), "allocate simplex points" );
            const PetscErrorCode weight_error = PetscMalloc1( components * count, &weights );
            if( weight_error != PETSC_SUCCESS )
            {
                PetscFree( points );
                points = nullptr;
                check_petsc( weight_error, "allocate simplex weights" );
            }
            const PetscReal weight_scale = PetscPowReal( 0.5, dim );
            for( PetscInt q = 0; q < count; ++q )
            {
                for( PetscInt d = 0; d < dim; ++d )
                    points[q * dim + d] = 0.5 * ( old_points[q * dim + d] + 1.0 );
                for( PetscInt c = 0; c < components; ++c )
                    weights[q * components + c] = weight_scale * old_weights[q * components + c];
            }
            check_petsc( PetscQuadratureCreate( PETSC_COMM_SELF, &normalized ),
                         "create normalized simplex quadrature" );
            DMPolytopeType cell_type = DM_POLYTOPE_UNKNOWN;
            check_petsc( PetscQuadratureGetCellType( source, &cell_type ), "query simplex cell type" );
            check_petsc( PetscQuadratureSetCellType( normalized, cell_type ), "set simplex cell type" );
            check_petsc( PetscQuadratureSetOrder( normalized, _degree ), "set simplex quadrature order" );
            check_petsc( PetscQuadratureSetData( normalized, dim, components, count, points, weights ),
                         "set normalized simplex quadrature data" );
            points  = nullptr;
            weights = nullptr;
        }
        catch( ... )
        {
            PetscFree( points );
            PetscFree( weights );
            if( normalized ) PetscQuadratureDestroy( &normalized );
            throw;
        }
        const PetscErrorCode destroy_error = PetscQuadratureDestroy( &source );
        if( destroy_error != PETSC_SUCCESS )
        {
            PetscQuadratureDestroy( &normalized );
            check_petsc( destroy_error, "destroy unnormalized simplex quadrature" );
        }
        return normalized;
    }

    static void check_petsc( PetscErrorCode ierr, const char* operation )
    {
        if( ierr != PETSC_SUCCESS ) KARMA_THROW( InvalidArgument( std::string( "PETSc failed to " ) + operation ) );
    }

    PetscQuadrature _quadrature = nullptr;
    moab::EntityType _topology;
    int _degree;
    unsigned _dimension;
    PetscInt _n_points = 0;
};
}  // namespace KARMA

#endif  // __karma_quadrature_h__
