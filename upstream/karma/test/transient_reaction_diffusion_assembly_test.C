#include "transient_reaction_diffusion_assembly.h"
#include "karma_mesh.h"
#include "karma_system.h"

#include <petsc.h>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace
{
void check_close( int& failures, KARMA::Real actual, KARMA::Real expected, const char* quantity )
{
    const KARMA::Real tolerance = 2.e-13 * std::max( KARMA::Real( 1 ), std::abs( expected ) );
    if( !std::isfinite( actual ) || std::abs( actual - expected ) > tolerance )
    {
        std::cerr << quantity << ": expected " << expected << ", got " << actual << '\n';
        ++failures;
    }
}
}

int main( int argc, char** argv )
{
    if( PetscInitialize( &argc, &argv, nullptr, nullptr ) ) return 1;
    int failures = 0;
    {
        KARMA::KARMAMesh mesh( 2, PETSC_COMM_WORLD );
        mesh.generate_box( 1, 1, 0 );
        KARMA::KARMASystem system( mesh, "transient assembly test" );
        system.add_variable( "u" );
        system.init();
        const auto element = *mesh.elements().begin();
        KARMA::RealDenseVector left( 4 ), right( 4 ), input( 4 );
        const KARMA::Real left_data[4] = { .2, -.4, .7, 1.1 };
        const KARMA::Real right_data[4] = { -.3, .6, .9, .1 };
        const KARMA::Real input_data[4] = { .5, -.2, .8, .4 };
        for( int i = 0; i < 4; ++i )
        {
            left( i ) = left_data[i];
            right( i ) = right_data[i];
            input( i ) = input_data[i];
        }

        constexpr KARMA::Real h = .25;
        constexpr KARMA::Real alpha = 2.3;
        constexpr KARMA::Real beta = -.7;
        constexpr KARMA::Real source = .4;
        KARMA::TransientReactionDiffusionModel model;
        model.diffusion = .25;
        model.reaction = []( const KARMA::Point&, KARMA::Real state, KARMA::Real coupled, KARMA::Real ) {
            return alpha * state + beta * coupled + source;
        };
        model.state_derivative = []( const KARMA::Point&, KARMA::Real, KARMA::Real, KARMA::Real ) { return alpha; };
        const auto terms = KARMA::element_transient_reaction_diffusion(
            system, element, model, { 0, h, left, right, input } );

        const KARMA::Real mass_data[4][4] = {
            { 4, 2, 1, 2 }, { 2, 4, 2, 1 }, { 1, 2, 4, 2 }, { 2, 1, 2, 4 } };
        KARMA::RealDenseMatrix mass( 4, 4 );
        for( int i = 0; i < 4; ++i )
            for( int j = 0; j < 4; ++j ) mass( i, j ) = mass_data[i][j] / 36.;
        const KARMA::Real expected_jacobian_factor = 1 / h - alpha / 2;
        KARMA::TransientReactionDiffusionModel no_diffusion = model;
        no_diffusion.diffusion = 0;
        const auto no_diffusion_terms = KARMA::element_transient_reaction_diffusion(
            system, element, no_diffusion, { 0, h, left, right, input } );
        bool diffusion_changed_residual = false, diffusion_changed_jacobian = false;
        for( int i = 0; i < 4; ++i )
        {
            KARMA::Real expected_residual = 0;
            for( int j = 0; j < 4; ++j )
            {
                const KARMA::Real pointwise_residual = ( right( j ) - left( j ) ) / h -
                    alpha * ( left( j ) + right( j ) ) / 2 - beta * input( j ) - source;
                expected_residual += mass( i, j ) * pointwise_residual;
                check_close( failures, no_diffusion_terms.jacobian( i, j ),
                             expected_jacobian_factor * mass( i, j ), "reaction Jacobian" );
            }
            check_close( failures, no_diffusion_terms.residual( i ), expected_residual, "reaction residual" );
            diffusion_changed_residual = diffusion_changed_residual ||
                std::abs( terms.residual( i ) - no_diffusion_terms.residual( i ) ) > 1.e-12;
            for( int j = 0; j < 4; ++j )
                diffusion_changed_jacobian = diffusion_changed_jacobian ||
                    std::abs( terms.jacobian( i, j ) - no_diffusion_terms.jacobian( i, j ) ) > 1.e-12;
        }
        if( !diffusion_changed_residual || !diffusion_changed_jacobian )
        {
            std::cerr << "nonzero diffusion did not change CN residual and Jacobian\n";
            ++failures;
        }

        const KARMA::Real epsilon = 1.e-7;
        KARMA::RealDenseVector perturbed_right = right;
        perturbed_right( 2 ) += epsilon;
        const auto perturbed = KARMA::element_transient_reaction_diffusion(
            system, element, model, { 0, h, left, perturbed_right, input } );
        for( int i = 0; i < 4; ++i )
        {
            const KARMA::Real finite_difference = ( perturbed.residual( i ) - terms.residual( i ) ) / epsilon;
            const KARMA::Real tolerance = 1.e-6 * std::max( KARMA::Real( 1 ), std::abs( terms.jacobian( i, 2 ) ) );
            if( !std::isfinite( finite_difference ) ||
                std::abs( finite_difference - terms.jacobian( i, 2 ) ) > tolerance )
            {
                std::cerr << "finite-difference Jacobian: expected " << terms.jacobian( i, 2 )
                          << ", got " << finite_difference << '\n';
                ++failures;
            }
        }
    }
    PetscFinalize();
    if( failures ) std::cerr << "transient assembly failures: " << failures << '\n';
    return failures ? 1 : 0;
}
