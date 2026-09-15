#include "transient_reaction_diffusion_component.h"
#include "karma_mesh.h"

#include <petsc.h>
#include <cmath>
#include <iostream>

int main( int argc, char** argv )
{
    if( PetscInitialize( &argc, &argv, nullptr, nullptr ) ) return 1;
    int failures = 0;
    {
        KARMA::KARMAMesh mesh( 2, PETSC_COMM_WORLD );
        mesh.generate_box( 1, 1, 0 );
        KARMA::TransientReactionDiffusionModel model;
        model.diffusion = .25;
        model.reaction = []( const KARMA::Point&, Real state, Real, Real ) { return -state; };
        model.state_derivative = []( const KARMA::Point&, Real, Real, Real ) { return -1.; };
        KARMA::TransientReactionDiffusionComponent component( mesh, model );
        component.init();
        Vec input = component.system().create_vector();
        VecSet( input, 0 );
        component.bind_input( input );
        VecDestroy( &input );
        VecSet( component.solution(), 1 );
        component.checkpoint();
        component.advance( 0, .1 );
        PetscReal trial_norm = 0;
        VecNorm( component.trial_solution(), NORM_INFINITY, &trial_norm );
        if( !std::isfinite( trial_norm ) || trial_norm <= 0 ) ++failures;
        if( component.committed_time() != 0 || component.trial_time() != .1 )
        {
            std::cerr << "unexpected trial times: " << component.committed_time() << ' ' << component.trial_time() << '\n';
            ++failures;
        }
        if( !component.restore() )
        {
            std::cerr << "restore failed\n";
            ++failures;
        }
        if( component.committed_time() != 0 )
        {
            std::cerr << "restore committed time: " << component.committed_time() << '\n';
            ++failures;
        }
        component.checkpoint();
        component.advance( 0, .1 );
        component.commit();
        if( std::abs( component.committed_time() - .1 ) > 1.e-14 ) ++failures;
    }
    PetscFinalize();
    if( failures ) std::cerr << "transient component failures: " << failures << '\n';
    return failures ? 1 : 0;
}
