#include "gray_scott_model.h"

#include <cmath>
#include <exception>
#include <iostream>
#include <limits>

namespace
{
int failures = 0;

void check( bool condition, const char* message )
{
    if( !condition )
    {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

template < class Function > bool throws_exception( Function&& function )
{
    try { function(); } catch( const std::exception& ) { return true; }
    return false;
}

void test_default_model()
{
    const KARMA::GrayScottModel model;
    const auto rates = model.reaction( 0.7, 0.2 );
    const double uvv = 0.7 * 0.2 * 0.2;
    check( std::abs( rates.u - ( -uvv + 0.060 * ( 1 - 0.7 ) ) ) < 1.e-14,
            "default U reaction follows Gray-Scott kinetics" );
    check( std::abs( rates.v - ( uvv - ( 0.060 + 0.062 ) * 0.2 ) ) < 1.e-14,
            "default V reaction follows Gray-Scott kinetics" );
    check( model.diffusion( KARMA::GrayScottSpecies::U ) == 0.16 &&
               model.diffusion( KARMA::GrayScottSpecies::V ) == 0.08,
           "default diffusion coefficients are exposed by species" );

    const auto jacobian = model.reaction_jacobian( 0.7, 0.2 );
    check( std::abs( jacobian[0][0] - ( -0.2 * 0.2 - 0.060 ) ) < 1.e-14 &&
               std::abs( jacobian[0][1] - ( -2 * 0.7 * 0.2 ) ) < 1.e-14,
           "U reaction derivatives are analytic" );
    check( std::abs( jacobian[1][0] - 0.2 * 0.2 ) < 1.e-14 &&
               std::abs( jacobian[1][1] - ( 2 * 0.7 * 0.2 - 0.060 - 0.062 ) ) < 1.e-14,
           "V reaction derivatives are analytic" );
}

void test_custom_parameters_and_validation()
{
    const KARMA::GrayScottParameters parameters{ 0.25, 0.125, 0.03, 0.07 };
    const KARMA::GrayScottModel model( parameters );
    const auto rates = model.reaction( 1.1, -0.4 );
    const double uvv = 1.1 * -0.4 * -0.4;
    check( std::abs( rates.u - ( -uvv + 0.03 * ( 1 - 1.1 ) ) ) < 1.e-14 &&
               std::abs( rates.v - ( uvv - 0.10 * -0.4 ) ) < 1.e-14,
           "custom parameters are used by both reactions" );
    check( throws_exception( [] { KARMA::GrayScottModel( { -1, 0.1, 0.03, 0.07 } ); } ),
           "negative diffusion is rejected" );
    check( throws_exception( [] {
               KARMA::GrayScottModel( { 0.1, 0.1, std::numeric_limits< Real >::quiet_NaN(), 0.07 } );
           } ),
           "nonfinite feed rate is rejected" );
}
}  // namespace

int main()
{
    test_default_model();
    test_custom_parameters_and_validation();
    return failures == 0 ? 0 : 1;
}
