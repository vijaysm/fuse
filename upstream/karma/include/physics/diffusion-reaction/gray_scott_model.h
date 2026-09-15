#ifndef __gray_scott_model_h__
#define __gray_scott_model_h__

#include "karma_geometry.h"
#include "karma_types.h"

#include <array>
#include <functional>

namespace KARMA
{
enum class GrayScottSpecies
{
    U,
    V
};

struct GrayScottParameters
{
    Real diffusion_u{ 0.16 };
    Real diffusion_v{ 0.08 };
    Real feed_rate{ 0.060 };
    Real kill_rate{ 0.062 };
    std::function< Real( const Point&, Real ) > source_u;
    std::function< Real( const Point&, Real ) > source_v;
};

struct GrayScottReaction
{
    Real u{ 0 };
    Real v{ 0 };
};

class GrayScottModel
{
  public:
    using Jacobian = std::array< std::array< Real, 2 >, 2 >;

    explicit GrayScottModel( GrayScottParameters parameters = {} );

    [[nodiscard]] const GrayScottParameters& parameters() const noexcept { return _parameters; }
    [[nodiscard]] Real diffusion( GrayScottSpecies species ) const noexcept;
    [[nodiscard]] GrayScottReaction reaction( Real u, Real v ) const noexcept;
    [[nodiscard]] Jacobian reaction_jacobian( Real u, Real v ) const noexcept;

  private:
    GrayScottParameters _parameters;
};
}  // namespace KARMA

#endif
