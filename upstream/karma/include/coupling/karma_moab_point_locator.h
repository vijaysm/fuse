#ifndef KARMA_COUPLING_MOAB_POINT_LOCATOR_H
#define KARMA_COUPLING_MOAB_POINT_LOCATOR_H

#include "karma_point_locator.h"

#include "moab/AdaptiveKDTree.hpp"

#include <memory>

namespace KARMA
{
class KARMAMoabPointLocator final : public KARMAPointLocator
{
  public:
    KARMAMoabPointLocator(const KARMAMesh& source_mesh, PointLocationOptions options = {});
    ~KARMAMoabPointLocator() override;

    void setup() override;
    [[nodiscard]] std::vector< LocatedPoint > locate(std::span<const Point> points) const override;
    [[nodiscard]] bool contains(const Point& point, Real tolerance) const noexcept;

  private:
    const KARMAMesh& _mesh;
    PointLocationOptions _options;
    std::unique_ptr<moab::AdaptiveKDTree> _tree;
    moab::EntityHandle _root = 0;
    double _lower[3]{};
    double _upper[3]{};
    bool _ready = false;
};
}  // namespace KARMA

#endif
