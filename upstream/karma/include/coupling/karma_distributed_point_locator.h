#ifndef KARMA_COUPLING_DISTRIBUTED_POINT_LOCATOR_H
#define KARMA_COUPLING_DISTRIBUTED_POINT_LOCATOR_H

#include "karma_moab_point_locator.h"

namespace KARMA
{
/**
 * Distributed locator using source-rank bounds and sparse MPI exchanges.
 * MOAB's CrystalRouter can replace the two Alltoallv phases later; the wire
 * record contract is deliberately independent of that transport choice.
 */
class KARMADistributedPointLocator final : public KARMAPointLocator
{
  public:
    KARMADistributedPointLocator(const KARMAMesh& source_mesh, PointLocationOptions options = {});

    void setup() override;
    [[nodiscard]] std::vector<LocatedPoint> locate(std::span<const Point> points) const override;

  private:
    const KARMAMesh& _mesh;
    PointLocationOptions _options;
    KARMAMoabPointLocator _local_locator;
    bool _ready = false;
};
}  // namespace KARMA

#endif
