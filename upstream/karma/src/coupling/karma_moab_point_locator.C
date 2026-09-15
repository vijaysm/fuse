#include "karma_moab_point_locator.h"

#include "karma_exceptions.h"

#include "moab/CN.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <optional>

namespace KARMA
{
namespace
{
FieldEntryKey element_key(const KARMAMesh& mesh, moab::EntityHandle element)
{
    return {FieldEntryKey::Kind::MoabGlobalId, static_cast<std::uint64_t>(mesh.global_id(element)), 0, {}};
}

bool in_box(const Point& point, const double* lower, const double* upper, Real tolerance)
{
    for (unsigned dimension = 0; dimension < 3; ++dimension)
        if (point(static_cast<Eigen::Index>(dimension)) < lower[dimension] - tolerance ||
            point(static_cast<Eigen::Index>(dimension)) > upper[dimension] + tolerance)
            return false;
    return true;
}
}

KARMAMoabPointLocator::KARMAMoabPointLocator(const KARMAMesh& source_mesh, PointLocationOptions options)
    : _mesh(source_mesh), _options(options)
{
    if (_options.reference_tolerance < 0.0 || _options.residual_tolerance < 0.0)
        KARMA_THROW(InvalidArgument("point locator tolerances must be nonnegative"));
}

KARMAMoabPointLocator::~KARMAMoabPointLocator() = default;

void KARMAMoabPointLocator::setup()
{
    // AdaptiveKDTree's API is non-const, but setup only builds/query-caches
    // search metadata; the mesh itself remains a non-owning const dependency.
    auto* moab = const_cast<moab::Interface*>(_mesh.moab());
    if (_mesh.elements().empty())
    {
        _ready = true;
        return;
    }
    _tree = std::make_unique<moab::AdaptiveKDTree>(moab);
    if (_tree->build_tree(_mesh.elements(), &_root) != moab::MB_SUCCESS)
        KARMA_THROW(InvalidArgument("failed to build MOAB AdaptiveKDTree"));
    unsigned int depth = 0;
    if (_tree->get_info(_root, _lower, _upper, depth) != moab::MB_SUCCESS)
        KARMA_THROW(InvalidArgument("failed to query MOAB AdaptiveKDTree bounds"));
    _ready = true;
}

bool KARMAMoabPointLocator::contains(const Point& point, Real tolerance) const noexcept
{
    return _ready && _tree && in_box(point, _lower, _upper, tolerance);
}

std::vector<LocatedPoint> KARMAMoabPointLocator::locate(std::span<const Point> points) const
{
    if (!_ready) KARMA_THROW(InvalidArgument("point locator has not been setup"));
    std::vector<LocatedPoint> results;
    if (!_tree)
    {
        results.resize(points.size());
        for (std::size_t index = 0; index < points.size(); ++index) results[index].physical_point = points[index];
        return results;
    }
    results.reserve(points.size());
    for (const Point& point : points)
    {
        LocatedPoint result;
        result.physical_point = point;
        if (!point.allFinite())
        {
            result.status = PointLocationStatus::BackendFailure;
            result.diagnostic = "point coordinates must be finite";
            results.push_back(std::move(result));
            continue;
        }
        if (!in_box(point, _lower, _upper, _options.residual_tolerance))
        {
            results.push_back(std::move(result));
            continue;
        }

        double xyz[3] = {point(0), point(1), point(2)};
        moab::EntityHandle leaf = 0;
        if (_tree->point_search(xyz, leaf, 1.0e-10, _options.reference_tolerance) != moab::MB_SUCCESS)
        {
            result.status = PointLocationStatus::BackendFailure;
            result.diagnostic = "MOAB AdaptiveKDTree point search failed";
            results.push_back(std::move(result));
            continue;
        }
        moab::Range candidates;
        if (leaf == 0 || _mesh.moab()->get_entities_by_handle(leaf, candidates) != moab::MB_SUCCESS)
        {
            results.push_back(std::move(result));
            continue;
        }

        std::optional<LocatedPoint> best;
        bool ambiguous = false;
        for (const auto candidate : candidates)
        {
            if (candidate == 0 || _mesh.moab()->type_from_handle(candidate) !=
                                      _mesh.moab()->type_from_handle(*_mesh.elements().begin()))
                continue;
            const KARMAGeometryMap geometry(_mesh, candidate);
            const auto inverse = geometry.try_inverse_map(point, _options.reference_tolerance,
                                                           _options.residual_tolerance);
            if (inverse.status == InverseMapStatus::InvalidGeometry)
            {
                result.status = PointLocationStatus::InvalidGeometry;
                result.diagnostic = "source candidate has invalid geometry";
                continue;
            }
            if (inverse.status != InverseMapStatus::Located) continue;
            LocatedPoint located;
            located.status = PointLocationStatus::Located;
            located.source_handle = candidate;
            located.source_element = element_key(_mesh, candidate);
            located.physical_point = point;
            located.reference_point = inverse.reference_point;
            located.residual = inverse.residual;
            if (best && best->source_element.global_id == located.source_element.global_id) continue;
            if (best) ambiguous = true;
            if (!best || located.source_element.global_id < best->source_element.global_id) best = located;
        }
        if (best && ambiguous && _options.trace_policy == BoundaryTracePolicy::RejectAmbiguous)
        {
            result.status = PointLocationStatus::BackendFailure;
            result.diagnostic = "point lies on an ambiguous source trace";
        }
        else if (best)
            result = *best;
        results.push_back(std::move(result));
    }
    return results;
}
}  // namespace KARMA
