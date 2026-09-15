#include "karma_distributed_point_locator.h"

#include "karma_exceptions.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace KARMA
{
namespace
{
struct WirePoint
{
    std::int32_t origin_rank = -1;
    std::int32_t local_index = -1;
    double coordinates[3]{};
};
static_assert(std::is_trivially_copyable_v<WirePoint>);

struct WireResult
{
    std::int32_t origin_rank = -1;
    std::int32_t local_index = -1;
    std::int32_t status = static_cast<std::int32_t>(PointLocationStatus::Uncovered);
    std::int32_t source_rank = -1;
    std::uint64_t source_global_id = 0;
    double reference[3]{};
    double residual = 0.0;
};
static_assert(std::is_trivially_copyable_v<WireResult>);

struct Bounds
{
    double lower[3]{};
    double upper[3]{};
};
static_assert(std::is_trivially_copyable_v<Bounds>);

WireResult encode(const WirePoint& point, const LocatedPoint& result, int source_rank)
{
    WireResult wire;
    wire.origin_rank = point.origin_rank;
    wire.local_index = point.local_index;
    wire.status = static_cast<std::int32_t>(result.status);
    wire.source_rank = result.status == PointLocationStatus::Located ? source_rank : -1;
    wire.source_global_id = result.source_element.global_id;
    for (unsigned dimension = 0; dimension < 3; ++dimension)
        wire.reference[dimension] = result.reference_point(static_cast<Eigen::Index>(dimension));
    wire.residual = result.residual;
    return wire;
}

LocatedPoint decode(const WireResult& wire, const Point& physical)
{
    LocatedPoint result;
    result.physical_point = physical;
    result.status = static_cast<PointLocationStatus>(wire.status);
    result.source_rank = wire.source_rank;
    result.source_element = {FieldEntryKey::Kind::MoabGlobalId, wire.source_global_id, 0, {}};
    for (unsigned dimension = 0; dimension < 3; ++dimension)
        result.reference_point(static_cast<Eigen::Index>(dimension)) = wire.reference[dimension];
    result.residual = wire.residual;
    return result;
}

bool in_bounds(const Point& point, const Bounds& bounds, Real tolerance)
{
    for (unsigned dimension = 0; dimension < 3; ++dimension)
    {
        const auto coordinate = point(static_cast<Eigen::Index>(dimension));
        if (coordinate < bounds.lower[dimension] - tolerance || coordinate > bounds.upper[dimension] + tolerance)
            return false;
    }
    return true;
}
}

KARMADistributedPointLocator::KARMADistributedPointLocator(const KARMAMesh& source_mesh,
                                                           PointLocationOptions options)
    : _mesh(source_mesh), _options(options), _local_locator(source_mesh, options)
{
}

void KARMADistributedPointLocator::setup()
{
    _local_locator.setup();
    _ready = true;
}

std::vector<LocatedPoint> KARMADistributedPointLocator::locate(std::span<const Point> points) const
{
    if (!_ready) KARMA_THROW(InvalidArgument("distributed point locator has not been setup"));
    const MPI_Comm communicator = _mesh.communicator();
    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(communicator, &rank);
    MPI_Comm_size(communicator, &ranks);
    if (ranks == 1) return _local_locator.locate(points);

    Bounds local_bounds;
    for (unsigned dimension = 0; dimension < 3; ++dimension)
    {
        local_bounds.lower[dimension] = std::numeric_limits<double>::infinity();
        local_bounds.upper[dimension] = -std::numeric_limits<double>::infinity();
    }
    if (_mesh.elements().empty())
    {
        for (unsigned dimension = 0; dimension < 3; ++dimension)
            local_bounds.lower[dimension] = local_bounds.upper[dimension] = 0.0;
    }
    else
    {
        for (const auto element : _mesh.elements())
        {
            std::array<double, 81> coordinates{};
            const auto count = _mesh.element_coordinates(element, coordinates);
            for (std::size_t node = 0; node < count; ++node)
                for (unsigned dimension = 0; dimension < 3; ++dimension)
                {
                    const double coordinate = coordinates[3 * node + dimension];
                    local_bounds.lower[dimension] = std::min(local_bounds.lower[dimension], coordinate);
                    local_bounds.upper[dimension] = std::max(local_bounds.upper[dimension], coordinate);
                }
        }
    }
    std::vector<Bounds> all_bounds(static_cast<std::size_t>(ranks));
    MPI_Allgather(&local_bounds, static_cast<int>(sizeof(Bounds)), MPI_BYTE, all_bounds.data(),
                  static_cast<int>(sizeof(Bounds)), MPI_BYTE, communicator);

    // Follow MOAB Coupler's sparse-routing pattern: exchange source bounds once,
    // send each point only to ranks whose source box may contain it, then return
    // only the originating rank's records.
    std::vector<std::vector<WirePoint>> outbound(static_cast<std::size_t>(ranks));
    for (std::size_t index = 0; index < points.size(); ++index)
    {
        WirePoint point;
        point.origin_rank = rank;
        point.local_index = static_cast<std::int32_t>(index);
        for (unsigned dimension = 0; dimension < 3; ++dimension)
            point.coordinates[dimension] = points[index](static_cast<Eigen::Index>(dimension));
        for (int destination = 0; destination < ranks; ++destination)
            if (in_bounds(points[index], all_bounds[static_cast<std::size_t>(destination)],
                          _options.residual_tolerance))
                outbound[static_cast<std::size_t>(destination)].push_back(point);
    }

    std::vector<int> send_counts(static_cast<std::size_t>(ranks));
    std::vector<int> send_displacements(static_cast<std::size_t>(ranks));
    std::vector<WirePoint> send_buffer;
    for (int destination = 0; destination < ranks; ++destination)
    {
        send_displacements[static_cast<std::size_t>(destination)] = static_cast<int>(send_buffer.size() * sizeof(WirePoint));
        send_counts[static_cast<std::size_t>(destination)] =
            static_cast<int>(outbound[static_cast<std::size_t>(destination)].size() * sizeof(WirePoint));
        send_buffer.insert(send_buffer.end(), outbound[static_cast<std::size_t>(destination)].begin(),
                           outbound[static_cast<std::size_t>(destination)].end());
    }
    std::vector<int> recv_counts(static_cast<std::size_t>(ranks));
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, communicator);
    std::vector<int> recv_displacements(static_cast<std::size_t>(ranks));
    int recv_bytes = 0;
    for (int source = 0; source < ranks; ++source)
    {
        recv_displacements[static_cast<std::size_t>(source)] = recv_bytes;
        recv_bytes += recv_counts[static_cast<std::size_t>(source)];
    }
    std::vector<std::byte> recv_buffer(static_cast<std::size_t>(recv_bytes));
    MPI_Alltoallv(send_buffer.data(), send_counts.data(), send_displacements.data(), MPI_BYTE, recv_buffer.data(),
                  recv_counts.data(), recv_displacements.data(), MPI_BYTE, communicator);

    const std::size_t received_points = recv_buffer.size() / sizeof(WirePoint);
    std::vector<WirePoint> received(received_points);
    std::memcpy(received.data(), recv_buffer.data(), recv_buffer.size());
    std::vector<Point> physical(received_points);
    for (std::size_t index = 0; index < received_points; ++index)
        for (unsigned dimension = 0; dimension < 3; ++dimension)
            physical[index](static_cast<Eigen::Index>(dimension)) = received[index].coordinates[dimension];
    const auto local_results = _local_locator.locate(physical);
    std::vector<std::vector<WireResult>> response(static_cast<std::size_t>(ranks));
    for (std::size_t index = 0; index < received.size(); ++index)
        response[static_cast<std::size_t>(received[index].origin_rank)].push_back(
            encode(received[index], local_results[index], rank));

    std::vector<WireResult> send_results;
    for (int destination = 0; destination < ranks; ++destination)
    {
        send_displacements[static_cast<std::size_t>(destination)] =
            static_cast<int>(send_results.size() * sizeof(WireResult));
        send_counts[static_cast<std::size_t>(destination)] =
            static_cast<int>(response[static_cast<std::size_t>(destination)].size() * sizeof(WireResult));
        send_results.insert(send_results.end(), response[static_cast<std::size_t>(destination)].begin(),
                            response[static_cast<std::size_t>(destination)].end());
    }
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, communicator);
    recv_bytes = 0;
    for (int source = 0; source < ranks; ++source)
    {
        recv_displacements[static_cast<std::size_t>(source)] = recv_bytes;
        recv_bytes += recv_counts[static_cast<std::size_t>(source)];
    }
    recv_buffer.assign(static_cast<std::size_t>(recv_bytes), std::byte{});
    MPI_Alltoallv(send_results.data(), send_counts.data(), send_displacements.data(), MPI_BYTE, recv_buffer.data(),
                  recv_counts.data(), recv_displacements.data(), MPI_BYTE, communicator);

    std::vector<LocatedPoint> results(points.size());
    std::vector<bool> seen(points.size(), false);
    const auto* records = reinterpret_cast<const WireResult*>(recv_buffer.data());
    for (std::size_t index = 0; index < recv_buffer.size() / sizeof(WireResult); ++index)
    {
        if (records[index].origin_rank != rank || records[index].local_index < 0 ||
            static_cast<std::size_t>(records[index].local_index) >= points.size())
            continue;
        const auto local_index = static_cast<std::size_t>(records[index].local_index);
        const auto candidate = decode(records[index], points[local_index]);
        if (!seen[local_index] || (candidate.status == PointLocationStatus::Located &&
                                   (results[local_index].status != PointLocationStatus::Located ||
                                    candidate.source_element.global_id < results[local_index].source_element.global_id)))
        {
            results[local_index] = candidate;
            seen[local_index] = true;
        }
    }
    return results;
}
}  // namespace KARMA
