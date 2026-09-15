#pragma once

/// CommGraph -- the rendezvous primitive that answers "which physical rank
/// holds, or wants, this data", for two components whose rank sets may be
/// disjoint, identical, or partially overlapping.
///
/// Spec: docs/specs/2026-09-11-comm-graph-design.md. Closes R5 for the
/// fully-disjoint case: `MpiTransport::build_plan()` already redistributes
/// correctly given `TransportSpec::src_ranks`/`dst_ranks`/`global_ids`, and
/// `Runner::Impl::attach_comm_graphs` (`src/runtime/runner.cpp`) is now the
/// consumer that populates those three from a `CommGraph`, at `setup()`, for
/// every disjoint-rank `Transport` instruction the lowering emits. Overlapping
/// rank sets are refused (`E-RT-TRANSPORT-OVERLAP`, R31) rather than silently
/// mishandled -- see the spec's status section for the complete list of what
/// ships and what is refused.
///
/// WHAT IT IS NOT. It is not mesh migration -- no geometry moves. It is not a
/// remap -- it computes no interpolation weights, and `Remap` still runs after
/// `Transport` on the target's ranks. It only decides who talks to whom.
///
/// NO MPI TYPE APPEARS HERE. Communicators cross this boundary as
/// `MPI_Comm_c2f()` handles, exactly as `MpiTransport::set_communicator` and
/// `MoabMeshBackend::set_communicator` already do, so `mpi.h` stays off a
/// consumer's include path.

#include <cstddef>
#include <vector>

#include "soma/base/result.hpp"
#include "soma/mesh/mesh_view.hpp"
#include "soma/transport/transport.hpp"

namespace soma {

#ifdef SOMA_HAVE_MPI

enum class MatchMode {
  /// Source and destination already share an identity space over the coupled
  /// region -- coincident meshes, e.g. `test/data/pair.yaml`'s
  /// `fluid_wall`/`solid_wall`. Matching is by `MeshView::global_id`.
  GlobalId,
  /// Source and destination do not share an identity space. The destination's
  /// centroids are matched to the nearest source centroid within
  /// `nn_tolerance`.
  NearestNeighbor,
};

/// One end of the rendezvous, as seen by the calling rank.
struct CommGraphSide {
  /// This rank's local view of the coupled region, or `nullptr` if this rank
  /// is not part of this component. Must be non-null exactly when
  /// `comm_handle` names a real communicator.
  const MeshView* mesh = nullptr;

  /// Topological dimension of the coupled entities (2 for a coupled surface
  /// of a volume mesh, and so on).
  int dim = 0;

  /// `MPI_Comm_c2f()` of THIS COMPONENT's own communicator, or
  /// `MPI_Comm_c2f(MPI_COMM_NULL)` if this rank is not in the component.
  ///
  /// WHY A COMMUNICATOR AND NOT A `RankSet`. `RankSet{first, count}`
  /// (`graph/component.hpp`) expresses a contiguous slice of one shared
  /// numbering, which cannot describe arbitrary partial overlap. Each side
  /// supplying its own communicator makes fully-overlapping, fully-disjoint
  /// and partially-overlapping rank sets uniform data rather than three
  /// special cases.
  int comm_handle = 0;
};

struct CommGraphOptions {
  MatchMode mode = MatchMode::GlobalId;

  /// Required (> 0) under `NearestNeighbor`; must be 0 under `GlobalId`.
  /// Checked before any collective work starts -- a precondition, not a
  /// runtime discovery -- so a bad value cannot desynchronise a job.
  double nn_tolerance = 0.0;
};

/// A matched pair whose source and destination live on the same rank.
///
/// LOAD BEARING, NOT AN OPTIMIZATION. `MpiTransport` refuses overlapping
/// source/destination rank sets by design (R24, and the note on
/// `MpiTransport` in transport.hpp): one `global_ids` list cannot describe a
/// rank that is in both. Splitting same-rank matches out here -- they never
/// reach `MpiTransport` -- is what makes a partially-overlapping component
/// pair deliverable at all.
struct LocalCopyPair {
  std::size_t source_local_index = 0;
  std::size_t dest_local_index = 0;
};

/// One end of a cross-rank exchange: everything this rank sends to, or
/// receives from, one peer.
///
/// Ordered by `(global_id, dest_local_index)` on BOTH sides of an exchange,
/// which is why a plain `MPI_Alltoallv` over these lists lines up: the sender
/// is told the receiver's local index during the directory phase precisely so
/// that it can sort on the same key. Without it, a destination that matched
/// two of its own entities to one source entity (legal, and normal, in
/// `NearestNeighbor` mode) would have no well-defined arrival order.
struct CommGraphPeer {
  /// Rank in the BASE communicator passed to `build()` -- not in the union
  /// communicator, which `build()` frees before it returns. See the note on
  /// `source_spec()`.
  int base_rank = -1;

  /// Local entity indices in this rank's own mesh view: source-side indices
  /// in `send_plan()`, destination-side indices in `recv_plan()`.
  std::vector<std::size_t> local_indices;

  /// The SOURCE entity's global id for each entry, parallel to
  /// `local_indices`. In `GlobalId` mode this is also the destination's id;
  /// in `NearestNeighbor` mode the destination has no such id, and this is
  /// the id of the source entity it matched.
  std::vector<std::size_t> global_ids;
};

/// What the directory phase actually cost this rank. Diagnostic only -- no
/// behaviour keys off it -- but it is the only way to observe the two claims
/// the design rests on, so the regression suite asserts on it.
struct CommGraphStats {
  /// Records this rank received in the FIRST routing round, i.e. the size of
  /// the candidate set it had to reconcile as a bucket owner.
  ///
  /// This is the number test 13 of the spec's matrix watches. A raw
  /// `hash(cell) % P` shard scrambles spatial locality and lets a refined
  /// patch pile onto a few ranks; the Morton partition with population-
  /// balanced splitters is chosen so that the maximum stays within a small
  /// factor of the mean, and asserting that is what would have caught the
  /// rejected design.
  std::size_t bucket_records = 0;

  /// `NearestNeighbor` mode only: this rank's destination entities, and the
  /// number of (entity, bucket) messages it took to probe their neighbourhoods.
  ///
  /// THE RATIO IS THE LOCALITY CLAIM, and it is the one measurement that
  /// actually separates the Morton partition from the raw
  /// `hash(cell) % P` the design rejected. A query must be probed against
  /// every bucket that could own a cell within `nn_tolerance`, which is its
  /// own cell and the `3^dim - 1` around it. Under a hash those land on up to
  /// `min(3^dim, P)` unrelated ranks; under a Morton partition they are
  /// adjacent on the curve and land on one or two. A population-balance
  /// assertion alone does NOT catch the hash -- a hash balances population
  /// perfectly well whenever there are many cells of comparable weight -- so
  /// the regression suite asserts on this ratio as well.
  std::size_t query_points = 0;
  std::size_t query_probes = 0;

  /// Rounds the last `CrystalRouter::route()` needed: at most `ceil(log2(P))`
  /// of the UNION communicator, not of the base one, and not O(P). A rank
  /// whose range halves into the smaller side finishes in fewer.
  int router_rounds = 0;
};

/// The result of a rendezvous. A pure value: `build()` creates a union
/// communicator, uses it, and frees it before returning, so nothing here owns
/// an MPI resource.
///
/// WHY THAT MATTERS. `MpiTransport`'s destructor is collective because it
/// frees a duplicated communicator, which makes asymmetric destruction across
/// ranks a hang (R24). A `CommGraph` has no such hazard: it can be copied,
/// stored, transposed and destroyed on one rank without any other rank
/// noticing.
class CommGraph {
 public:
  /// COLLECTIVE over `base_comm_handle`. Every rank of the base communicator
  /// must call this, in the same order, with the same `opts` -- including a
  /// rank in neither component, which gets an empty graph back, and a rank
  /// contributing zero local entities, which is ordinary in parallel and is
  /// the same rule `MpiTransport` already states.
  ///
  /// `base_comm_handle` is `MPI_Comm_c2f()` of a communicator whose group
  /// contains every rank of both components -- `MPI_COMM_WORLD` under spec §6
  /// mode 1, which is the only integration mode v1 implements. It is an
  /// explicit argument rather than an assumed `MPI_COMM_WORLD` because
  /// "assumed world" is not testable against a subset without spawning a job
  /// per case, and because a Fortran handle has no portable "unset" value to
  /// use as a sentinel.
  ///
  /// Errors:
  ///   * `E-COMMGRAPH-OPTS` -- `nn_tolerance <= 0` under `NearestNeighbor`,
  ///     or non-zero under `GlobalId`; a side with a communicator but no mesh
  ///     or a mesh but no communicator. Returned before any collective.
  ///   * `E-COMMGRAPH-COMM` -- the base communicator is null, a side's
  ///     communicator disagrees with the membership its ranks reported, or
  ///     `MPI_Group_union`/`MPI_Comm_create_group` failed.
  ///   * `E-COMMGRAPH-DUPID` -- one side's own list carries a global id
  ///     twice, or two source ranks claim to own the same id. Ambiguous
  ///     ownership is refused, never silently resolved.
  ///   * `E-COMMGRAPH-NOMATCH` -- a wanted id has no source (`GlobalId`), or
  ///     a query point has no source centroid within `nn_tolerance`
  ///     (`NearestNeighbor`, naming the point and the nearest distance
  ///     actually found).
  ///   * `E-CROUTER` -- the router underneath failed. Its own code
  ///     (`E-CROUTER-RANK`, `-SIZE`, `-MPI`, `-COMM`) leads the message, so
  ///     the specific failure is not lost, but the returned CODE is the
  ///     generic one: a routing failure detected on one rank is reported on
  ///     every rank, and only the rank that saw it has the specific code.
  ///
  /// Every one of them is agreed with an `MPI_Allreduce` before any rank acts
  /// on it, so a failure is an identical error return on every rank rather
  /// than one rank unwinding while its peers block in the next collective.
  [[nodiscard]] static Result<CommGraph> build(int base_comm_handle,
                                               const CommGraphSide& src,
                                               const CommGraphSide& dst,
                                               const CommGraphOptions& opts);

  /// The cross-rank subset this rank must SEND, in `TransportSpec` shape:
  /// `global_ids` are the source ids this rank owns and some other rank
  /// wants, `elements` is their count, and `src_ranks`/`dst_ranks` are the
  /// two rank sets of the exchange.
  ///
  /// RANKS ARE BASE-COMMUNICATOR RANKS. Hand `MpiTransport::set_communicator`
  /// the same base communicator that was passed to `build()`. The union
  /// communicator is an internal detail and no longer exists by the time this
  /// returns; a rank of it would be meaningless to the caller.
  ///
  /// GATHER/SCATTER. `global_ids` is a SUBSET of this rank's entities, so it
  /// does not index the field buffer. `source_indices()` gives the local
  /// index of each entry, in the same order, which is the gather map from the
  /// field buffer into the send buffer. `dest_indices()` is the matching
  /// scatter map on the other side.
  ///
  /// ONLY VALID WHEN `specs_are_disjoint()`. See that function.
  [[nodiscard]] TransportSpec source_spec() const;

  /// The cross-rank subset this rank must RECEIVE. `global_ids` are the
  /// source ids it wants; `dest_indices()` says where each lands.
  [[nodiscard]] TransportSpec dest_spec() const;

  /// Local index of each `source_spec().global_ids` entry, same order.
  [[nodiscard]] const std::vector<std::size_t>& source_indices() const noexcept;
  /// Local index of each `dest_spec().global_ids` entry, same order.
  [[nodiscard]] const std::vector<std::size_t>& dest_indices() const noexcept;

  /// Matches whose two ends are on this rank. Copy these directly; they never
  /// reach a `Transport` at all.
  [[nodiscard]] const std::vector<LocalCopyPair>& local_pairs() const noexcept;

  /// The complete cross-rank plan, peer by peer, ascending in `base_rank`.
  ///
  /// WHY THIS EXISTS ALONGSIDE `source_spec()`. `TransportSpec` carries one
  /// `global_ids` list per rank, so it cannot describe a rank that is both a
  /// source and a destination -- and under PARTIAL overlap a rank genuinely
  /// can be both, for different entities: rank 2 in both components may own
  /// an entity rank 3 wants (a send) while wanting an entity rank 0 owns (a
  /// receive). The same-rank shortcut removes only matches whose two ends are
  /// on ONE rank; it cannot remove that. `send_plan()`/`recv_plan()` express
  /// the answer in every overlap shape; `source_spec()`/`dest_spec()` express
  /// it in the shape `MpiTransport` consumes, which is available whenever
  /// `specs_are_disjoint()` is true.
  [[nodiscard]] const std::vector<CommGraphPeer>& send_plan() const noexcept;
  [[nodiscard]] const std::vector<CommGraphPeer>& recv_plan() const noexcept;

  /// True iff the two components' rank sets are disjoint, i.e. iff this graph
  /// can be executed by a single `MpiTransport` plan. Identical on every rank
  /// of the base communicator, including ranks in neither component: it is
  /// derived from the membership allgather, which every rank took part in.
  ///
  /// FALSE IS NOT AN ERROR, AND IT IS NOT RARE. Under full or partial overlap
  /// one `global_ids` list per rank cannot describe a rank that is a source
  /// for some entities and a destination for others, which is exactly why
  /// `MpiTransport::build_plan` refuses overlapping rank sets (R24). Under
  /// FULL overlap every match is a `local_pairs()` entry and there is nothing
  /// left to transport. Under PARTIAL overlap there is: use `send_plan()` and
  /// `recv_plan()`, which describe it exactly.
  [[nodiscard]] bool specs_are_disjoint() const noexcept;

  /// The reverse leg of a bidirectional channel: the same matched pairs with
  /// the two sides swapped. O(1) in communication -- no collective, no
  /// recomputation -- because the correspondence is symmetric by construction
  /// in both matching modes.
  ///
  /// A CAVEAT THAT IS REAL IN `NearestNeighbor` MODE. Nearest-neighbor
  /// matching may be many-to-one: two destination entities can match one
  /// source entity. Transposed, that is one entity carrying two global ids --
  /// or rather two entities carrying the same id -- on the new source side,
  /// which `MpiTransport::build_plan` refuses (`E-TRANSPORT-MATCH`, "pass
  /// owned entities only"), correctly: which of the two values should be
  /// transported is not defined. `local_pairs()`, `send_plan()` and
  /// `recv_plan()` remain exactly right in that case -- it is a scatter, and
  /// a scatter is well defined; only the `TransportSpec` shape cannot say it.
  [[nodiscard]] CommGraph transpose() const;

  /// True if this rank belongs to at least one of the two components. A rank
  /// in neither still calls `build()` (it is in the base communicator) and
  /// gets an empty, participating-in-nothing graph.
  [[nodiscard]] bool participates() const noexcept { return participates_; }

  /// Matched pairs whose two ends are on this rank plus those that cross.
  /// Diagnostic; the sum over ranks of `local_pairs().size() +
  /// recv_plan()` sizes is the number of destination entities matched.
  [[nodiscard]] std::size_t matched_count() const noexcept;

  [[nodiscard]] const CommGraphStats& stats() const noexcept { return stats_; }

  /// Union-communicator size and this rank's place in it. `-1`/`0` when
  /// `participates()` is false. Diagnostic.
  [[nodiscard]] int union_size() const noexcept { return union_size_; }
  [[nodiscard]] int union_rank() const noexcept { return union_rank_; }

 private:
  CommGraph() = default;

  std::vector<CommGraphPeer> send_;
  std::vector<CommGraphPeer> recv_;
  std::vector<LocalCopyPair> local_;

  // Flattened views of send_/recv_, in peer-ascending order, so that
  // source_spec()/source_indices() are O(1) to hand out.
  std::vector<std::size_t> send_ids_, send_idx_;
  std::vector<std::size_t> recv_ids_, recv_idx_;

  std::vector<int> src_ranks_;  // base ranks of the source component
  std::vector<int> dst_ranks_;  // base ranks of the destination component
  CommGraphStats   stats_;
  int              union_size_ = -1;
  int              union_rank_ = 0;
  bool             disjoint_ = true;
  bool             participates_ = false;
};

#endif  // SOMA_HAVE_MPI

}  // namespace soma
