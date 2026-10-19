// CommGraph. See include/fuse/transport/comm_graph.hpp for the contract and
// docs/specs/2026-09-11-comm-graph-design.md for the design.
//
// DEADLOCK DISCIPLINE. Identical to the one stated at the top of
// mpi_transport.cpp, applied consistently rather than reinvented:
//
//   1. Preconditions that need no data from other ranks (a bad
//      `nn_tolerance`, a mesh without a communicator) are checked and
//      returned BEFORE the first collective. Nothing is waiting yet.
//   2. Everything after that is agreed. Local validation sets a code; the
//      code is MPI_Allreduce(MAX)'d over every rank that could be waiting
//      before any rank returns. A rank that fails still reaches the
//      Allreduce, and every `CrystalRouter::route()` between here and there
//      is still called -- with fewer envelopes, never with none of the call.
//   3. The union communicator is created by every rank that is in it and
//      freed by every rank that is in it, on every path including the error
//      paths, before `build()` returns. `CommGraph` therefore owns no MPI
//      resource and its destructor is not collective -- which is exactly the
//      hazard R24 records for `MpiTransport`.
//   4. The two windows that cannot be agreed -- an MPI failure and a heap
//      failure strictly between two collectives -- abort loudly with a
//      message rather than hanging.
//
// WHY THE DIRECTORY IS ROUTED AND NOT ALLGATHERED. `MpiTransport::build_plan`
// allgathers every global id to every rank, O(total entities) per rank (R24).
// The directory phases below never do: each rank's traffic is bounded by its
// own local entity count plus whatever it receives as a bucket owner, and the
// router delivers it in O(log P) rounds. The one place an O(P)-ish exchange
// survives is the NearestNeighbor splitter selection, which operates on a
// per-rank histogram of OCCUPIED CELLS -- not points -- and is named as such
// where it happens.

#include "fuse/base/config.hpp"
#include "fuse/transport/comm_graph.hpp"

#include "fuse/transport/tags.hpp"

#if FUSE_HAVE_MPI

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <map>
#include <new>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "crystal_router.hpp"

namespace fuse {
namespace {

using detail::CrystalRouter;
using detail::Envelope;

using u64 = std::uint64_t;
using i32 = std::int32_t;

/// Codes the agreement Allreduce carries. Ordered so MPI_MAX picks a
/// deterministic winner when ranks fail differently.
enum class Err : int {
  None = 0,
  Comm = 1,
  DupId = 2,
  NoMatch = 3,
  Router = 4,
};

[[nodiscard]] const char* code_of(Err e) noexcept {
  switch (e) {
    case Err::None:    return "";
    case Err::Comm:    return "E-COMMGRAPH-COMM";
    case Err::DupId:   return "E-COMMGRAPH-DUPID";
    case Err::NoMatch: return "E-COMMGRAPH-NOMATCH";
    case Err::Router:  return "E-CROUTER";
  }
  return "E-COMMGRAPH-COMM";
}

[[noreturn]] void abort_job(MPI_Comm comm, const char* what) {
  // Between two collectives, where a throw would skip the next one and
  // hang every peer. No allocation, no formatting library, no locale.
  std::fprintf(stderr,  // NOLINT(fuse-raw-io)
               "FUSE: CommGraph cannot continue between two collectives (%s). "
               "Aborting rather than deadlocking the job.\n",
               what);
  std::fflush(stderr);
  MPI_Abort(comm, 1);
  std::abort();
}

// ---------------------------------------------------------------------------
// Wire records. Field order is chosen so that every struct is padding-free on
// any ABI where the fixed-width types have their natural sizes and alignments,
// which the static_asserts pin. Both ends of an exchange are the same binary
// in one MPI job (spec Section 6 mode 1), so the bytes are memcpy'd as they are.
// ---------------------------------------------------------------------------

/// Round 1, GlobalId mode. `kind` 0 = "I own this", 1 = "I want this".
struct IdRec {
  u64 gid = 0;
  u64 idx = 0;
  i32 rank = 0;
  i32 kind = 0;
};
static_assert(sizeof(IdRec) == 24, "IdRec must be padding-free");

/// One matched pair, in the currency both ends understand. The receiver
/// works out its own role by comparing `src_rank`/`dst_rank` against its own
/// union rank, so the bucket that produced it sends the SAME record to the
/// owner and to the requester -- or once, to one rank, when they coincide.
struct PairRec {
  u64 gid = 0;
  u64 src_idx = 0;
  u64 dst_idx = 0;
  i32 src_rank = 0;
  i32 dst_rank = 0;
};
static_assert(sizeof(PairRec) == 32, "PairRec must be padding-free");

/// Round 1, NearestNeighbor mode. `kind` 0 = source candidate, 1 = query.
struct PtRec {
  u64    morton = 0;
  u64    gid = 0;
  u64    idx = 0;
  double x = 0.0, y = 0.0, z = 0.0;
  i32    rank = 0;
  i32    kind = 0;
};
static_assert(sizeof(PtRec) == 56, "PtRec must be padding-free");

/// Round 2, NearestNeighbor mode: one bucket's outcome on one query. A query
/// is probed against several buckets, so the requester receives several of
/// these per entity and reduces them; `found == 0` still carries `dist` so a
/// rejection can name the nearest distance actually found anywhere.
struct ReplyRec {
  u64    dst_idx = 0;
  u64    gid = 0;
  u64    src_idx = 0;
  double dist = 0.0;
  i32    src_rank = 0;
  i32    found = 0;
};
static_assert(sizeof(ReplyRec) == 40, "ReplyRec must be padding-free");

/// Accumulates outgoing records per destination rank, so the router carries
/// one envelope per peer instead of one per record.
class Outbox {
 public:
  explicit Outbox(int n) : buf_(static_cast<std::size_t>(n)) {}

  template <class T>
  void put(int rank, const T& rec) {
    std::vector<std::uint8_t>& b = buf_[static_cast<std::size_t>(rank)];
    const std::size_t at = b.size();
    b.resize(at + sizeof(T));
    std::memcpy(b.data() + at, &rec, sizeof(T));
  }

  [[nodiscard]] std::vector<Envelope> take() {
    std::vector<Envelope> out;
    for (std::size_t r = 0; r < buf_.size(); ++r) {
      if (buf_[r].empty()) continue;
      Envelope e;
      e.dest_rank = static_cast<int>(r);
      e.payload = std::move(buf_[r]);
      out.push_back(std::move(e));
    }
    buf_.clear();
    return out;
  }

 private:
  std::vector<std::vector<std::uint8_t>> buf_;
};

/// Walks every record of type `T` in a routed batch.
template <class T, class F>
void for_each_rec(const std::vector<Envelope>& in, F&& f) {
  for (const Envelope& e : in) {
    const std::size_t n = e.payload.size() / sizeof(T);
    for (std::size_t i = 0; i < n; ++i) {
      T rec;
      std::memcpy(&rec, e.payload.data() + i * sizeof(T), sizeof(T));
      f(rec);
    }
  }
}

/// splitmix64. A hash, not a mask: `gid % P` alone buckets a contiguous id
/// block onto a handful of ranks, which is the load imbalance this shard
/// function exists to avoid. (NearestNeighbor mode does NOT use a hash --
/// see the Morton partition below and Section 4.4 of the spec for why.)
[[nodiscard]] u64 mix64(u64 x) noexcept {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

// ---------------------------------------------------------------------------
// Morton (Z-order) codes
// ---------------------------------------------------------------------------

/// Bits per axis. Three axes at 21 bits fit a 63-bit code, so the grid is at
/// most 2^21 cells on a side; `plan_grid` below coarsens the cell size rather
/// than overflowing when a tolerance is tiny against the domain.
constexpr int    kMortonBits = 21;
constexpr u64    kMaxPerAxis = u64{1} << kMortonBits;

/// Spread the low 21 bits of `v` into every third bit.
[[nodiscard]] u64 spread3(u64 v) noexcept {
  v &= 0x1FFFFFULL;
  v = (v | (v << 32)) & 0x1F00000000FFFFULL;
  v = (v | (v << 16)) & 0x1F0000FF0000FFULL;
  v = (v | (v << 8))  & 0x100F00F00F00F00FULL;
  v = (v | (v << 4))  & 0x10C30C30C30C30C3ULL;
  v = (v | (v << 2))  & 0x1249249249249249ULL;
  return v;
}

[[nodiscard]] u64 morton3(u64 ix, u64 iy, u64 iz) noexcept {
  return spread3(ix) | (spread3(iy) << 1) | (spread3(iz) << 2);
}

/// The uniform cell grid every point is binned into.
struct Grid {
  double lo[3] = {0.0, 0.0, 0.0};
  double cell = 1.0;
  u64    n[3] = {1, 1, 1};

  void bin(double x, double y, double z, u64 c[3]) const noexcept {
    const double p[3] = {x, y, z};
    for (int k = 0; k < 3; ++k) {
      double t = (p[k] - lo[k]) / cell;
      if (!(t >= 0.0)) t = 0.0;  // also catches NaN
      auto  i = static_cast<u64>(t);
      if (i >= n[k]) i = n[k] - 1;
      c[k] = i;
    }
  }
};

/// Choose the cell size and the per-axis resolution.
///
/// `2 * tolerance` is the default the spec gives, and it is the value that
/// makes the `3^dim` neighbourhood probe SUFFICIENT rather than merely
/// plausible: a source within `tolerance` of a query differs from it by at
/// most `tolerance <= cell/2` on each axis, so it cannot be more than one
/// cell away on any axis. Coarsening the cell (which is what the clamp below
/// does) preserves that inequality; refining it would not.
[[nodiscard]] Grid plan_grid(const double lo[3], const double hi[3],
                             double tolerance) noexcept {
  Grid g;
  double cell = 2.0 * tolerance;
  for (int k = 0; k < 3; ++k) {
    g.lo[k] = lo[k] <= hi[k] ? lo[k] : 0.0;
  }
  // One pass to find the coarsening factor, then a second to apply it, so
  // every axis shares one cell size and the code stays a cube lattice.
  for (int k = 0; k < 3; ++k) {
    const double extent = (lo[k] <= hi[k]) ? (hi[k] - lo[k]) : 0.0;
    const double want = extent / cell + 1.0;
    if (want > static_cast<double>(kMaxPerAxis)) {
      cell = extent / static_cast<double>(kMaxPerAxis - 1);
    }
  }
  if (!(cell > 0.0)) cell = 1.0;
  g.cell = cell;
  for (int k = 0; k < 3; ++k) {
    const double extent = (lo[k] <= hi[k]) ? (hi[k] - lo[k]) : 0.0;
    const double want = std::floor(extent / cell) + 1.0;
    u64 nk = want >= static_cast<double>(kMaxPerAxis)
                 ? kMaxPerAxis
                 : static_cast<u64>(want);
    if (nk < 1) nk = 1;
    g.n[k] = nk;
  }
  return g;
}

// ---------------------------------------------------------------------------
// One side's local contribution
// ---------------------------------------------------------------------------

struct LocalEntities {
  std::vector<u64>         gid;
  std::vector<std::size_t> idx;
  std::vector<double>      xyz;  // 3 per entry, NearestNeighbor mode only
};

/// The source side contributes OWNED entities only; the destination side
/// contributes every locally visible entity.
///
/// WHY THEY DIFFER. Two source ranks advertising one id is ambiguous
/// ownership, and `MpiTransport::build_plan` rejects it outright ("pass owned
/// entities only, or the value transported would depend on which copy won").
/// A destination's ghost slots, by contrast, are part of the buffer that has
/// to be filled -- `MeshView::entity_count` is documented as "the size a
/// per-entity field buffer over the whole mesh must have" -- and several
/// destinations wanting one id is a multicast, which the transport handles.
[[nodiscard]] bool gather_side(const MeshView& mv, int dim, bool owned_only,
                               bool need_coords, LocalEntities* out,
                               std::string* why) {
  const std::size_t n = mv.entity_count(dim);
  out->gid.reserve(n);
  out->idx.reserve(n);
  if (need_coords) out->xyz.reserve(3 * n);
  for (std::size_t i = 0; i < n; ++i) {
    if (owned_only && !mv.is_owned(dim, i)) continue;
    out->gid.push_back(static_cast<u64>(mv.global_id(dim, i)));
    out->idx.push_back(i);
    if (need_coords) {
      const Point p = mv.centroid(dim, i);
      out->xyz.push_back(p.x);
      out->xyz.push_back(p.y);
      out->xyz.push_back(p.z);
    }
  }
  // A duplicate inside ONE rank's own list is a configuration error either
  // way round: as a source it is ambiguous ownership, as a destination it is
  // two buffer slots claiming one identity.
  std::unordered_map<u64, std::size_t> seen;
  seen.reserve(out->gid.size());
  for (std::size_t k = 0; k < out->gid.size(); ++k) {
    const auto [it, inserted] = seen.emplace(out->gid[k], out->idx[k]);
    if (!inserted) {
      *why = "global id " + std::to_string(out->gid[k]) +
             " appears at local index " + std::to_string(it->second) +
             " and again at " + std::to_string(out->idx[k]) +
             " of one side's own entity list";
      return false;
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// Accumulating the matched pairs into the shape CommGraph hands out
// ---------------------------------------------------------------------------

struct Entry {
  u64         gid;
  u64         key_idx;    // the DESTINATION's local index: the order both ends sort on
  std::size_t own_idx;    // this rank's own local index
};

struct Accum {
  std::vector<LocalCopyPair>        local;
  std::map<int, std::vector<Entry>> send;  // keyed by peer UNION rank
  std::map<int, std::vector<Entry>> recv;
};

void absorb(Accum* a, const PairRec& r, int me) {
  if (r.src_rank == me && r.dst_rank == me) {
    a->local.push_back(LocalCopyPair{static_cast<std::size_t>(r.src_idx),
                                     static_cast<std::size_t>(r.dst_idx)});
    return;
  }
  if (r.src_rank == me) {
    a->send[r.dst_rank].push_back(
        Entry{r.gid, r.dst_idx, static_cast<std::size_t>(r.src_idx)});
    return;
  }
  if (r.dst_rank == me) {
    a->recv[r.src_rank].push_back(
        Entry{r.gid, r.dst_idx, static_cast<std::size_t>(r.dst_idx)});
  }
  // Neither: not addressed to this rank. Cannot happen -- the router
  // delivers only what names us -- and is ignored rather than asserted so a
  // stray record can never become a one-rank abort inside a collective.
}

}  // namespace

// ---------------------------------------------------------------------------
// CommGraph
// ---------------------------------------------------------------------------

const std::vector<std::size_t>& CommGraph::source_indices() const noexcept {
  return send_idx_;
}
const std::vector<std::size_t>& CommGraph::dest_indices() const noexcept {
  return recv_idx_;
}
const std::vector<LocalCopyPair>& CommGraph::local_pairs() const noexcept {
  return local_;
}
const std::vector<CommGraphPeer>& CommGraph::send_plan() const noexcept {
  return send_;
}
const std::vector<CommGraphPeer>& CommGraph::recv_plan() const noexcept {
  return recv_;
}
bool CommGraph::specs_are_disjoint() const noexcept { return disjoint_; }

std::size_t CommGraph::matched_count() const noexcept {
  return local_.size() + recv_idx_.size();
}

TransportSpec CommGraph::source_spec() const {
  TransportSpec s;
  s.src_rank_set = 0;
  s.dst_rank_set = 1;
  s.src_ranks = src_ranks_;
  s.dst_ranks = dst_ranks_;
  s.global_ids = send_ids_;
  s.elements = send_ids_.size();
  s.components = 1;
  return s;
}

TransportSpec CommGraph::dest_spec() const {
  TransportSpec s;
  s.src_rank_set = 0;
  s.dst_rank_set = 1;
  s.src_ranks = src_ranks_;
  s.dst_ranks = dst_ranks_;
  s.global_ids = recv_ids_;
  s.elements = recv_ids_.size();
  s.components = 1;
  return s;
}

CommGraph CommGraph::transpose() const {
  CommGraph t;
  t.send_ = recv_;
  t.recv_ = send_;
  t.send_ids_ = recv_ids_;
  t.send_idx_ = recv_idx_;
  t.recv_ids_ = send_ids_;
  t.recv_idx_ = send_idx_;
  t.src_ranks_ = dst_ranks_;
  t.dst_ranks_ = src_ranks_;
  t.disjoint_ = disjoint_;
  t.participates_ = participates_;
  t.stats_ = stats_;
  t.union_size_ = union_size_;
  t.union_rank_ = union_rank_;
  t.local_.reserve(local_.size());
  for (const LocalCopyPair& p : local_) {
    t.local_.push_back(LocalCopyPair{p.dest_local_index, p.source_local_index});
  }
  return t;
}

Result<CommGraph> CommGraph::build(int base_comm_handle,
                                   const CommGraphSide& src,
                                   const CommGraphSide& dst,
                                   const CommGraphOptions& opts) {
  // -- Preconditions, before any collective -------------------------------
  // Nothing on any other rank is waiting yet, so returning here cannot
  // desynchronise a job. These are the checks spec Section 6 requires to be
  // preconditions rather than runtime discoveries.
  const bool nn = opts.mode == MatchMode::NearestNeighbor;
  if (nn && !(opts.nn_tolerance > 0.0)) {
    return Failure{"E-COMMGRAPH-OPTS",
                 "NearestNeighbor matching needs a positive nn_tolerance; got " +
                     std::to_string(opts.nn_tolerance) +
                     ". There is no defensible default: the right value is a "
                     "property of the two discretizations being coupled"};
  }
  if (!nn && opts.nn_tolerance != 0.0) {
    return Failure{"E-COMMGRAPH-OPTS",
                 "nn_tolerance is " + std::to_string(opts.nn_tolerance) +
                     " under GlobalId matching, where it is not consulted. "
                     "Refused rather than ignored, because a caller who set "
                     "it meant to select NearestNeighbor"};
  }

  MPI_Comm base = MPI_Comm_f2c(base_comm_handle);
  if (base == MPI_COMM_NULL) {
    return Failure{"E-COMMGRAPH-COMM",
                 "build() needs a base communicator whose group contains both "
                 "components; got MPI_COMM_NULL"};
  }

  const MPI_Comm src_comm = MPI_Comm_f2c(src.comm_handle);
  const MPI_Comm dst_comm = MPI_Comm_f2c(dst.comm_handle);
  const bool     in_src = src_comm != MPI_COMM_NULL;
  const bool     in_dst = dst_comm != MPI_COMM_NULL;

  if (in_src != (src.mesh != nullptr)) {
    return Failure{"E-COMMGRAPH-OPTS",
                 "the source side has a communicator but no mesh, or a mesh "
                 "but no communicator; a rank is in a component exactly when "
                 "it can see that component's local view"};
  }
  if (in_dst != (dst.mesh != nullptr)) {
    return Failure{"E-COMMGRAPH-OPTS",
                 "the destination side has a communicator but no mesh, or a "
                 "mesh but no communicator"};
  }

  int base_rank = 0;
  int base_size = 1;
  MPI_Comm_rank(base, &base_rank);
  MPI_Comm_size(base, &base_size);
  const auto ubase = static_cast<std::size_t>(base_size);

  // -- Collective 1: who is in which component ----------------------------
  // O(P) chars at setup, once. It is what replaces MPI_Group_union: the
  // membership IS the group, and deriving it here means a rank in only one
  // component never has to hold the other's communicator (it cannot -- there
  // is no MPI_Comm to call MPI_Comm_group on).
  std::vector<char> flags(ubase, 0);
  {
    char mine = static_cast<char>((in_src ? 1 : 0) | (in_dst ? 2 : 0));
    const int rc =
        MPI_Allgather(&mine, 1, MPI_CHAR, flags.data(), 1, MPI_CHAR, base);
    if (rc != MPI_SUCCESS) {
      return Failure{"E-COMMGRAPH-COMM",
                   "MPI_Allgather of the component membership failed"};
    }
  }

  std::vector<int> union_ranks;
  int              n_src = 0;
  int              n_dst = 0;
  for (std::size_t r = 0; r < ubase; ++r) {
    if ((flags[r] & 1) != 0) ++n_src;
    if ((flags[r] & 2) != 0) ++n_dst;
    if (flags[r] != 0) union_ranks.push_back(static_cast<int>(r));
  }

  // -- Collective 2: agree that the supplied communicators say the same ---
  // A caller who hands a communicator built over a different set of ranks
  // than the ones reporting membership has a bug that would otherwise show
  // up as a silently wrong rank set. One MPI_Comm_size per side catches it.
  Err         err = Err::None;
  std::string detail;
  {
    if (in_src) {
      int s = 0;
      MPI_Comm_size(src_comm, &s);
      if (s != n_src) {
        err = Err::Comm;
        detail = "the source communicator holds " + std::to_string(s) +
                 " ranks but " + std::to_string(n_src) +
                 " ranks of the base communicator report being in it";
      }
    }
    if (err == Err::None && in_dst) {
      int s = 0;
      MPI_Comm_size(dst_comm, &s);
      if (s != n_dst) {
        err = Err::Comm;
        detail = "the destination communicator holds " + std::to_string(s) +
                 " ranks but " + std::to_string(n_dst) +
                 " ranks of the base communicator report being in it";
      }
    }
    auto      e = static_cast<int>(err);
    int       agreed = e;
    const int rc = MPI_Allreduce(&e, &agreed, 1, MPI_INT, MPI_MAX, base);
    if (rc != MPI_SUCCESS) {
      return Failure{"E-COMMGRAPH-COMM",
                   "MPI_Allreduce failed while agreeing the membership"};
    }
    if (agreed != 0) {
      if (err != Err::None) return Failure{code_of(err), detail};
      return Failure{code_of(static_cast<Err>(agreed)),
                   "another rank rejected the component membership; this "
                   "rank's own communicators were consistent"};
    }
  }

  // -- The two rank sets, in base-communicator numbering -------------------
  //
  // THEY ARE THE COMPONENTS' MEMBERSHIPS, not "the ranks that turned out to
  // have cross traffic", and that is deliberate. `MpiTransport::build_plan`
  // is collective over the communicator it was handed -- the base one -- and
  // needs the SAME `src_ranks`/`dst_ranks` on every rank of it, including
  // ranks in neither component and ranks whose own cross-traffic list came
  // out empty. Both are ordinary for it: "a rank in neither rank set still
  // enters MPI_Alltoallv with zero counts", and a rank that is in a set but
  // holds no elements is the zero-element case its own test suite already
  // pins. Deriving the sets from the membership makes them available here,
  // identically, on EVERY base rank -- with no collective beyond the
  // membership allgather that just ran, and with no rank able to disagree.
  std::vector<int> all_src, all_dst;
  bool             disjoint = true;
  for (std::size_t r = 0; r < ubase; ++r) {
    if ((flags[r] & 1) != 0) all_src.push_back(static_cast<int>(r));
    if ((flags[r] & 2) != 0) all_dst.push_back(static_cast<int>(r));
    if ((flags[r] & 3) == 3) disjoint = false;
  }

  // A rank in neither component is done. It took part in the two collectives
  // above -- it is in the base communicator, so it had to -- and it is not in
  // the union group, so it must NOT call MPI_Comm_create_group below. It
  // still leaves holding the rank sets, because it still has to enter
  // `MpiTransport`'s collectives alongside everyone else.
  if (!in_src && !in_dst) {
    CommGraph empty;
    empty.participates_ = false;
    empty.src_ranks_ = std::move(all_src);
    empty.dst_ranks_ = std::move(all_dst);
    empty.disjoint_ = disjoint;
    return empty;
  }

  // -- The union communicator ---------------------------------------------
  MPI_Comm  ucomm = MPI_COMM_NULL;
  MPI_Group bgroup = MPI_GROUP_NULL;
  MPI_Group ugroup = MPI_GROUP_NULL;
  {
    // Review 2026-10-11 TR-9. MPI_Comm_group/MPI_Group_incl are both purely
    // local -- no communication -- but MPI_Comm_create_group right after
    // them is collective over `ugroup`, and every rank that reaches it did
    // so because its OWN local pair above succeeded. A local failure here
    // on one rank only used to return a Result while its ugroup peers, who
    // saw no failure, go on to enter that collective and wait for a rank
    // that now never will -- exactly the "MPI failure strictly between two
    // collectives" window this file's own header discipline (point 4)
    // already names and answers by aborting rather than returning.
    if (MPI_Comm_group(base, &bgroup) != MPI_SUCCESS ||
        MPI_Group_incl(bgroup, static_cast<int>(union_ranks.size()),
                       union_ranks.data(), &ugroup) != MPI_SUCCESS) {
      if (bgroup != MPI_GROUP_NULL) MPI_Group_free(&bgroup);
      abort_job(base, "could not form the union of the two components' groups");
    }
    // Collective over `ugroup` only, which is exactly the set of ranks that
    // reach this line.
    const int rc = MPI_Comm_create_group(base, ugroup, tags::kUnionCommCreate, &ucomm);
    MPI_Group_free(&ugroup);
    MPI_Group_free(&bgroup);
    if (rc != MPI_SUCCESS || ucomm == MPI_COMM_NULL) {
      return Failure{"E-COMMGRAPH-COMM",
                   "MPI_Comm_create_group could not materialise the union "
                   "communicator; the base communicator must contain every "
                   "rank of both components"};
    }
    // Errors become a Result rather than an abort inside the caller's
    // communicator, exactly as MpiTransport does on its own duplicate.
    MPI_Comm_set_errhandler(ucomm, MPI_ERRORS_RETURN);
  }

  const int usize = static_cast<int>(union_ranks.size());
  int       urank = 0;
  for (int i = 0; i < usize; ++i) {
    if (union_ranks[static_cast<std::size_t>(i)] == base_rank) urank = i;
  }

  // From here on every return goes through `finish`, which frees the union
  // communicator on every rank that created it.
  const auto finish = [&](Result<CommGraph> r) -> Result<CommGraph> {
    MPI_Comm_free(&ucomm);
    return r;
  };

  // -- Local entity lists --------------------------------------------------
  LocalEntities s_loc, d_loc;
  {
    std::string why;
    try {
      if (in_src && !gather_side(*src.mesh, src.dim, /*owned_only=*/true, nn,
                                 &s_loc, &why)) {
        err = Err::DupId;
        detail = "source side: " + why;
      }
      if (err == Err::None && in_dst &&
          !gather_side(*dst.mesh, dst.dim, /*owned_only=*/false, nn, &d_loc,
                       &why)) {
        err = Err::DupId;
        detail = "destination side: " + why;
      }
    } catch (const std::exception& e) {
      err = Err::Comm;
      detail = std::string{"reading the local mesh view failed: "} + e.what();
    } catch (...) {
      err = Err::Comm;
      detail = "reading the local mesh view failed";
    }
  }

  // -- The directory -------------------------------------------------------
  Accum       acc;
  std::size_t bucket_records = 0;
  std::size_t query_points = 0;
  std::size_t query_probes = 0;
  CrystalRouter router(ucomm);

  const auto route = [&](std::vector<Envelope> out,
                         std::vector<Envelope>* got) {
    Result<std::vector<Envelope>> r = router.route(std::move(out));
    if (!r.ok()) {
      if (err == Err::None) {
        err = Err::Router;
        detail = r.error().code + ": " + r.error().message;
      }
      got->clear();
      return;
    }
    *got = std::move(r.value());
  };

  // Every allocation from here to the agreement Allreduce sits BETWEEN two
  // collectives, so a heap failure on one rank cannot be reported: its peers
  // are already committed to the next routing round. Same window, same
  // answer, as mpi_transport.cpp -- abort with a message rather than hang.
  try {
    if (opts.mode == MatchMode::GlobalId) {
      // Round 1: every id, owned or wanted, to its bucket.
      std::vector<Envelope> got1;
      {
        Outbox out(usize);
        if (err == Err::None) {
          for (std::size_t k = 0; k < s_loc.gid.size(); ++k) {
            const int b = static_cast<int>(mix64(s_loc.gid[k]) %
                                           static_cast<u64>(usize));
            out.put(b, IdRec{s_loc.gid[k], s_loc.idx[k], urank, 0});
          }
          for (std::size_t k = 0; k < d_loc.gid.size(); ++k) {
            const int b = static_cast<int>(mix64(d_loc.gid[k]) %
                                           static_cast<u64>(usize));
            out.put(b, IdRec{d_loc.gid[k], d_loc.idx[k], urank, 1});
          }
        }
        route(out.take(), &got1);
      }

      // Reconcile, on the bucket rank, and route the outcome home.
      std::vector<Envelope> got2;
      {
        Outbox out(usize);
        std::unordered_map<u64, IdRec> owner;
        std::vector<IdRec>             wants;
        for (const Envelope& e : got1) {
          bucket_records += e.payload.size() / sizeof(IdRec);
        }
        for_each_rec<IdRec>(got1, [&](const IdRec& r) {
          if (r.kind == 0) {
            const auto [it, inserted] = owner.emplace(r.gid, r);
            if (!inserted && err == Err::None) {
              err = Err::DupId;
              detail = "global id " + std::to_string(r.gid) +
                       " is owned by two different source ranks (union ranks " +
                       std::to_string(it->second.rank) + " and " +
                       std::to_string(r.rank) +
                       "); pass owned entities only, or the value transported "
                       "would depend on which copy won";
            }
          } else {
            wants.push_back(r);
          }
        });
        for (const IdRec& w : wants) {
          const auto it = owner.find(w.gid);
          if (it == owner.end()) {
            if (err == Err::None) {
              err = Err::NoMatch;
              detail = "destination union rank " + std::to_string(w.rank) +
                       " wants global id " + std::to_string(w.gid) +
                       " (its local entity " + std::to_string(w.idx) +
                       ") but no source rank owns it";
            }
            continue;
          }
          const PairRec p{w.gid, it->second.idx, w.idx, it->second.rank, w.rank};
          out.put(p.dst_rank, p);
          if (p.src_rank != p.dst_rank) out.put(p.src_rank, p);
        }
        route(out.take(), &got2);
      }

      for_each_rec<PairRec>(got2, [&](const PairRec& r) { absorb(&acc, r, urank); });
    } else {
      // ---------------- NearestNeighbor -----------------------------------
      // Step 1: the union's bounding box. O(1) data -- six doubles -- not O(N).
      double lo[3] = {std::numeric_limits<double>::infinity(),
                      std::numeric_limits<double>::infinity(),
                      std::numeric_limits<double>::infinity()};
      double hi[3] = {-std::numeric_limits<double>::infinity(),
                      -std::numeric_limits<double>::infinity(),
                      -std::numeric_limits<double>::infinity()};
      for (const std::vector<double>* v : {&s_loc.xyz, &d_loc.xyz}) {
        for (std::size_t k = 0; k + 3 <= v->size(); k += 3) {
          for (std::size_t a = 0; a < 3; ++a) {
            lo[a] = std::min(lo[a], (*v)[k + a]);
            hi[a] = std::max(hi[a], (*v)[k + a]);
          }
        }
      }
      {
        double reduced[3] = {0.0, 0.0, 0.0};
        if (MPI_Allreduce(lo, reduced, 3, MPI_DOUBLE, MPI_MIN, ucomm) !=
            MPI_SUCCESS) {
          if (err == Err::None) {
            err = Err::Comm;
            detail = "MPI_Allreduce of the bounding box minimum failed";
          }
        } else {
          std::memcpy(lo, reduced, sizeof lo);
        }
        if (MPI_Allreduce(hi, reduced, 3, MPI_DOUBLE, MPI_MAX, ucomm) !=
            MPI_SUCCESS) {
          if (err == Err::None) {
            err = Err::Comm;
            detail = "MPI_Allreduce of the bounding box maximum failed";
          }
        } else {
          std::memcpy(hi, reduced, sizeof hi);
        }
      }

      const Grid grid = plan_grid(lo, hi, opts.nn_tolerance);

      const auto morton_at = [&](double x, double y, double z) {
        u64 c[3];
        grid.bin(x, y, z, c);
        return morton3(c[0], c[1], c[2]);
      };

      // Step 2: the local histogram -- one entry per OCCUPIED CELL, not per
      // point, which is what keeps the splitter exchange below cheap on a mesh
      // with many points in few cells.
      std::map<u64, u64> hist;
      for (std::size_t k = 0; k < s_loc.gid.size(); ++k) {
        ++hist[morton_at(s_loc.xyz[k * 3], s_loc.xyz[k * 3 + 1],
                         s_loc.xyz[k * 3 + 2])];
      }
      for (std::size_t k = 0; k < d_loc.gid.size(); ++k) {
        ++hist[morton_at(d_loc.xyz[k * 3], d_loc.xyz[k * 3 + 1],
                         d_loc.xyz[k * 3 + 2])];
      }

      // Step 3: splitters that balance POPULATION, not cell count.
      //
      // THIS IS THE ONE O(occupied cells) EXCHANGE IN THE FILE, and it is here
      // deliberately: the summary is orders of magnitude smaller than the point
      // set, every rank derives the identical splitter array from identical
      // input (so no rank can disagree about where a point routes), and it runs
      // once at setup. A sampling scheme would cut it further and is the
      // obvious next step if a profile ever says so.
      std::vector<u64> splitters;
      {
        std::vector<u64> mine;
        mine.reserve(hist.size() * 2);
        for (const auto& [m, c] : hist) {
          mine.push_back(m);
          mine.push_back(c);
        }
        std::vector<int> counts(static_cast<std::size_t>(usize), 0);
        auto             my_n = static_cast<int>(mine.size());
        if (MPI_Allgather(&my_n, 1, MPI_INT, counts.data(), 1, MPI_INT, ucomm) !=
            MPI_SUCCESS) {
          if (err == Err::None) {
            err = Err::Comm;
            detail = "MPI_Allgather of the histogram sizes failed";
          }
        }
        std::vector<int> displs(static_cast<std::size_t>(usize), 0);
        std::size_t      total = 0;
        for (std::size_t r = 0; r < static_cast<std::size_t>(usize); ++r) {
          displs[r] = static_cast<int>(total);
          total += static_cast<std::size_t>(counts[r]);
        }
        std::vector<u64> all(total == 0 ? 1 : total);
        if (MPI_Allgatherv(mine.empty() ? nullptr : mine.data(), my_n,
                           MPI_UINT64_T, all.data(), counts.data(), displs.data(),
                           MPI_UINT64_T, ucomm) != MPI_SUCCESS) {
          if (err == Err::None) {
            err = Err::Comm;
            detail = "MPI_Allgatherv of the cell histogram failed";
          }
        }
        std::map<u64, u64> merged;
        for (std::size_t k = 0; k + 1 < total; k += 2) merged[all[k]] += all[k + 1];

        u64 grand = 0;
        for (const auto& [m, c] : merged) {
          (void)m;
          grand += c;
        }
        const auto P = static_cast<u64>(usize);
        u64        acc_pop = 0;
        u64        next = 1;
        for (const auto& [m, c] : merged) {
          acc_pop += c;
          // grand * next / P, formed without overflowing.
          while (next < P &&
                 acc_pop >= (grand / P) * next + ((grand % P) * next) / P) {
            splitters.push_back(m);
            ++next;
          }
        }
        const u64 last = merged.empty() ? 0 : merged.rbegin()->first;
        while (splitters.size() + 1 < static_cast<std::size_t>(usize)) {
          splitters.push_back(last);
        }
      }

      const auto bucket_of = [&](u64 m) {
        const auto it = std::upper_bound(splitters.begin(), splitters.end(), m);
        auto       b = static_cast<int>(it - splitters.begin());
        if (b >= usize) b = usize - 1;
        return b;
      };

      // Round 1: candidates to their own cell's bucket, queries to their cell's
      // bucket AND to every bucket owning a neighbouring cell. Neighbours are
      // deduplicated by bucket, so a query costs a small constant number of
      // messages, not 3^dim of them.
      std::vector<Envelope> got1;
      {
        Outbox out(usize);
        if (err == Err::None) {
          for (std::size_t k = 0; k < s_loc.gid.size(); ++k) {
            const double x = s_loc.xyz[k * 3];
            const double y = s_loc.xyz[k * 3 + 1];
            const double z = s_loc.xyz[k * 3 + 2];
            const u64    m = morton_at(x, y, z);
            out.put(bucket_of(m),
                    PtRec{m, s_loc.gid[k], s_loc.idx[k], x, y, z, urank, 0});
          }
          query_points += d_loc.gid.size();
          for (std::size_t k = 0; k < d_loc.gid.size(); ++k) {
            const double x = d_loc.xyz[k * 3];
            const double y = d_loc.xyz[k * 3 + 1];
            const double z = d_loc.xyz[k * 3 + 2];
            u64          c[3];
            grid.bin(x, y, z, c);
            const u64 m = morton3(c[0], c[1], c[2]);
            int       seen[27];
            int       n_seen = 0;
            for (int dx = -1; dx <= 1; ++dx) {
              for (int dy = -1; dy <= 1; ++dy) {
                for (int dz = -1; dz <= 1; ++dz) {
                  const long long ix = static_cast<long long>(c[0]) + dx;
                  const long long iy = static_cast<long long>(c[1]) + dy;
                  const long long iz = static_cast<long long>(c[2]) + dz;
                  if (ix < 0 || iy < 0 || iz < 0) continue;
                  if (static_cast<u64>(ix) >= grid.n[0] ||
                      static_cast<u64>(iy) >= grid.n[1] ||
                      static_cast<u64>(iz) >= grid.n[2]) {
                    continue;
                  }
                  const int b = bucket_of(morton3(static_cast<u64>(ix),
                                                  static_cast<u64>(iy),
                                                  static_cast<u64>(iz)));
                  bool dup = false;
                  for (int q = 0; q < n_seen; ++q) {
                    if (seen[q] == b) dup = true;
                  }
                  if (dup) continue;
                  seen[n_seen++] = b;
                  ++query_probes;
                  out.put(b, PtRec{m, 0, d_loc.idx[k], x, y, z, urank, 1});
                }
              }
            }
          }
        }
        route(out.take(), &got1);
      }

      // The bucket's own search. Candidates are indexed by cell so a query is
      // compared against its own cell and the 26 around it, not against
      // everything the bucket happens to hold: Morton-adjacent cells land in
      // the same bucket, so that neighbourhood is mostly local.
      std::vector<Envelope> got2;
      {
        Outbox                              out(usize);
        std::vector<PtRec>                  cands;
        std::unordered_map<u64, std::vector<std::size_t>> by_cell;
        std::vector<PtRec>                  queries;
        for (const Envelope& e : got1) {
          bucket_records += e.payload.size() / sizeof(PtRec);
        }
        for_each_rec<PtRec>(got1, [&](const PtRec& r) {
          if (r.kind == 0) {
            by_cell[r.morton].push_back(cands.size());
            cands.push_back(r);
          } else {
            queries.push_back(r);
          }
        });

        for (const PtRec& q : queries) {
          u64 c[3];
          grid.bin(q.x, q.y, q.z, c);
          double best = std::numeric_limits<double>::infinity();
          const PtRec* win = nullptr;
          for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
              for (int dz = -1; dz <= 1; ++dz) {
                const long long ix = static_cast<long long>(c[0]) + dx;
                const long long iy = static_cast<long long>(c[1]) + dy;
                const long long iz = static_cast<long long>(c[2]) + dz;
                if (ix < 0 || iy < 0 || iz < 0) continue;
                if (static_cast<u64>(ix) >= grid.n[0] ||
                    static_cast<u64>(iy) >= grid.n[1] ||
                    static_cast<u64>(iz) >= grid.n[2]) {
                  continue;
                }
                const auto it = by_cell.find(morton3(static_cast<u64>(ix),
                                                     static_cast<u64>(iy),
                                                     static_cast<u64>(iz)));
                if (it == by_cell.end()) continue;
                for (const std::size_t ci : it->second) {
                  const PtRec& s = cands[ci];
                  const double ddx = s.x - q.x;
                  const double ddy = s.y - q.y;
                  const double ddz = s.z - q.z;
                  const double d = std::sqrt(ddx * ddx + ddy * ddy + ddz * ddz);
                  // Lowest global id breaks a tie, so the answer does not
                  // depend on the order records happened to arrive in.
                  if (d < best || (d == best && win != nullptr && s.gid < win->gid)) {
                    best = d;
                    win = &s;
                  }
                }
              }
            }
          }
          ReplyRec rep{q.idx, 0, 0, best, 0, 0};
          if (win != nullptr && best <= opts.nn_tolerance) {
            rep.gid = win->gid;
            rep.src_idx = win->idx;
            rep.src_rank = win->rank;
            rep.found = 1;
          }
          out.put(q.rank, rep);
        }
        route(out.take(), &got2);
      }

      // Reduce the replies, then tell the winning owner. This third round is
      // what GlobalId mode does not need: there, the bucket already knows the
      // owner and can address it directly, but here the owner is not known
      // until the requester has compared every bucket's answer.
      {
        struct Best {
          double dist = std::numeric_limits<double>::infinity();
          u64    gid = 0;
          u64    src_idx = 0;
          i32    src_rank = -1;
          bool   found = false;
          double nearest = std::numeric_limits<double>::infinity();
        };
        std::unordered_map<u64, Best> best;
        for_each_rec<ReplyRec>(got2, [&](const ReplyRec& r) {
          Best& b = best[r.dst_idx];
          b.nearest = std::min(b.nearest, r.dist);
          if (r.found == 0) return;
          if (!b.found || r.dist < b.dist ||
              (r.dist == b.dist && r.gid < b.gid)) {
            b.found = true;
            b.dist = r.dist;
            b.gid = r.gid;
            b.src_idx = r.src_idx;
            b.src_rank = r.src_rank;
          }
        });

        Outbox out(usize);
        for (std::size_t k = 0; k < d_loc.idx.size(); ++k) {
          const auto it = best.find(static_cast<u64>(d_loc.idx[k]));
          if (it == best.end() || !it->second.found) {
            if (err == Err::None) {
              const double nearest = it == best.end()
                                         ? std::numeric_limits<double>::infinity()
                                         : it->second.nearest;
              err = Err::NoMatch;
              detail =
                  "destination entity " + std::to_string(d_loc.idx[k]) +
                  " at (" + std::to_string(d_loc.xyz[k * 3]) + ", " +
                  std::to_string(d_loc.xyz[k * 3 + 1]) + ", " +
                  std::to_string(d_loc.xyz[k * 3 + 2]) +
                  ") has no source centroid within nn_tolerance = " +
                  std::to_string(opts.nn_tolerance) +
                  "; the nearest one found anywhere is at distance " +
                  (std::isfinite(nearest) ? std::to_string(nearest)
                                          : std::string{"infinity (no source "
                                                        "entity exists at all)"});
            }
            continue;
          }
          const Best&   b = it->second;
          const PairRec p{b.gid, b.src_idx, static_cast<u64>(d_loc.idx[k]),
                          b.src_rank, urank};
          // The requester already knows its own half; only the owner has to be
          // told, and only when it is somebody else.
          absorb(&acc, p, urank);
          if (b.src_rank != urank) out.put(b.src_rank, p);
        }
        std::vector<Envelope> got3;
        route(out.take(), &got3);
        for_each_rec<PairRec>(got3, [&](const PairRec& r) {
          absorb(&acc, r, urank);
        });
      }
    }
  } catch (const std::exception& e) {
    abort_job(ucomm, e.what());
  } catch (...) {
    abort_job(ucomm, "an unknown exception building the directory");
  }

  // -- Agree the outcome ---------------------------------------------------
  {
    auto      e = static_cast<int>(err);
    int       agreed = e;
    const int rc = MPI_Allreduce(&e, &agreed, 1, MPI_INT, MPI_MAX, ucomm);
    if (rc != MPI_SUCCESS) {
      return finish(Failure{"E-COMMGRAPH-COMM",
                          "MPI_Allreduce failed while agreeing the rendezvous "
                          "outcome"});
    }
    if (agreed != 0) {
      if (err != Err::None) return finish(Failure{code_of(err), detail});
      return finish(Failure{code_of(static_cast<Err>(agreed)),
                          "another rank rejected the rendezvous (" +
                              std::string{code_of(static_cast<Err>(agreed))} +
                              "); this rank's own side matched completely"});
    }
  }

  // -- Assemble ------------------------------------------------------------
  CommGraph g;
  g.participates_ = true;

  const auto emit = [&](std::map<int, std::vector<Entry>>& from,
                        std::vector<CommGraphPeer>* peers,
                        std::vector<std::size_t>*    ids,
                        std::vector<std::size_t>*    idxs,
                        bool                          dedupe_flat) {
    // The per-peer `CommGraphPeer` lists (`peers`) are keyed by destination
    // and MUST keep one entry per (peer, entity) pair -- the peer-wise
    // transport path (R31, `peer_plan_needed` above) gathers/scatters each
    // peer's own block through them. `ids`/`idxs` are a DIFFERENT, FLAT
    // thing: `TransportSpec::global_ids`, one rank's own declared offer
    // list, fed to `MpiTransport::build_plan`'s from-scratch id matching,
    // which requires exactly one entry per UNIQUE entity on a SOURCE rank
    // (its own "mine" map construction refuses a repeat as ambiguous
    // ownership) -- see `dedupe_flat_source_offers`'s own doc comment
    // (transport.hpp) for why the fan-out this exists to express does not
    // need the declaration itself repeated. `dedupe_flat` selects which
    // contract `ids`/`idxs` must satisfy: true for the SEND side (SOURCE,
    // where a single owned entity routinely has more than one destination
    // peer wanting it -- KNOWN_ISSUES.md D73 round 9), false for the RECV
    // side (DESTINATION, where a genuine repeat means this rank itself
    // wants the same remote entity written into two different local slots,
    // which `idxs`' own per-occurrence local index must keep distinct).
    std::map<int, std::vector<std::pair<std::uint64_t, std::size_t>>>
        flat_by_peer;
    for (auto& [peer, entries] : from) {
      // Both ends of an exchange sort on the same key -- the source id, then
      // the DESTINATION's local index -- which is what makes the two lists
      // line up element for element even when two destination entities match
      // one source entity.
      std::sort(entries.begin(), entries.end(),
                [](const Entry& a, const Entry& b) {
                  if (a.gid != b.gid) return a.gid < b.gid;
                  return a.key_idx < b.key_idx;
                });
      CommGraphPeer p;
      p.base_rank = union_ranks[static_cast<std::size_t>(peer)];
      p.local_indices.reserve(entries.size());
      p.global_ids.reserve(entries.size());
      std::vector<std::pair<std::uint64_t, std::size_t>>& flat =
          flat_by_peer[peer];
      flat.reserve(entries.size());
      for (const Entry& e : entries) {
        p.local_indices.push_back(e.own_idx);
        p.global_ids.push_back(static_cast<std::size_t>(e.gid));
        flat.emplace_back(e.gid, e.own_idx);
      }
      peers->push_back(std::move(p));
    }
    if (dedupe_flat) {
      dedupe_flat_source_offers(flat_by_peer, ids, idxs);
    } else {
      for (auto& [peer, offers] : flat_by_peer) {
        (void)peer;
        for (auto& [gid, idx] : offers) {
          ids->push_back(static_cast<std::size_t>(gid));
          idxs->push_back(idx);
        }
      }
    }
  };
  emit(acc.send, &g.send_, &g.send_ids_, &g.send_idx_, /*dedupe_flat=*/true);
  emit(acc.recv, &g.recv_, &g.recv_ids_, &g.recv_idx_, /*dedupe_flat=*/false);
  g.local_ = std::move(acc.local);

  g.stats_.bucket_records = bucket_records;
  g.stats_.query_points = query_points;
  g.stats_.query_probes = query_probes;
  g.stats_.router_rounds = router.last_rounds();
  g.union_size_ = usize;
  g.union_rank_ = urank;
  g.src_ranks_ = std::move(all_src);
  g.dst_ranks_ = std::move(all_dst);
  g.disjoint_ = disjoint;

  return finish(std::move(g));
}

// ---------------------------------------------------------------------------
// accumulate_shared_rows -- docs/dev/nodal-fields.md's "H^T on a distributed
// interface". See comm_graph.hpp for the contract.
// ---------------------------------------------------------------------------

Result<void> accumulate_shared_rows(const MeshView& view, int dim,
                                    std::size_t components,
                                    std::vector<double>& local_values,
                                    int base_comm_handle) {
  const std::size_t nv = view.entity_count(dim);

  // `base_comm_handle` is a structural precondition -- every rank is
  // required to pass the same, valid communicator -- so a null handle or a
  // failed MPI_Comm_size is checked and returned before any collective,
  // exactly like CommGraph::build's own E-COMMGRAPH-OPTS (comm_graph.cpp's
  // own header comment, point 1).
  const MPI_Comm comm = MPI_Comm_f2c(static_cast<MPI_Fint>(base_comm_handle));
  if (comm == MPI_COMM_NULL) {
    return Failure{"E-ACCUM-COMM",
                   "accumulate_shared_rows: base_comm_handle names "
                   "MPI_COMM_NULL"};
  }
  int num_ranks = 0;
  if (MPI_Comm_size(comm, &num_ranks) != MPI_SUCCESS) {
    return Failure{"E-ACCUM-COMM", "accumulate_shared_rows: MPI_Comm_size failed"};
  }

  // Review 2026-10-11 TR-7. `components` and `local_values` are per-rank
  // CALLER data, unlike `base_comm_handle` above -- a caller bug can size
  // either wrong on one rank only, and a local return here used to leave
  // every other rank to enter the MPI_Allgather below alone. The verdict is
  // agreed (MPI_Allreduce MAX) before anyone acts on it, the same
  // Collective-1 discipline build_plan() uses (mpi_transport.cpp).
  enum class SizeErr : int { None = 0, Components = 1, Size = 2 };
  SizeErr local_err = SizeErr::None;
  if (components < 1) {
    local_err = SizeErr::Components;
  } else if (local_values.size() != nv * components) {
    local_err = SizeErr::Size;
  }
  {
    auto      in  = static_cast<int>(local_err);
    int       out = in;
    const int rc  = MPI_Allreduce(&in, &out, 1, MPI_INT, MPI_MAX, comm);
    if (rc != MPI_SUCCESS) {
      return Failure{"E-ACCUM-COMM",
                     "accumulate_shared_rows: MPI_Allreduce while agreeing the "
                     "components/size check failed"};
    }
    const auto agreed = static_cast<SizeErr>(out);
    if (agreed != SizeErr::None) {
      if (local_err == SizeErr::Components) {
        return Failure{"E-ACCUM-COMPONENTS",
                       fmt::format("accumulate_shared_rows: components must be "
                                  "at least 1, got {}",
                                  components)};
      }
      if (local_err == SizeErr::Size) {
        return Failure{
            "E-ACCUM-SIZE",
            fmt::format("accumulate_shared_rows: {} entities x {} components "
                       "needs a buffer of {} values, got {}",
                       nv, components, nv * components, local_values.size())};
      }
      return Failure{agreed == SizeErr::Components ? "E-ACCUM-COMPONENTS" : "E-ACCUM-SIZE",
                     "accumulate_shared_rows: another rank's components/buffer-size "
                     "check failed; every rank refuses together rather than one side "
                     "entering the Allgather alone"};
    }
  }

  // Every LOCAL row contributes its own global id and value EXACTLY ONCE,
  // regardless of ownership (comm_graph.hpp's contract) -- unlike
  // gather_side() above, which restricts a SOURCE side to owned entities to
  // avoid ambiguous ownership. There is no such ambiguity here: this is a
  // SUM, not a copy, and every rank's local partial is a real, independent
  // contribution to the total, whether or not that rank happens to own the
  // vertex.
  std::vector<std::int64_t> local_gids(nv);
  for (std::size_t i = 0; i < nv; ++i) {
    local_gids[i] = static_cast<std::int64_t>(view.global_id(dim, i));
  }

  const int local_count = static_cast<int>(nv);
  std::vector<int> counts(static_cast<std::size_t>(num_ranks));
  if (MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, comm) !=
      MPI_SUCCESS) {
    return Failure{"E-ACCUM-COMM", "accumulate_shared_rows: MPI_Allgather (counts) failed"};
  }
  std::vector<int> displs(static_cast<std::size_t>(num_ranks));
  int total = 0;
  for (int r = 0; r < num_ranks; ++r) {
    displs[static_cast<std::size_t>(r)] = total;
    total += counts[static_cast<std::size_t>(r)];
  }

  std::vector<std::int64_t> gathered_gids(static_cast<std::size_t>(total));
  if (MPI_Allgatherv(local_gids.data(), local_count, MPI_INT64_T,
                     gathered_gids.data(), counts.data(), displs.data(),
                     MPI_INT64_T, comm) != MPI_SUCCESS) {
    return Failure{"E-ACCUM-COMM", "accumulate_shared_rows: MPI_Allgatherv (gids) failed"};
  }

  // Values: one MPI_Allgatherv per component, rather than one interleaved
  // call, so `counts`/`displs` (row counts) apply unchanged -- an
  // interleaved layout would need every count/displ scaled by `components`,
  // and a mismatch there is exactly the class of defect a second, tested
  // code path invites.
  std::vector<double> gathered_vals(static_cast<std::size_t>(total) * components);
  std::vector<double> local_component(nv);
  for (std::size_t c = 0; c < components; ++c) {
    for (std::size_t i = 0; i < nv; ++i) local_component[i] = local_values[i * components + c];
    std::vector<double> gathered_c(static_cast<std::size_t>(total));
    if (MPI_Allgatherv(local_component.data(), local_count, MPI_DOUBLE,
                       gathered_c.data(), counts.data(), displs.data(),
                       MPI_DOUBLE, comm) != MPI_SUCCESS) {
      return Failure{"E-ACCUM-COMM", "accumulate_shared_rows: MPI_Allgatherv (values) failed"};
    }
    for (std::size_t i = 0; i < static_cast<std::size_t>(total); ++i) {
      gathered_vals[i * components + c] = gathered_c[i];
    }
  }

  // Sum every gathered row into its global id's running total. A gid
  // appearing on several ranks (or, once, twice within one rank's own
  // segment -- not a valid partition, but summed rather than silently
  // overwritten either way) accumulates every contribution.
  std::unordered_map<std::int64_t, std::vector<double>> total_of;
  total_of.reserve(static_cast<std::size_t>(total));
  for (std::size_t i = 0; i < static_cast<std::size_t>(total); ++i) {
    auto [it, inserted] = total_of.try_emplace(gathered_gids[i], components, 0.0);
    for (std::size_t c = 0; c < components; ++c) it->second[c] += gathered_vals[i * components + c];
  }

  for (std::size_t i = 0; i < nv; ++i) {
    const auto& t = total_of.at(local_gids[i]);
    for (std::size_t c = 0; c < components; ++c) local_values[i * components + c] = t[c];
  }
  return Result<void>::success();
}

// ---------------------------------------------------------------------------
// copy_owner_shared_rows -- the Intensive/None companion. See comm_graph.hpp
// for the contract; the gather below mirrors accumulate_shared_rows's, with
// an `is_owned` flag riding alongside each row so the OWNER's contribution
// (not the sum) can be picked out.
// ---------------------------------------------------------------------------

Result<void> copy_owner_shared_rows(const MeshView& view, int dim,
                                    std::size_t components,
                                    std::vector<double>& local_values,
                                    int base_comm_handle) {
  const std::size_t nv = view.entity_count(dim);

  // `base_comm_handle` is a structural precondition, checked and returned
  // before any collective, same as accumulate_shared_rows above.
  const MPI_Comm comm = MPI_Comm_f2c(static_cast<MPI_Fint>(base_comm_handle));
  if (comm == MPI_COMM_NULL) {
    return Failure{"E-ACCUM-COMM",
                   "copy_owner_shared_rows: base_comm_handle names "
                   "MPI_COMM_NULL"};
  }
  int num_ranks = 0;
  if (MPI_Comm_size(comm, &num_ranks) != MPI_SUCCESS) {
    return Failure{"E-ACCUM-COMM", "copy_owner_shared_rows: MPI_Comm_size failed"};
  }

  // Review 2026-10-11 TR-7 (copy_owner_shared_rows is this finding's second
  // site). Same fix as accumulate_shared_rows: `components`/`local_values`
  // are per-rank caller data, so their verdict is agreed before anyone acts
  // on it rather than returned locally ahead of the MPI_Allgather below.
  enum class SizeErr : int { None = 0, Components = 1, Size = 2 };
  SizeErr local_err = SizeErr::None;
  if (components < 1) {
    local_err = SizeErr::Components;
  } else if (local_values.size() != nv * components) {
    local_err = SizeErr::Size;
  }
  {
    auto      in  = static_cast<int>(local_err);
    int       out = in;
    const int rc  = MPI_Allreduce(&in, &out, 1, MPI_INT, MPI_MAX, comm);
    if (rc != MPI_SUCCESS) {
      return Failure{"E-ACCUM-COMM",
                     "copy_owner_shared_rows: MPI_Allreduce while agreeing the "
                     "components/size check failed"};
    }
    const auto agreed = static_cast<SizeErr>(out);
    if (agreed != SizeErr::None) {
      if (local_err == SizeErr::Components) {
        return Failure{"E-ACCUM-COMPONENTS",
                       fmt::format("copy_owner_shared_rows: components must be "
                                  "at least 1, got {}",
                                  components)};
      }
      if (local_err == SizeErr::Size) {
        return Failure{
            "E-ACCUM-SIZE",
            fmt::format("copy_owner_shared_rows: {} entities x {} components "
                       "needs a buffer of {} values, got {}",
                       nv, components, nv * components, local_values.size())};
      }
      return Failure{agreed == SizeErr::Components ? "E-ACCUM-COMPONENTS" : "E-ACCUM-SIZE",
                     "copy_owner_shared_rows: another rank's components/buffer-size "
                     "check failed; every rank refuses together rather than one side "
                     "entering the Allgather alone"};
    }
  }

  std::vector<std::int64_t> local_gids(nv);
  std::vector<int>          local_owned(nv);
  for (std::size_t i = 0; i < nv; ++i) {
    local_gids[i]  = static_cast<std::int64_t>(view.global_id(dim, i));
    local_owned[i] = view.is_owned(dim, i) ? 1 : 0;
  }

  const int        local_count = static_cast<int>(nv);
  std::vector<int> counts(static_cast<std::size_t>(num_ranks));
  if (MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, comm) !=
      MPI_SUCCESS) {
    return Failure{"E-ACCUM-COMM", "copy_owner_shared_rows: MPI_Allgather (counts) failed"};
  }
  std::vector<int> displs(static_cast<std::size_t>(num_ranks));
  int total = 0;
  for (int r = 0; r < num_ranks; ++r) {
    displs[static_cast<std::size_t>(r)] = total;
    total += counts[static_cast<std::size_t>(r)];
  }

  std::vector<std::int64_t> gathered_gids(static_cast<std::size_t>(total));
  if (MPI_Allgatherv(local_gids.data(), local_count, MPI_INT64_T,
                     gathered_gids.data(), counts.data(), displs.data(),
                     MPI_INT64_T, comm) != MPI_SUCCESS) {
    return Failure{"E-ACCUM-COMM", "copy_owner_shared_rows: MPI_Allgatherv (gids) failed"};
  }
  std::vector<int> gathered_owned(static_cast<std::size_t>(total));
  if (MPI_Allgatherv(local_owned.data(), local_count, MPI_INT,
                     gathered_owned.data(), counts.data(), displs.data(),
                     MPI_INT, comm) != MPI_SUCCESS) {
    return Failure{"E-ACCUM-COMM", "copy_owner_shared_rows: MPI_Allgatherv (owned) failed"};
  }

  std::vector<double> gathered_vals(static_cast<std::size_t>(total) * components);
  std::vector<double> local_component(nv);
  for (std::size_t c = 0; c < components; ++c) {
    for (std::size_t i = 0; i < nv; ++i) local_component[i] = local_values[i * components + c];
    std::vector<double> gathered_c(static_cast<std::size_t>(total));
    if (MPI_Allgatherv(local_component.data(), local_count, MPI_DOUBLE,
                       gathered_c.data(), counts.data(), displs.data(),
                       MPI_DOUBLE, comm) != MPI_SUCCESS) {
      return Failure{"E-ACCUM-COMM", "copy_owner_shared_rows: MPI_Allgatherv (values) failed"};
    }
    for (std::size_t i = 0; i < static_cast<std::size_t>(total); ++i) {
      gathered_vals[i * components + c] = gathered_c[i];
    }
  }

  // Exactly one gathered row per global id may claim ownership -- more than
  // one is two ranks disagreeing about docs/dev/native-mesh.md's rule (a
  // mesh contract violation), and zero is a global id nobody holds as owner,
  // neither of which this function can resolve by guessing.
  std::unordered_map<std::int64_t, std::vector<double>> owner_of;
  owner_of.reserve(static_cast<std::size_t>(total));
  for (std::size_t i = 0; i < static_cast<std::size_t>(total); ++i) {
    if (gathered_owned[i] != 1) continue;
    std::vector<double> row(components);
    for (std::size_t c = 0; c < components; ++c) row[c] = gathered_vals[i * components + c];
    const auto [it, inserted] = owner_of.try_emplace(gathered_gids[i], std::move(row));
    if (!inserted) {
      return Failure{"E-ACCUM-OWNER",
                     fmt::format("copy_owner_shared_rows: more than one rank "
                                "claims ownership of global id {}",
                                gathered_gids[i])};
    }
  }

  // Review 2026-10-11 TR-8. This used to walk THIS rank's own `local_gids`
  // -- a per-rank subset of the mesh -- so a missing owner fired only on
  // the ranks that happen to hold the orphaned vertex among their own
  // entities, while every other rank returned success for the very same
  // global defect: a mixed outcome across ranks for one shared mesh
  // contract violation. `gathered_gids` is the identical union on every
  // rank (it came from the Allgatherv above), so checking that instead
  // makes the check -- and its outcome -- the same everywhere, exactly
  // like the "more than one owner" check just above it.
  for (std::size_t i = 0; i < static_cast<std::size_t>(total); ++i) {
    if (owner_of.find(gathered_gids[i]) != owner_of.end()) continue;
    return Failure{"E-ACCUM-OWNER",
                   fmt::format("copy_owner_shared_rows: no rank claims "
                              "ownership of global id {}",
                              gathered_gids[i])};
  }

  for (std::size_t i = 0; i < nv; ++i) {
    // Every id was just confirmed, identically on every rank, to have
    // exactly one owner -- this lookup cannot fail.
    const auto it = owner_of.find(local_gids[i]);
    for (std::size_t c = 0; c < components; ++c) local_values[i * components + c] = it->second[c];
  }
  return Result<void>::success();
}

}  // namespace fuse

#endif  // FUSE_HAVE_MPI
