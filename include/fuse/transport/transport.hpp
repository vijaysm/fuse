#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "fuse/base/config.hpp"
#include "fuse/base/result.hpp"

namespace fuse {

struct CommGraphPeer;
struct LocalCopyPair;

struct TransportSpec {
  int         src_rank_set = -1;
  int         dst_rank_set = -1;
  std::size_t elements = 0;
  int         components = 1;

  /// Redistribution pattern for non-local transports (spec §4.1: a component's
  /// ranks and the coupler's ranks may be disjoint). All three are empty for a
  /// local transport, which is why `CopyTransport` never looks at them.
  ///
  /// `src_ranks` and `dst_ranks` are ranks of the communicator handed to
  /// `MpiTransport::set_communicator`, and must be **disjoint** -- see the
  /// note on `MpiTransport` for why a single `global_ids` list cannot describe
  /// a rank that is in both.
  ///
  /// `global_ids` is *this* rank's list: the entity ids it holds if it is a
  /// source, or the ids it wants if it is a destination. Its length must equal
  /// `elements`.
  std::vector<int>         src_ranks;
  std::vector<int>         dst_ranks;
  std::vector<std::size_t> global_ids;
};

/// Spec §2.2: the one seam by which in-process, MPI, MPMD, and LowFive
/// coupling share a runtime. Everything else in the DAG is unchanged.
///
/// HOT-LOOP CONTRACT (Task 4.6). `Runner::Impl::execute` calls `prepare()`
/// immediately before `move()`, inside the coupling-cycle loop, from a `TransportSpec`
/// resolved once at setup. `prepare()` is therefore required to be
/// allocation-free and communication-free: it may re-read the spec, it may
/// validate it against work already done, but it may NOT build a plan. A
/// backend that needs a plan builds it in a one-time, explicitly collective
/// entry point of its own -- see `MpiTransport::build_plan`.
class Transport {
 public:
  virtual ~Transport() = default;

  virtual Result<void> prepare(const TransportSpec& spec) = 0;
  virtual Result<void> move(const double* src, double* dst, std::size_t n) = 0;
  [[nodiscard]] virtual std::string name() const = 0;
  [[nodiscard]] virtual bool is_local() const = 0;

  /// True iff `move()` may be handed device pointers directly for THIS
  /// backend, right now, in this process (docs/dev/device-resident-arcs.md
  /// Phase 3). Not pure: `false` is the correct answer for every backend
  /// that never crosses a process boundary (`CopyTransport` -- the
  /// Runner's own `OpKind::Transport` case does a `Kokkos::deep_copy`
  /// there directly and never asks) or that always stages through host
  /// buffers today (`LowFiveTransport`). `MpiTransport` is the one
  /// override that can answer true, and does so at run time via
  /// `fuse::linalg::mpi_is_gpu_aware()` (comm.hpp) -- the SAME probe
  /// `fuse::native::mpi_is_gpu_aware()` calls, no longer duplicated
  /// between the two libraries (Migration track, 2026-10-15:
  /// src/linalg/mpi_gpu_aware.cpp's own header comment has the history).
  [[nodiscard]] virtual bool is_gpu_aware() const noexcept { return false; }
};

/// In-process copy. Phase 5 adds "mpi"; Phase 8 adds "lowfive".
///
/// `move` requires that `src` and `dst` either designate the same address (a
/// no-op, which compiling into the dataflow graph legitimately emits when a component reads
/// its own output) or do not overlap at all. Partial overlap is not part of
/// the contract.
class CopyTransport final : public Transport {
 public:
  Result<void> prepare(const TransportSpec& spec) override;
  Result<void> move(const double* src, double* dst, std::size_t n) override;
  [[nodiscard]] std::string name() const override { return "copy"; }
  [[nodiscard]] bool is_local() const override { return true; }

 private:
  // Only the two fields `move` reads, not the whole TransportSpec. `prepare`
  // runs inside the hot loop, and `spec_ = spec` on a struct that now carries
  // three std::vectors is a heap allocation the moment any of them is
  // non-empty -- silently, and only in the configurations that fill them.
  std::size_t elements_ = 0;
  int         components_ = 1;
};

#if FUSE_HAVE_MPI
/// n->m redistribution across two disjoint rank sets (spec §4.1, §6 mode 1).
///
/// The communicator crosses the boundary as `MPI_Comm_c2f(comm)`, so this
/// header names no MPI type and `mpi.h` stays off a consumer's include path,
/// exactly as `MoabMeshBackend::set_communicator` does.
///
/// WHERE THE WORK HAPPENS. `build_plan()` is the one-time, collective setup:
/// it validates the pattern, exchanges global ids, and builds the
/// `MPI_Alltoallv` displacement plan and its buffers. `prepare()` is the hot
/// loop's call and does nothing but check that the spec it is handed is the
/// one that was planned -- no allocation, no MPI, no matching. `move()` packs,
/// exchanges and unpacks into buffers `build_plan()` already sized.
///
/// COLLECTIVE CALLS, in order: `set_communicator` (it duplicates the
/// communicator), `build_plan`, every `move`, and the destructor (it frees the
/// duplicate). Every rank of the communicator must reach each of them, in the
/// same order -- including a rank that is in neither rank set and moves no
/// data at all, which participates with zero counts.
///
/// WHY THE RANK SETS MUST BE DISJOINT. `TransportSpec` carries one
/// `global_ids` list and one `elements` count per rank. For a rank that is
/// only a source they mean "what I hold"; for a rank that is only a
/// destination they mean "what I want". A rank in both sets has two
/// distributions and no way to write the second one down, so `build_plan`
/// rejects the overlap (`E-TRANSPORT-RANKS`) rather than silently using the
/// source partition for both. A rank set that overlaps is a local exchange
/// anyway: that is what `CopyTransport` is for.
///
/// ISOLATION. `set_communicator` duplicates the communicator, so each
/// `MpiTransport` owns a private communication context. Two field transfers running
/// concurrently on the same user communicator therefore cannot mismatch each
/// other's collectives, and the duplicate carries `MPI_ERRORS_RETURN` so an
/// MPI failure becomes a `Result` rather than an abort inside the caller's
/// communicator.
class MpiTransport final : public Transport {
 public:
  MpiTransport();
  ~MpiTransport() override;
  MpiTransport(MpiTransport&&) noexcept;
  MpiTransport& operator=(MpiTransport&&) noexcept;
  MpiTransport(const MpiTransport&) = delete;
  MpiTransport& operator=(const MpiTransport&) = delete;

  /// COLLECTIVE. Adopts a duplicate of `MPI_Comm_f2c(fortran_comm_handle)`.
  /// Errors: `E-TRANSPORT-MPIINIT` if MPI is not initialised,
  /// `E-TRANSPORT-COMM` for `MPI_COMM_NULL` or a call made after
  /// `build_plan`, `E-TRANSPORT-MPI` if an MPI call fails.
  Result<void> set_communicator(int fortran_comm_handle);

  /// COLLECTIVE, once, before the run. Builds the redistribution plan.
  ///
  /// Every rank returns the same outcome: local validation is agreed with an
  /// `MPI_Allreduce` before any rank acts on it, and the matching itself is
  /// computed from data every rank has, so a pattern error is an identical
  /// error return everywhere instead of one rank bailing out while the others
  /// block in a collective.
  ///
  /// Errors: `E-TRANSPORT-COMM`, `E-TRANSPORT-RANKS`, `E-TRANSPORT-SIZE`,
  /// `E-TRANSPORT-MATCH`, `E-TRANSPORT-MPI`, `E-TRANSPORT-PEERMATCH` (R-30:
  /// ranks disagreeing on `TransportSpec::components`), `E-TRANSPORT-ALLOC`.
  Result<void> build_plan(const TransportSpec& spec);

  /// COLLECTIVE, once, before the run. The same plan, built from a rank graph's
  /// **peer-wise** form instead of matched from a `TransportSpec`.
  ///
  /// WHY A SECOND BUILDER. `TransportSpec` carries one `global_ids` list per
  /// rank, which cannot describe a rank that is a source for some entities and
  /// a destination for others -- and under partial overlap a rank genuinely
  /// can be both. `CommGraph::send_plan()`/`recv_plan()` express the answer in
  /// every overlap shape, and `local_pairs()` carries the matches whose two
  /// ends are on one rank, which never travel. Handed those three, this
  /// function does no matching at all: the peer lists already hold exactly
  /// what `MPI_Alltoallv`'s count arrays need, peer by peer in ascending rank.
  ///
  /// THE NUMBERING PRECONDITION, checked rather than assumed. `CommGraphPeer`
  /// ranks are in the **base** communicator the rank graph was built over,
  /// while `MPI_Alltoallv` indexes its counts by rank in *this* transport's
  /// communicator. The two agree only if `set_communicator` was given that
  /// same base communicator. Rather than trust it, this builder exchanges the
  /// intended counts (`MPI_Alltoall`) and requires every peer's expectation to
  /// match what its partner intends to send: a numbering mismatch is
  /// `E-TRANSPORT-PEERMATCH` before the run, not data delivered to the wrong
  /// rank during it.
  ///
  /// `elements` and `components` describe THIS rank's field extent, exactly as
  /// in `build_plan`. Errors: `E-TRANSPORT-COMM`, `E-TRANSPORT-RANKS`,
  /// `E-TRANSPORT-SIZE`, `E-TRANSPORT-MATCH` (a peer list not in ascending
  /// rank order), `E-TRANSPORT-PEERMATCH`, `E-TRANSPORT-MPI`,
  /// `E-TRANSPORT-ALLOC`.
  Result<void> build_plan_from_peers(const std::vector<CommGraphPeer>& send,
                                     const std::vector<CommGraphPeer>& recv,
                                     const std::vector<LocalCopyPair>& local,
                                     std::size_t elements, int components);

  /// Hot loop. Allocation-free and communication-free; checks only that
  /// `spec` is the one `build_plan` was given.
  /// Errors: `E-TRANSPORT-UNPLANNED`, `E-TRANSPORT-RESPEC`.
  Result<void> prepare(const TransportSpec& spec) override;

  /// COLLECTIVE. `n` must be exactly `elements * components` of the planned
  /// spec. `src` and `dst` may be null when `n` is zero -- a rank that owns no
  /// elements of a mesh entity set is ordinary in parallel, and `std::vector<double>{}
  /// .data()` is null.
  /// Errors: `E-TRANSPORT-UNPLANNED`, `E-TRANSPORT-NULL`,
  /// `E-TRANSPORT-SIZE`, `E-TRANSPORT-MPI`.
  Result<void> move(const double* src, double* dst, std::size_t n) override;

  [[nodiscard]] std::string name() const override { return "mpi"; }
  [[nodiscard]] bool is_local() const override { return false; }

  /// `fuse::linalg::mpi_is_gpu_aware()`, cached once (the probe is a
  /// syscall-free library query, but `prepare()`'s hot-loop contract bars
  /// even that from `move()`, so this reads the cached value, never the
  /// probe itself).
  [[nodiscard]] bool is_gpu_aware() const noexcept override;

  /// Doubles this rank sends / receives per `move`. Diagnostic; zero before
  /// `build_plan`.
  [[nodiscard]] std::size_t send_volume() const noexcept;
  [[nodiscard]] std::size_t recv_volume() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
#endif  // FUSE_HAVE_MPI

/// Task 8.4, spec §5 mechanism 4: coupled data carried through LowFive,
/// Argonne's HDF5 Virtual Object Layer connector. A component writes its
/// fields with ordinary `H5Fcreate` / `H5Dcreate` / `H5Dwrite` calls; the VOL
/// connector intercepts them in memory and the data is handed to the coupler
/// with **no file on disk**.
///
/// DECLARED UNCONDITIONALLY, unlike `MpiTransport` above. Every member below
/// links in every configuration; on a build without LowFive they return
/// `E-TRANSPORT-NOLOWFIVE` naming the CMake option, rather than vanishing from
/// the header and breaking a consumer's translation unit (Task 5.2's lesson:
/// an always-visible header must not declare a function that does not link).
/// `is_available()` is the cheap capability query.
///
/// WHERE THE WORK HAPPENS, exactly as `MpiTransport` splits it. Registering
/// the VOL connector, building the file-access property list, sizing the
/// dataspace and proving that interception is actually in effect all happen
/// once in `build_plan()`. `prepare()` is the hot loop's call: it compares
/// scalars against the planned spec and does nothing else -- no allocation, no
/// HDF5, no property-list work.
///
/// HOW THE CONNECTOR IS ATTACHED. Programmatically, via
/// `LowFive::H5VOLProperty::apply()` on a private FAPL -- **not** through the
/// `HDF5_VOL_CONNECTOR` environment variable, which makes the process abort
/// after `main` returns (docs/user/lowfive.md records the trace). The connector is
/// registered exactly once per process and intentionally never unregistered:
/// `LowFive::MetadataVOL::create_MetadataVOL()` is a singleton and
/// `register_plugin()` is not reference counted, so two live `H5VOLProperty`
/// objects share one `vol_id` and the first destructor invalidates it under
/// the second (observed: "not a VOL connector ID").
///
/// IT FAILS LOUDLY RATHER THAN WRITING A FILE. `build_plan()` proves the
/// directory it was pointed at is writable, then round-trips a probe dataset
/// and checks (a) that LowFive's `after_dataset_write` callback fired, which
/// is connector code executing, and (b) that no file appeared at the path.
/// Every `move()` re-checks both. A configuration in which the connector did
/// not attach therefore reports `E-TRANSPORT-LF-NOINTERCEPT` or
/// `E-TRANSPORT-LF-ONDISK`; it never silently degrades to file I/O.
///
/// NOT THREAD-SAFE, and serial: the VOL singleton, its interception counter
/// and its in-memory file map are process-wide state shared by every
/// `LowFiveTransport`. Two transports must therefore name different files;
/// `set_location` rejects a name another live transport already claimed.
class LowFiveTransport final : public Transport {
 public:
  LowFiveTransport();
  ~LowFiveTransport() override;
  LowFiveTransport(LowFiveTransport&&) noexcept;
  LowFiveTransport& operator=(LowFiveTransport&&) noexcept;
  LowFiveTransport(const LowFiveTransport&) = delete;
  LowFiveTransport& operator=(const LowFiveTransport&) = delete;

  /// True iff this build was configured with a usable LowFive
  /// (`-DFUSE_ENABLE_LOWFIVE=ON` and the find module's dlopen probe passed).
  [[nodiscard]] static bool is_available() noexcept;

  /// One-time setup. `file` is the HDF5 file name the writing component would
  /// have used; it is never created, but its directory must exist and be
  /// writable -- otherwise "no file on disk" would be true for the wrong
  /// reason and would prove nothing.
  /// Errors: `E-TRANSPORT-NOLOWFIVE`, `E-TRANSPORT-LF-LOCATION`.
  Result<void> set_location(std::string file, std::string dataset);

  /// One-time setup, after `set_location`. Registers the connector, builds
  /// the FAPL and the dataspace, and proves interception.
  /// Errors: `E-TRANSPORT-NOLOWFIVE`, `E-TRANSPORT-LF-LOCATION`,
  /// `E-TRANSPORT-SIZE`, `E-TRANSPORT-LF-VOL`,
  /// `E-TRANSPORT-LF-NOINTERCEPT`, `E-TRANSPORT-LF-ONDISK`,
  /// `E-TRANSPORT-HDF5`.
  Result<void> build_plan(const TransportSpec& spec);

  // Review 2026-10-11 TR-3: this class used to also declare
  // `build_plan_from_peers`, copied from `MpiTransport`'s contract, with no
  // definition anywhere in the tree -- an undefined-symbol link error for
  // any caller, and a doc describing MPI_Alltoallv machinery LowFive does
  // not have. Removed rather than defined: nothing in the tree calls it
  // (runner.cpp's own peer-wise path calls it only through an
  // `MpiTransport*`), and the comm-graph peer-wise plan this class would
  // need it for has no LowFive-specific meaning to add.

  /// Hot loop. Allocation-free; checks only that `spec` is the planned one.
  /// Errors: `E-TRANSPORT-UNPLANNED`, `E-TRANSPORT-RESPEC`.
  Result<void> prepare(const TransportSpec& spec) override;

  /// Writes `src` through HDF5 and reads it back into `dst`, both through the
  /// VOL. `n` must be exactly `elements * components` of the planned spec.
  /// Errors: `E-TRANSPORT-UNPLANNED`, `E-TRANSPORT-SIZE`,
  /// `E-TRANSPORT-NULL`, `E-TRANSPORT-HDF5`,
  /// `E-TRANSPORT-LF-NOINTERCEPT`, `E-TRANSPORT-LF-ONDISK`.
  Result<void> move(const double* src, double* dst, std::size_t n) override;

  [[nodiscard]] std::string name() const override { return "lowfive"; }
  /// False: the data leaves the process's ordinary memory and travels through
  /// the HDF5 stack, which is the whole point of the backend.
  [[nodiscard]] bool is_local() const override { return false; }

  /// The intercepting file-access property list, as a plain integer. It is an
  /// HDF5 `hid_t`; it crosses this boundary as `long long` for the same reason
  /// `MpiTransport::set_communicator` takes `MPI_Comm_c2f(comm)` -- so that
  /// `hdf5.h` stays off a consumer's include path. Negative before
  /// `build_plan`. Hand it to an otherwise unmodified HDF5 program as the
  /// `fapl_id` of its `H5Fcreate`/`H5Fopen` and its I/O is intercepted.
  [[nodiscard]] long long access_plist_handle() const noexcept;

  /// Dataset writes LowFive has intercepted for this transport. Zero means
  /// nothing went through the connector. Diagnostic.
  [[nodiscard]] std::size_t intercepted_writes() const noexcept;

  /// The path that is deliberately never created.
  [[nodiscard]] std::string file_path() const;

  /// Task 8.4 continuation (S6 LowFive track, Demo B): switches this
  /// transport into LowFive's DISTRIBUTED mode, where data crosses an MPI
  /// intercommunicator to a PEER PROCESS instead of staying in this
  /// process's own in-memory file map. This is what makes two separate
  /// executables -- not two calls in one process -- the actual coupling in
  /// Demo B (examples/demo_b).
  ///
  /// MUST be called before `set_location()`; it decides which VOL connector
  /// (LowFive's `MetadataVOL` or `DistMetadataVOL`) this process attaches,
  /// and a process attaches at most one, ever (see the note on
  /// `vol_singleton()` in lowfive_transport.cpp -- unregistering and
  /// re-registering a different one is not safe).
  ///
  /// `local_comm`/`intercomm` cross this boundary as `MPI_Comm_c2f(...)`
  /// handles, exactly as `MpiTransport::set_communicator` takes one, so this
  /// header never names `mpi.h`. `intercomm` is the MPI intercommunicator
  /// between this process's group and the peer executable's -- obtained via
  /// `MPI_Comm_connect`/`MPI_Comm_accept` (examples/demo_b's own rendezvous)
  /// or an MPMD launch's split communicator; forming it is NOT this
  /// transport's job.
  ///
  /// Measured end to end for ONE producer process and ONE consumer process
  /// (ctest row example.demo_b, owner's machine 2026-10-14: values bit for
  /// bit, no HDF5 file on disk). Not measured: more than one rank per side,
  /// more than one file or dataset per transport, or a second exchange in
  /// the same process. KNOWN_ISSUES.md D14 is the authoritative record.
  ///
  /// Errors: `E-TRANSPORT-NOLOWFIVE`, `E-TRANSPORT-NOMPI` (built without
  /// `FUSE_ENABLE_MPI`), `E-TRANSPORT-MPIINIT`, `E-TRANSPORT-COMM`,
  /// `E-TRANSPORT-LF-LOCATION` (called after `set_location`/`build_plan`).
  Result<void> set_intercomm(int fortran_local_comm, int fortran_intercomm);

  /// Distributed mode only, called by the PRODUCER side after it has
  /// written its dataset(s) and called its own, unmodified `H5Fclose`:
  /// pushes the served file across the intercommunicator
  /// (`DistMetadataVOL::serve_all`). Refused (`E-TRANSPORT-LF-NOTDIST`) if
  /// `set_intercomm` was never called.
  Result<void> serve_as_producer();

  /// Distributed mode only, called by the CONSUMER side BEFORE its own,
  /// unmodified `H5Fopen`: receives the producer's file index across the
  /// intercommunicator (`DistMetadataVOL::broadcast_files`), so the
  /// subsequent `H5Fopen` resolves against served, in-memory data rather
  /// than a file that was never written to disk. Refused
  /// (`E-TRANSPORT-LF-NOTDIST`) if `set_intercomm` was never called.
  Result<void> receive_as_consumer();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class TransportRegistry {
 public:
  using Factory = std::function<std::unique_ptr<Transport>()>;

  /// Registering a name that is already present replaces its factory.
  void register_backend(std::string name, Factory factory);
  [[nodiscard]] Result<std::unique_ptr<Transport>> create(
      const std::string& name) const;
  [[nodiscard]] std::vector<std::string> available() const;

  static TransportRegistry& instance();

 private:
  std::vector<std::pair<std::string, Factory>> factories_;
};

}  // namespace fuse
