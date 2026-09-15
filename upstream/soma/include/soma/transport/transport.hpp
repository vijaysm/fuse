#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "soma/base/result.hpp"

namespace soma {

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
/// immediately before `move()`, inside the period loop, from a `TransportSpec`
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
};

/// In-process copy. Phase 5 adds "mpi"; Phase 8 adds "lowfive".
///
/// `move` requires that `src` and `dst` either designate the same address (a
/// no-op, which the lowering pass legitimately emits when a component reads
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

#ifdef SOMA_HAVE_MPI
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
/// refuses the overlap (`E-TRANSPORT-RANKS`) rather than silently using the
/// source partition for both. A rank set that overlaps is a local exchange
/// anyway: that is what `CopyTransport` is for.
///
/// ISOLATION. `set_communicator` duplicates the communicator, so each
/// `MpiTransport` owns a private communication context. Two channels running
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
  /// Every rank returns the same verdict: local validation is agreed with an
  /// `MPI_Allreduce` before any rank acts on it, and the matching itself is
  /// computed from data every rank has, so a pattern error is an identical
  /// error return everywhere instead of one rank bailing out while the others
  /// block in a collective.
  ///
  /// Errors: `E-TRANSPORT-COMM`, `E-TRANSPORT-RANKS`, `E-TRANSPORT-SIZE`,
  /// `E-TRANSPORT-MATCH`, `E-TRANSPORT-MPI`.
  Result<void> build_plan(const TransportSpec& spec);

  /// Hot loop. Allocation-free and communication-free; checks only that
  /// `spec` is the one `build_plan` was given.
  /// Errors: `E-TRANSPORT-UNPLANNED`, `E-TRANSPORT-RESPEC`.
  Result<void> prepare(const TransportSpec& spec) override;

  /// COLLECTIVE. `n` must be exactly `elements * components` of the planned
  /// spec. `src` and `dst` may be null when `n` is zero -- a rank that owns no
  /// elements of a region is ordinary in parallel, and `std::vector<double>{}
  /// .data()` is null.
  /// Errors: `E-TRANSPORT-UNPLANNED`, `E-TRANSPORT-NULL`,
  /// `E-TRANSPORT-SIZE`, `E-TRANSPORT-MPI`.
  Result<void> move(const double* src, double* dst, std::size_t n) override;

  [[nodiscard]] std::string name() const override { return "mpi"; }
  [[nodiscard]] bool is_local() const override { return false; }

  /// Doubles this rank sends / receives per `move`. Diagnostic; zero before
  /// `build_plan`.
  [[nodiscard]] std::size_t send_volume() const noexcept;
  [[nodiscard]] std::size_t recv_volume() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
#endif  // SOMA_HAVE_MPI

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
/// after `main` returns (docs/BUILDING.md records the trace). The connector is
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
  /// (`-DSOMA_ENABLE_LOWFIVE=ON` and the find module's dlopen probe passed).
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

}  // namespace soma
