#pragma once
// Collective agreement on a per-rank outcome, with provenance.
//
// A collective path never returns on a locally computed condition: the ranks
// agree first, then every rank returns the SAME status. Before 2026-10-30 the
// agreement in FieldTransaction::commit_collective() (and the plugin-mount
// agreement in src/abi/participant.cpp) carried only a 0/1 verdict and the
// originating rank across the communicator, so a non-originating rank could
// name the rank that refused but not WHY it refused; the original code and
// message existed on one rank only. The owner's rule (KNOWN_ISSUES.md D97,
// 2026-10-30): the original error and the rank it originated from must be
// known on every rank, for provenance and debugging.
//
// agree_outcome() is that rule as one function: after the verdict is reduced,
// the originating rank broadcasts its own Failure (code and message) and
// every rank, the originator included, returns the identical Failure whose
// `code` is the agreed code the caller names, whose `origin_rank` and
// `origin_code` carry the provenance as data, and whose message starts with
// the origin rank and the count of refusing ranks followed by the original
// message verbatim. When several ranks refuse, the lowest-numbered one is
// the origin reported (MPI_MAXLOC breaks the tie in the refused flag on the
// smallest index, MPI 4.1 Section 6.9.4) and the count says how many
// there were; the other refusals are visible on their own ranks' logs.
//
// LAYERING: this header names no MPI type (the communicator is the Fortran
// integer handle the rest of fuse passes around); the .cpp is the one place
// in `field` that calls MPI, under `#if FUSE_HAVE_MPI`, the exception
// docs/dev/layering.md's "MPI outside transport" section records.

#include <cstdint>
#include <string>
#include <string_view>

#include "fuse/base/result.hpp"

namespace fuse {

/// The data the agreement moves from the originating rank to every rank.
struct AgreedRefusal {
  int origin_rank = -1;        ///< the rank whose refusal is reported (the lowest refusing one)
  int refusing_ranks = 0;      ///< how many ranks refused (>= 1)
  int communicator_size = 0;   ///< how many ranks took part
  std::string origin_code;     ///< the origin's own Failure.code
  std::string origin_message;  ///< the origin's own Failure.message, verbatim
  std::int32_t origin_status = 0;  ///< the origin's numeric status, when the caller has one
};

/// The one message shape every agreed refusal carries, pure so tier 0 can
/// pin it: "<phase>: rank R of N refused (K refusing rank(s); the
/// lowest-numbered is reported) with <origin_code>: <origin_message>".
[[nodiscard]] std::string format_agreed_refusal(std::string_view phase,
                                                const AgreedRefusal& refusal);

/// Build the Failure every rank returns: `agreed_code`, the formatted
/// message, and the provenance fields set from `refusal`.
[[nodiscard]] Failure make_agreed_failure(std::string_view agreed_code,
                                          std::string_view phase,
                                          const AgreedRefusal& refusal);

/// Agree `local` across every rank of `fortran_communicator` (an MPI
/// communicator's Fortran handle, MPI_Comm_c2f). Collective: every rank of
/// the communicator must call it with the same `agreed_code` and `phase`.
/// Returns success on every rank iff every rank's `local` is ok; otherwise
/// returns the identical Failure on every rank (see the header comment).
/// A failed MPI call returns Failure{comm_failure_code, ...} on the rank that
/// saw it; that is the one case where ranks can disagree, and the caller's
/// communicator is then unusable anyway.
///
/// Without MPI in the build, a call with any handle returns
/// Failure{comm_failure_code, ...}: there is no communicator to agree over,
/// and silently returning `local` would be a locally computed condition
/// dressed as an agreement.
///
/// `local_status_number` is an optional numeric status beside `local` (the C
/// ABI's fuse_status); the origin's value travels with its code and message
/// and is returned in `Failure::origin_status`, so a caller with a numeric
/// status space returns the ORIGIN's number on every rank, not a reduction
/// over numbers that could name a different rank than the message does.
[[nodiscard]] Result<void> agree_outcome(int fortran_communicator, const Result<void>& local,
                                         std::string_view agreed_code,
                                         std::string_view comm_failure_code,
                                         std::string_view phase,
                                         std::int32_t local_status_number = 0) noexcept;

}  // namespace fuse
