#pragma once

/// Restart (docs/dev/restart.md): the Runner-level
/// orchestration two free functions below provide -- writing/restoring a
/// restart point's TIME-LEVEL scalars and every restartable component's
/// and accelerated region's own state (`Runner::write_restart_fields()`/
/// `restore_restart_fields()`, fuse/runtime/runner.hpp). Backend-agnostic:
/// no HDF5 type appears here, and `finish_point()` is called from this
/// layer, not from the concrete writer's own create()/open().
///
/// PROVENANCE IS DELIBERATELY NOT HERE. `RestartWriter::put_scalar()`
/// stores a `double`, exact only up to 2^53 -- fine for a time level or a
/// period index (the SAME bound `src/backend/native_mesh_io.cpp`'s own
/// header comment already accepts for mesh global ids: "ids ride as
/// double, exact for every integer up to 2^53"), but NOT fine for a
/// `fnv1a` descriptor hash, whose 64-bit range routinely exceeds it.
/// Provenance (format version, FUSE version, descriptor hash, rank
/// count, mesh checksum) is therefore the CONCRETE backend's own job,
/// written as real fixed-width HDF5 attributes at file-create time and
/// checked at file-open time, BEFORE a caller ever gets a `RestartReader`
/// to pass to `restore_restart_point()` -- see fuse/restart_io/
/// hdf5_restart_io.hpp's own `create_restart_point_file()`/
/// `open_restart_point_file()`.

#include <cstdint>

#include "fuse/base/rational.hpp"
#include "fuse/base/result.hpp"
#include "fuse/model/restart.hpp"
#include "fuse/runtime/component.hpp"
#include "fuse/runtime/runner.hpp"

namespace fuse {

/// Writes `time`/`period_index` (docs/dev/restart.md Section 4.1's
/// `@time_numerator`/`@time_denominator`/`@period_index`) plus every
/// restartable component's own state and every accelerated region's
/// accelerator memory into `writer`, then calls `writer.finish_point()`
/// -- the batched id check, which must run exactly once,
/// AFTER every other call for this point has already succeeded.
[[nodiscard]] Result<void> write_restart_point(Runner& runner,
                                               ComponentSet& components,
                                               RestartWriter& writer,
                                               const Rational& time,
                                               std::int64_t period_index);

/// The read-back counterpart. `reader` must already be open on a point
/// whose provenance (format version, descriptor hash, rank count, mesh
/// checksum) the caller has already checked -- this function restores
/// FIELDS only, never provenance. `out_time`/`out_period_index` receive
/// the point's own recorded values, so the caller can resume `Runner::
/// run()`/`run_period()` counting from where the point was taken.
/// `allow_accelerator_reset_on_mismatch`/`any_accelerator_reset` are
/// `Runner::restore_restart_fields()`'s own parameters, forwarded
/// unchanged (the off-by-default reset option).
[[nodiscard]] Result<void> restore_restart_point(
    Runner& runner, ComponentSet& components, RestartReader& reader,
    bool allow_accelerator_reset_on_mismatch, bool& any_accelerator_reset,
    Rational& out_time, std::int64_t& out_period_index);

/// Owner decision 1's own rank-count refusal: a restart on a DIFFERENT
/// rank count is refused BY NAME, collectively, if ANY component in the
/// graph offers Form B (`kCapabilityRestartFormB`) without ALSO offering
/// Form A (`kCapabilityRestartFormA`) -- a B-only component's own blob is
/// opaque, same-decomposition-only, by construction (docs/dev/restart.md
/// Section 3). `written_by_rank_count` is the restart point's own
/// recorded provenance value; `target_rank_count` is the CURRENT run's
/// own rank count. Equal values never refuse, regardless of which forms
/// any component offers.
[[nodiscard]] Result<void> check_restart_rank_count(
    const ComponentSet& components, std::uint32_t written_by_rank_count,
    std::uint32_t target_rank_count);

}  // namespace fuse
