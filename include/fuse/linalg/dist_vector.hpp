#pragma once

/// A dense vector split across participants by row: each one holds an
/// OWNED block in its own local order, plus a GHOST region -- copies of
/// entries other participants own, refreshed on request rather than kept
/// continuously in sync. `DistSparseMatrix::apply` is the reason the ghost
/// region exists at all: `y = A x` needs the off-process columns' current
/// values, and a ghost update is the one communication step that supplies
/// them (docs/dev/linalg-distributed.md has the full layout).
///
/// Communication is injected (`fuse::linalg::Reducer`,
/// `fuse::linalg::GhostExchanger`, comm.hpp), never included: this header
/// names neither `mpi.h` nor any MPI type, the same boundary `linalg`'s
/// position in docs/dev/layering.md requires (it depends on `base` alone).

#include <cstddef>
#include <memory>
#include <vector>

#include "fuse/linalg/comm.hpp"

namespace fuse::linalg {

class DistVector {
 public:
  /// `owned_n` entries owned by this participant; the ghost count is
  /// `exchanger->ghost_size()` (comm.hpp's header comment states why that
  /// is where it comes from rather than a fourth argument here). Storage
  /// is one contiguous buffer, owned block first, so `data()` matches the
  /// layout `DistSparseMatrix::apply`'s local CSR products read directly.
  DistVector(std::size_t owned_n, std::shared_ptr<const GhostExchanger> exchanger,
             std::shared_ptr<const Reducer> reducer);

  [[nodiscard]] double* data() noexcept { return storage_.data(); }
  [[nodiscard]] const double* data() const noexcept { return storage_.data(); }
  [[nodiscard]] std::size_t owned_size() const noexcept { return owned_n_; }
  [[nodiscard]] std::size_t ghost_size() const noexcept { return storage_.size() - owned_n_; }
  [[nodiscard]] std::size_t size() const noexcept { return storage_.size(); }

  /// `data()[owned_size() + g]` becomes `exchanger`'s current answer for
  /// ghost `g`, for every `g`. COLLECTIVE under MPI -- see
  /// `GhostExchanger::update`'s own documentation.
  [[nodiscard]] Result<void> update_ghosts();

  /// The owned-block partial dot, reduced across every participant. Reads
  /// only `[0, owned_size())` of `this` and `other` -- a caller that wants
  /// the ghost region included has mixed up which block means what.
  [[nodiscard]] double dot(const DistVector& other) const;
  /// `sqrt(dot(*this, *this))`.
  [[nodiscard]] double norm2() const;
  /// The owned-block partial max absolute value, reduced (via
  /// `Reducer::max`) across every participant.
  [[nodiscard]] double norm_inf() const;

  /// `this <- this + a * x`, owned block AND ghost region both (an axpy
  /// is elementwise and needs no communication either way; leaving the
  /// ghost region stale here would silently disagree with whichever of
  /// `this`/`x` was updated more recently).
  void axpy(double a, const DistVector& x);
  void scale(double a);
  void set_zero();
  void copy_from(const DistVector& other);

 private:
  std::vector<double> storage_;  ///< [0, owned_n_) owned, [owned_n_, size()) ghost
  std::size_t owned_n_ = 0;
  std::shared_ptr<const GhostExchanger> exchanger_;
  std::shared_ptr<const Reducer> reducer_;
};

}  // namespace fuse::linalg
