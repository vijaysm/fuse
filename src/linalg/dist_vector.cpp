#include "fuse/linalg/dist_vector.hpp"

#include <algorithm>
#include <cmath>

#include "fuse/base/error.hpp"

namespace fuse::linalg {

namespace {

/// Preconditions here are caller mistakes (two vectors built against
/// different distributions), not a runtime outcome -- throws rather than
/// returning a Result, the same split least_squares.cpp's check_shape
/// draws between a caller's shape error and a Result-carried failure.
void check_same_distribution(const DistVector& a, const DistVector& b, const char* op) {
  if (a.owned_size() != b.owned_size()) {
    FUSE_THROW(InvalidArgument,
               "DistVector::{}: owned sizes disagree ({} vs {}); the two vectors were not "
               "built against the same distribution",
               op, a.owned_size(), b.owned_size());
  }
}

}  // namespace

DistVector::DistVector(std::size_t owned_n, std::shared_ptr<const GhostExchanger> exchanger,
                       std::shared_ptr<const Reducer> reducer)
    : storage_(owned_n + (exchanger ? exchanger->ghost_size() : 0), 0.0),
      owned_n_(owned_n),
      exchanger_(std::move(exchanger)),
      reducer_(std::move(reducer)) {
  FUSE_ASSERT(exchanger_ != nullptr, "DistVector built with a null GhostExchanger");
  FUSE_ASSERT(reducer_ != nullptr, "DistVector built with a null Reducer");
}

Result<void> DistVector::update_ghosts() {
  return exchanger_->update(storage_.data(), owned_n_, storage_.data() + owned_n_,
                            ghost_size());
}

double DistVector::dot(const DistVector& other) const {
  check_same_distribution(*this, other, "dot");
  double local = 0.0;
  for (std::size_t i = 0; i < owned_n_; ++i) local += storage_[i] * other.storage_[i];
  return reducer_->sum(local);
}

double DistVector::norm2() const { return std::sqrt(dot(*this)); }

double DistVector::norm_inf() const {
  double local = 0.0;
  for (std::size_t i = 0; i < owned_n_; ++i) local = std::max(local, std::fabs(storage_[i]));
  return reducer_->max(local);
}

void DistVector::axpy(double a, const DistVector& x) {
  check_same_distribution(*this, x, "axpy");
  FUSE_ASSERT(size() == x.size(),
             "DistVector::axpy: total sizes disagree ({} vs {}); the two vectors carry a "
             "different number of ghosts",
             size(), x.size());
  for (std::size_t i = 0; i < storage_.size(); ++i) storage_[i] += a * x.storage_[i];
}

void DistVector::scale(double a) {
  for (double& v : storage_) v *= a;
}

void DistVector::set_zero() { std::fill(storage_.begin(), storage_.end(), 0.0); }

void DistVector::copy_from(const DistVector& other) {
  check_same_distribution(*this, other, "copy_from");
  FUSE_ASSERT(size() == other.size(),
             "DistVector::copy_from: total sizes disagree ({} vs {}); the two vectors carry a "
             "different number of ghosts",
             size(), other.size());
  storage_ = other.storage_;
}

}  // namespace fuse::linalg
