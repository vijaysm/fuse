// The one-rank, no-MPI-symbol-touched Reducer and GhostExchanger
// (comm.hpp's own header comment): always compiled, the reference every
// other implementation's one-participant behaviour must agree with.

#include "fuse/linalg/comm.hpp"

#include <algorithm>

#include "fuse/base/error.hpp"

namespace fuse::linalg {

namespace {

class SerialReducer : public Reducer {
 public:
  [[nodiscard]] double sum(double local) const override { return local; }
  void sum(double* inout, std::size_t n) const override {
    // One participant: every value already IS the global sum; nothing to
    // combine it with. `inout`/`n` are unused on purpose, not a dangling
    // reference to a removed body.
    (void)inout;
    (void)n;
  }
  [[nodiscard]] double max(double local) const override { return local; }
  [[nodiscard]] int rank() const override { return 0; }
  [[nodiscard]] int size() const override { return 1; }

  void allgather(const double* send, std::size_t n_each, double* recv) const override {
    // One participant: the concatenation over `[0, size())` is this
    // participant's own `send`, copied once, same as every other
    // `SerialReducer` operation being its argument's identity.
    std::copy(send, send + n_each, recv);
  }
};

class SerialExchanger : public GhostExchanger {
 public:
  [[nodiscard]] std::size_t ghost_size() const override { return 0; }

  [[nodiscard]] Result<void> update(const double* owned, std::size_t owned_n, double* ghosts,
                                    std::size_t ghost_n) const override {
    (void)owned;
    (void)owned_n;
    (void)ghosts;
    // A one-rank build has no peer to carry a ghost's value, so a nonzero
    // ghost_n reaching here is this library's own wiring mistake (a
    // DistVector was built with the serial exchanger but a nonempty
    // GhostPattern), not a recoverable runtime condition -- the same
    // distinction collectives.cpp's refuse_no_mpi() draws between "the
    // caller's input was bad" (Result) and "our own invariant broke"
    // (FUSE_ASSERT).
    FUSE_ASSERT(ghost_n == 0,
               "SerialExchanger::update called with ghost_n={} > 0; the serial exchanger "
               "carries no pattern and can only answer for zero ghosts",
               ghost_n);
    return Result<void>::success();
  }
};

}  // namespace

std::shared_ptr<const Reducer> make_serial_reducer() {
  return std::make_shared<const SerialReducer>();
}

std::shared_ptr<const GhostExchanger> make_serial_exchanger() {
  return std::make_shared<const SerialExchanger>();
}

}  // namespace fuse::linalg
