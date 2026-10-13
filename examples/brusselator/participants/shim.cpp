#include "shim.h"
#include "brusselator_grid.h"
#include "lowfive_attach.h"

#include <cstring>
#include <string>
#include <utility>

#include "fuse/backend/native_backend.hpp"
#include "fuse/base/config.hpp"
#include "fuse/base/rational.hpp"
#include "fuse/linalg/comm.hpp"

#include "../species_core.hpp"

#if FUSE_HAVE_LOWFIVE
#include <hdf5.h>
#endif

namespace {

// `std::string`, not a fixed char buffer: a failure message
// (brusselator::UCore::step()'s own messages) is built with std::to_string
// concatenations of unbounded length (reaction-tolerance/iteration-count/
// time-bounds), so truncating it into a fixed buffer risked losing the
// one diagnostic value (which window, which time bound) the caller needs
// to act on. Owned by the handle, overwritten each failing step() call,
// exactly as shim.h's own doc comment on bruss_*_last_error() promises.
std::string ToStdString(const fuse::Failure& f) {
  return f.code + ": " + f.message;
}

// Physics track: the communication pair a decomposed (`num_ranks > 1`)
// participant needs, built from the caller's own `comm_f` (an
// `MPI_Comm_c2f` handle of THIS component's physics-only communicator --
// shim.h's own doc comment on `bruss_u_create`/`bruss_v_create` has the
// full account). A null `shared_ptr`/empty `std::function` return is this
// pair's own failure sentinel (no exception crosses the C ABI from here):
// `num_ranks > 1` in a build without `FUSE_ENABLE_MPI` is exactly that
// failure, since there is no other way to form a real cross-rank
// collective.
std::shared_ptr<const fuse::linalg::Reducer> MakeReducerFor(int num_ranks, int64_t comm_f) {
  if (num_ranks <= 1) return fuse::linalg::make_serial_reducer();
#if FUSE_HAVE_MPI
  return fuse::linalg::make_mpi_reducer(static_cast<int>(comm_f));
#else
  (void)comm_f;
  return nullptr;
#endif
}

brusselator::GhostExchangerFactory MakeExchangerFactoryFor(int num_ranks, int64_t comm_f) {
  if (num_ranks <= 1) return brusselator::serial_exchanger_factory();
#if FUSE_HAVE_MPI
  const int local_comm_f = static_cast<int>(comm_f);
  return [local_comm_f](const fuse::linalg::GhostPattern& pattern, std::size_t owned_n) {
    return fuse::linalg::make_mpi_exchanger(local_comm_f, pattern, owned_n);
  };
#else
  (void)comm_f;
  return brusselator::GhostExchangerFactory{};
#endif
}

}  // namespace

struct bruss_u_handle {
  brusselator::UCore core;
  std::string last_error;

  bruss_u_handle(const brusselator::NativeGrid& grid, brusselator::PhysicsConfig cfg,
                brusselator::Vector initial_v_on_u,
                std::shared_ptr<const fuse::linalg::Reducer> reducer,
                brusselator::GhostExchangerFactory make_exchanger)
      : core(grid, cfg, std::move(initial_v_on_u), std::move(reducer), std::move(make_exchanger)) {}
};

struct bruss_v_handle {
  brusselator::VParticipant core;
  std::string last_error;

  explicit bruss_v_handle(brusselator::VParticipant c) : core(std::move(c)) {}
};

extern "C" {

bruss_u_handle* bruss_u_create(int nx, int ny, double diffusion, double A,
                               double B, int forcing,
                               const double* initial_v_on_u,
                               int rank, int num_ranks, int64_t comm_f,
                               const char* solver) {
  const auto kind_r = brusselator::parse_solver_kind(solver ? solver : "auto");
  if (!kind_r.ok()) return nullptr;

  fuse::native::NativeMeshBackend backend(rank, num_ranks);
  auto grid_r = brusselator::NativeGrid::make(backend, "u", nx, ny);
  if (!grid_r.ok()) return nullptr;
  const brusselator::NativeGrid& grid = grid_r.value();

  auto reducer = MakeReducerFor(num_ranks, comm_f);
  if (!reducer) return nullptr;
  auto exchanger_factory = MakeExchangerFactoryFor(num_ranks, comm_f);
  if (!exchanger_factory) return nullptr;

  brusselator::PhysicsConfig cfg;
  cfg.diffusion = diffusion;
  cfg.A = A;
  cfg.B = B;
  cfg.forcing = forcing != 0;
  cfg.solver = kind_r.value();

  brusselator::Vector v0(grid.owned_size());
  for (int i = 0; i < grid.owned_size(); ++i) v0[i] = initial_v_on_u[i];

  // make_diffusion_solver (species_core.hpp, called from UCore's own
  // constructor) THROWS fuse::InvalidArgument for a caller-configuration
  // mistake (SolverKind::Direct explicitly requested at num_ranks() > 1)
  // -- a C++ exception must never cross this extern "C" boundary (u_main.c/
  // v_main.f90 are not exception-aware frames), so it is caught here and
  // turned into this function's own NULL-return convention, exactly like
  // every other construction-time failure above.
  try {
    return new (std::nothrow)
        bruss_u_handle(grid, cfg, std::move(v0), std::move(reducer), std::move(exchanger_factory));
  } catch (const fuse::Error&) {
    return nullptr;
  }
}

void bruss_grid_dims(int* nx_u, int* ny_u, int* nx_v, int* ny_v) {
  if (nx_u) *nx_u = BRUSS_NX_U;
  if (ny_u) *ny_u = BRUSS_NY_U;
  if (nx_v) *nx_v = BRUSS_NX_V;
  if (ny_v) *ny_v = BRUSS_NY_V;
}
double bruss_diffusion(void) { return BRUSS_DIFFUSION; }

int bruss_owned_size(int nx, int ny, int rank, int num_ranks) {
  fuse::native::NativeMeshBackend backend(rank, num_ranks);
  auto grid_r = brusselator::NativeGrid::make(backend, "owned_size_probe", nx, ny);
  if (!grid_r.ok()) return -1;
  return grid_r.value().owned_size();
}

int bruss_owned_slice(const double* whole, int nx, int ny, int rank, int num_ranks,
                      double* owned_out) {
  fuse::native::NativeMeshBackend backend(rank, num_ranks);
  auto grid_r = brusselator::NativeGrid::make(backend, "owned_slice_probe", nx, ny);
  if (!grid_r.ok()) return 1;
  const brusselator::NativeGrid& grid = grid_r.value();
  for (int lj = 0; lj < grid.owned_rows(); ++lj) {
    const int j = grid.j_lo() + lj;
    for (int i = 0; i < nx; ++i) {
      owned_out[grid.local_index(i, j)] = whole[grid.index(i, j)];
    }
  }
  return 0;
}

void bruss_u_provided(const bruss_u_handle* h, double* out) {
  const brusselator::Vector& p = h->core.provided();
  for (int i = 0; i < h->core.n(); ++i) out[i] = p[i];
}

void bruss_u_destroy(bruss_u_handle* h) { delete h; }

int bruss_u_size(const bruss_u_handle* h) { return h->core.n(); }

int bruss_u_step(bruss_u_handle* h, long long t_begin_num, long long t_begin_den,
                 long long t_end_num, long long t_end_den, const double* required,
                 double* provided_out) {
  const int n = h->core.n();
  brusselator::Vector req(n);
  for (int i = 0; i < n; ++i) req[i] = required[i];
  // EXACT rational arithmetic, rounded to double ONCE -- the same
  // discipline fuse::StepContext::dt() follows, and the reason this
  // shim takes t_begin/t_end as rationals rather than doubles (shim.h's
  // own comment on bruss_u_step has the full account, including the
  // bit-identical check that caught the alternative).
  const fuse::Rational t_begin{t_begin_num, t_begin_den};
  const fuse::Rational t_end{t_end_num, t_end_den};
  const double t_begin_d = t_begin.to_double();
  const double t_end_d = t_end.to_double();
  const double dt_d = (t_end - t_begin).to_double();
  const fuse::Result<void> r = h->core.step(t_begin_d, t_end_d, dt_d, req);
  if (!r.ok()) {
    h->last_error = ToStdString(r.error());
    return 1;
  }
  const brusselator::Vector& provided = h->core.provided();
  for (int i = 0; i < n; ++i) provided_out[i] = provided[i];
  return 0;
}

void bruss_u_save_state(bruss_u_handle* h) { h->core.save_state(); }
void bruss_u_restore_state(bruss_u_handle* h) { h->core.restore_state(); }

void bruss_u_state(const bruss_u_handle* h, double* out) {
  const brusselator::Vector& s = h->core.state();
  for (int i = 0; i < h->core.n(); ++i) out[i] = s[i];
}

double bruss_u_external_total(const bruss_u_handle* h) { return h->core.external_total(); }
double bruss_u_exchange_out(const bruss_u_handle* h) { return h->core.exchange_out(); }
double bruss_u_state_mean(const bruss_u_handle* h) { return h->core.state_mean(); }
const char* bruss_u_last_error(const bruss_u_handle* h) { return h->last_error.c_str(); }

bruss_v_handle* bruss_v_create(int nx_u, int ny_u, int nx_v, int ny_v, double diffusion,
                               int rank, int num_ranks, int64_t comm_f, const char* solver) {
  const auto kind_r = brusselator::parse_solver_kind(solver ? solver : "auto");
  if (!kind_r.ok()) return nullptr;

  fuse::native::NativeMeshBackend backend(rank, num_ranks);
  // Halo track: set BEFORE loading U/V (native_halo_mpi_test.cpp's own
  // Fixture order) so `fuse::build_remap_halo`'s own `communicator_size()`
  // check, made inside `VParticipant::make`, sees the real value. A
  // single rank never adopts anything regardless (`communicator_size()`
  // reports 1 with no communicator set at all, the same no-op this
  // header's own doc comment on `set_communicator` states), so skipping
  // the call at `num_ranks<=1` changes nothing observable and avoids
  // dereferencing a communicator handle nothing provided one for.
  if (num_ranks > 1) backend.set_communicator(static_cast<fuse::native::CommHandle>(comm_f));

  auto reducer = MakeReducerFor(num_ranks, comm_f);
  if (!reducer) return nullptr;
  auto exchanger_factory = MakeExchangerFactoryFor(num_ranks, comm_f);
  if (!exchanger_factory) return nullptr;

  // Same reasoning as bruss_u_create's own try/catch: a caller-
  // configuration mistake (e.g. SolverKind::Direct at num_ranks() > 1)
  // THROWS fuse::InvalidArgument from inside make_diffusion_solver
  // (species_core.hpp, called from VDiffusionCore's own constructor,
  // itself called from VParticipant::make), which must never cross this
  // extern "C" boundary.
  try {
    auto made = brusselator::VParticipant::make(std::move(backend), nx_u, ny_u, nx_v, ny_v,
                                                diffusion, kind_r.value(), /*solver_tolerance=*/1e-13,
                                                std::move(reducer), std::move(exchanger_factory));
    if (!made.ok()) return nullptr;
    return new (std::nothrow) bruss_v_handle(std::move(made.value()));
  } catch (const fuse::Error&) {
    return nullptr;
  }
}

void bruss_v_provided(const bruss_v_handle* h, double* out) {
  // const_cast: provided() is non-const only because the remap helper it
  // calls is; nothing observable changes (species_core.hpp's own note).
  auto p = const_cast<bruss_v_handle*>(h)->core.provided();
  if (!p.ok()) {
    const_cast<bruss_v_handle*>(h)->last_error = ToStdString(p.error());
    for (int i = 0; i < h->core.n_u(); ++i) out[i] = 0.0;
    return;
  }
  for (int i = 0; i < h->core.n_u(); ++i) out[i] = p.value()[i];
}

void bruss_v_destroy(bruss_v_handle* h) { delete h; }

int bruss_v_size(const bruss_v_handle* h) { return h->core.n_u(); }

int bruss_v_step(bruss_v_handle* h, long long t_begin_num, long long t_begin_den,
                 long long t_end_num, long long t_end_den, const double* required,
                 double* provided_out) {
  const int n = h->core.n_u();
  brusselator::Vector req(n);
  for (int i = 0; i < n; ++i) req[i] = required[i];
  // EXACT rational arithmetic, rounded to double ONCE -- shim.h's own
  // comment on bruss_v_step has the full account (mirrors bruss_u_step
  // above).
  const fuse::Rational t_begin{t_begin_num, t_begin_den};
  const fuse::Rational t_end{t_end_num, t_end_den};
  const double dt = (t_end - t_begin).to_double();
  brusselator::Vector provided(n);
  const fuse::Result<void> r = h->core.step(dt, req, &provided);
  if (!r.ok()) {
    h->last_error = ToStdString(r.error());
    return 1;
  }
  for (int i = 0; i < n; ++i) provided_out[i] = provided[i];
  return 0;
}

void bruss_v_save_state(bruss_v_handle* h) { h->core.save_state(); }
void bruss_v_restore_state(bruss_v_handle* h) { h->core.restore_state(); }

void bruss_v_state(const bruss_v_handle* h, double* out) {
  const brusselator::Vector& s = h->core.state();
  for (int i = 0; i < static_cast<int>(s.size()); ++i) out[i] = s[i];
}

double bruss_v_exchange_in(const bruss_v_handle* h) { return h->core.exchange_in(); }
double bruss_v_state_mean(const bruss_v_handle* h) { return h->core.state_mean(); }
const char* bruss_v_last_error(const bruss_v_handle* h) { return h->last_error.c_str(); }

}  // extern "C"

namespace {

// Copies `msg` into `errbuf` (capacity `errbuf_len`), truncated, always
// NUL-terminated -- shim.h's own out-parameter convention for the two
// functions below. A null/zero-length errbuf is a no-op, not a fault: a
// caller that does not want the message (unlikely, but not this
// function's call to refuse) still gets the nonzero return code.
void CopyErrorMessage(const std::string& msg, char* errbuf, long long errbuf_len) {
  if (!errbuf || errbuf_len <= 0) return;
  const auto cap = static_cast<std::size_t>(errbuf_len - 1);
  const std::size_t n = msg.size() < cap ? msg.size() : cap;
  std::memcpy(errbuf, msg.data(), n);
  errbuf[n] = '\0';
}

}  // namespace

#if FUSE_HAVE_LOWFIVE

namespace {

// Shared by both functions below: HDF5's own C error-reporting
// convention (H5Eget_current_stack/H5Eprint2-style walking) is a lot of
// machinery for a demo shim -- this just formats the HDF5 call that
// failed and leaves the detailed stack on stderr, where HDF5's own
// default error handler already sent it (never disabled here).
std::string Hdf5CallFailed(const char* what, const char* file, const char* dataset) {
  return std::string("HDF5 call failed: ") + what + " (file=" + file +
         ", dataset=" + dataset + ") -- see stderr for the HDF5 stack trace";
}

}  // namespace

#endif  // FUSE_HAVE_LOWFIVE

// `extern "C" { ... }` wraps BOTH branches of every #if below (rather
// than each branch opening/closing its own) -- the preprocessor sees
// only one branch per build, and an extern "C" opened inside a branch
// that the OTHER build's preprocessor pass skips entirely would leave
// this file's one closing brace (the end of this translation unit)
// unmatched in that configuration. Caught by this session's own
// standalone syntax check against FUSE_HAVE_LOWFIVE=0 before it reached
// a real build.
extern "C" {

int bruss_lowfive_available(void) { return FUSE_HAVE_LOWFIVE; }

#if FUSE_HAVE_LOWFIVE

int bruss_lowfive_write(const char* file, const char* dataset,
                        const double* data, long long n,
                        long long fapl_id,
                        char* errbuf, long long errbuf_len) {
  // `fapl_id`, not H5P_DEFAULT, per shim.h's own updated comment ("LowFive
  // attach" track): the VOL that makes this interception rather than a
  // real file on disk is attached PROGRAMMATICALLY, by bruss_lowfive_
  // attach() (lowfive_attach.h), and this call's own fapl_id argument is
  // the file-access property list that carries it. A negative fapl_id
  // means attach either was never called or failed -- refused here,
  // rather than silently falling back to H5P_DEFAULT, which would reach
  // a real file with no interception and no diagnostic.
  if (fapl_id < 0) {
    CopyErrorMessage(
        "bruss_lowfive_write('" + std::string(file) +
            "'): fapl_id < 0; call bruss_lowfive_attach() successfully "
            "first",
        errbuf, errbuf_len);
    return 1;
  }
  const hid_t fapl = static_cast<hid_t>(fapl_id);
  const hid_t fid = H5Fcreate(file, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
  if (fid < 0) {
    CopyErrorMessage(Hdf5CallFailed("H5Fcreate", file, dataset), errbuf, errbuf_len);
    return 1;
  }
  const hsize_t dims[1] = {static_cast<hsize_t>(n)};
  const hid_t space = H5Screate_simple(1, dims, nullptr);
  if (space < 0) {
    H5Fclose(fid);
    CopyErrorMessage(Hdf5CallFailed("H5Screate_simple", file, dataset), errbuf, errbuf_len);
    return 1;
  }
  const hid_t did = H5Dcreate2(fid, dataset, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT,
                               H5P_DEFAULT, H5P_DEFAULT);
  if (did < 0) {
    H5Sclose(space);
    H5Fclose(fid);
    CopyErrorMessage(Hdf5CallFailed("H5Dcreate2", file, dataset), errbuf, errbuf_len);
    return 1;
  }
  const herr_t wrc =
      H5Dwrite(did, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
  H5Dclose(did);
  H5Sclose(space);
  H5Fclose(fid);
  if (wrc < 0) {
    CopyErrorMessage(Hdf5CallFailed("H5Dwrite", file, dataset), errbuf, errbuf_len);
    return 1;
  }
  // "called by the producer side after it has written ... and called
  // its own, unmodified H5Fclose" -- fuse::LowFiveTransport::
  // serve_as_producer()'s own placement (fuse/transport/transport.hpp),
  // mirrored here; a no-op when the live attach is single-process.
  if (bruss_lowfive_serve_after_write(errbuf, errbuf_len) != 0) {
    return 1;
  }
  return 0;
}

int bruss_lowfive_read(const char* file, const char* dataset,
                       double* data, long long n,
                       long long fapl_id,
                       char* errbuf, long long errbuf_len) {
  if (fapl_id < 0) {
    CopyErrorMessage(
        "bruss_lowfive_read('" + std::string(file) +
            "'): fapl_id < 0; call bruss_lowfive_attach() successfully "
            "first",
        errbuf, errbuf_len);
    return 1;
  }
  // "called by the CONSUMER side BEFORE its own, unmodified H5Fopen" --
  // fuse::LowFiveTransport::receive_as_consumer()'s own placement
  // (fuse/transport/transport.hpp), mirrored here; a no-op when the
  // live attach is single-process.
  if (bruss_lowfive_receive_before_read(errbuf, errbuf_len) != 0) {
    return 1;
  }
  const hid_t fapl = static_cast<hid_t>(fapl_id);
  const hid_t fid = H5Fopen(file, H5F_ACC_RDONLY, fapl);
  if (fid < 0) {
    CopyErrorMessage(Hdf5CallFailed("H5Fopen", file, dataset), errbuf, errbuf_len);
    return 1;
  }
  const hid_t did = H5Dopen2(fid, dataset, H5P_DEFAULT);
  if (did < 0) {
    H5Fclose(fid);
    CopyErrorMessage(Hdf5CallFailed("H5Dopen2", file, dataset), errbuf, errbuf_len);
    return 1;
  }
  // `n` is the caller's own buffer length, checked against the dataset's
  // actual extent rather than trusted -- a mismatch here means the
  // writer's and this reader's own graphs disagree about this field's
  // width, which a silently short/overrunning H5Dread must never paper
  // over.
  const hid_t space = H5Dget_space(did);
  hsize_t dims[1] = {0};
  if (space >= 0) H5Sget_simple_extent_dims(space, dims, nullptr);
  if (space < 0 || dims[0] != static_cast<hsize_t>(n)) {
    if (space >= 0) H5Sclose(space);
    H5Dclose(did);
    H5Fclose(fid);
    CopyErrorMessage(
        Hdf5CallFailed("H5Dget_space/extent-mismatch", file, dataset), errbuf,
        errbuf_len);
    return 1;
  }
  H5Sclose(space);
  const herr_t rrc =
      H5Dread(did, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
  H5Dclose(did);
  H5Fclose(fid);
  if (rrc < 0) {
    CopyErrorMessage(Hdf5CallFailed("H5Dread", file, dataset), errbuf, errbuf_len);
    return 1;
  }
  return 0;
}

#else  // !FUSE_HAVE_LOWFIVE

int bruss_lowfive_write(const char*, const char*, const double*, long long,
                        long long,
                        char* errbuf, long long errbuf_len) {
  CopyErrorMessage("not built with LowFive (FUSE_ENABLE_LOWFIVE was OFF)", errbuf,
                   errbuf_len);
  return 1;
}

int bruss_lowfive_read(const char*, const char*, double*, long long,
                       long long,
                       char* errbuf, long long errbuf_len) {
  CopyErrorMessage("not built with LowFive (FUSE_ENABLE_LOWFIVE was OFF)", errbuf,
                   errbuf_len);
  return 1;
}

#endif  // FUSE_HAVE_LOWFIVE

}  // extern "C"