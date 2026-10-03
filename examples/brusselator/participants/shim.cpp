#include "shim.h"

#include <cstring>
#include <string>
#include <utility>

#include "fuse/base/rational.hpp"

#include "../species_core.hpp"

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

}  // namespace

struct bruss_u_handle {
  brusselator::UCore core;
  std::string last_error;

  bruss_u_handle(int nx, int ny, brusselator::PhysicsConfig cfg,
                brusselator::Vector initial_v_on_u)
      : core(nx, ny, cfg, std::move(initial_v_on_u)) {}
};

struct bruss_v_handle {
  brusselator::VParticipant core;
  std::string last_error;

  explicit bruss_v_handle(brusselator::VParticipant c) : core(std::move(c)) {}
};

extern "C" {

bruss_u_handle* bruss_u_create(int nx, int ny, double diffusion, double A,
                               double B, int forcing,
                               const double* initial_v_on_u) {
  brusselator::PhysicsConfig cfg;
  cfg.diffusion = diffusion;
  cfg.A = A;
  cfg.B = B;
  cfg.forcing = forcing != 0;
  brusselator::Vector v0(nx * ny);
  for (int i = 0; i < nx * ny; ++i) v0[i] = initial_v_on_u[i];
  return new (std::nothrow) bruss_u_handle(nx, ny, cfg, std::move(v0));
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
const char* bruss_u_last_error(const bruss_u_handle* h) { return h->last_error.c_str(); }

bruss_v_handle* bruss_v_create(int nx_u, int ny_u, int nx_v, int ny_v, double diffusion) {
  auto made = brusselator::VParticipant::make(nx_u, ny_u, nx_v, ny_v, diffusion);
  if (!made.ok()) return nullptr;
  return new (std::nothrow) bruss_v_handle(std::move(made.value()));
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
const char* bruss_v_last_error(const bruss_v_handle* h) { return h->last_error.c_str(); }

}  // extern "C"