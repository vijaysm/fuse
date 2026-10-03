#ifndef FUSE_EXAMPLES_BRUSSELATOR_PARTICIPANTS_BRUSSELATOR_CHECK_CORE_H_
#define FUSE_EXAMPLES_BRUSSELATOR_PARTICIPANTS_BRUSSELATOR_CHECK_CORE_H_

/// Time windows track (Conformance, Unit 2, Q3): the assembler that reads
/// u_main.c's and v_main.f90's own per-window log files and recomputes
/// `examples/brusselator/main.cpp`'s own `balance_err`/`exchange_err`/
/// `max_window_balance_error`/`global_balance_error`/
/// `max_window_exchange_error` from them -- bit for bit the SAME formulas
/// `main.cpp`'s own `--check` block uses (that file's own lines around
/// `balance_err = std::abs((mean_u + mean_v - mu0 - mv0) - d_external)`),
/// just reconstructed from two independent processes' printed logs
/// instead of two live `UComponent`/`VComponent` pointers in one process.
///
/// LOG FORMAT (one line per window, lowest window index first; hexfloat
/// via `%a`/`%la`, exact round trip through `strtod`/`scanf("%la", ...)`
/// -- no decimal rounding anywhere in this tool's own arithmetic):
///   U's log:  "window=<k> mean_u=<hex> external_total=<hex> exchange_out=<hex>\n"
///   V's log:  "window=<k> mean_v=<hex> exchange_in=<hex>\n"
/// Row `window=0` is the BASELINE -- the state before any window has
/// advanced (`external_total`/`exchange_out`/`exchange_in` all zero at
/// construction, matching `UCore`'s/`VDiffusionCore`'s own initial
/// values), exactly what `main.cpp`'s own `eu0`/`xo0`/`xi0`/`mu0`/`mv0`
/// snapshots play for ONE process's own window loop; row `window=k` for
/// k >= 1 is the state after that window's own last sweep converged. The
/// two logs must report the SAME number of windows, in order.
///
/// Separated from brusselator_check.cpp's own `main()` (CLI argument
/// parsing, usage text) so a test can call `run_brusselator_check()`
/// directly on logs it just wrote, without forking a subprocess --
/// shim.h's and loop_step.h's own split between physics/ABI logic and
/// whatever drives them follows the same pattern.

#include <optional>
#include <string>
#include <vector>

namespace brusselator_check {

struct Verdict {
  bool ok = false;
  std::string error;  // non-empty only when ok == false AND the failure is
                       // structural (log read/parse/length-mismatch), not
                       // a threshold miss -- see `failed_thresholds`.
  std::vector<std::string> failed_thresholds;  // one entry per threshold
                                               // this run exceeded, empty
                                               // when every threshold held
                                               // (mirrors main.cpp's own
                                               // three independent
                                               // --check conditions).

  int windows = 0;
  double final_mean_u = 0.0, final_mean_v = 0.0;
  double max_window_balance_error = 0.0;
  double global_balance_error = 0.0;
  double max_window_exchange_error = 0.0;

  [[nodiscard]] bool passed() const { return ok && failed_thresholds.empty(); }
};

/// Same thresholds main.cpp's own --check block uses (that file's own
/// comment on them has the measured 1x/2x/4x refinement history):
/// max_window_balance_error <= 1e-11, |global_balance_error| <= 1e-9,
/// max_window_exchange_error <= 1e-9.
[[nodiscard]] Verdict run_brusselator_check(const std::string& u_log_path,
                                            const std::string& v_log_path);

/// `main.cpp`'s own summary line format (that file's own `std::printf`
/// call just above its `--check` block), reproduced so a host comparing
/// this tool's stdout against the in-process example's own output sees
/// the identical text.
void print_summary(const Verdict& v);

}  // namespace brusselator_check

#endif  // FUSE_EXAMPLES_BRUSSELATOR_PARTICIPANTS_BRUSSELATOR_CHECK_CORE_H_
