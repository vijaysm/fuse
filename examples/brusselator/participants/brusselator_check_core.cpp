#include "brusselator_check_core.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>

namespace brusselator_check {
namespace {

constexpr double kMaxWindowBalanceThreshold = 1e-11;
constexpr double kGlobalBalanceThreshold = 1e-9;
constexpr double kMaxWindowExchangeThreshold = 1e-9;

struct URow { int window; double mean_u, external_total, exchange_out; };
struct VRow { int window; double mean_v, exchange_in; };

[[nodiscard]] bool parse_u_line(const std::string& line, URow* out) {
  return std::sscanf(line.c_str(), "window=%d mean_u=%la external_total=%la exchange_out=%la",
                     &out->window, &out->mean_u, &out->external_total, &out->exchange_out) == 4;
}

[[nodiscard]] bool parse_v_line(const std::string& line, VRow* out) {
  return std::sscanf(line.c_str(), "window=%d mean_v=%la exchange_in=%la",
                     &out->window, &out->mean_v, &out->exchange_in) == 3;
}

template <typename Row, typename Parser>
[[nodiscard]] bool read_rows(const std::string& path, Parser parse,
                             std::vector<Row>* out, std::string* error) {
  std::ifstream in(path);
  if (!in) {
    *error = "could not open '" + path + "' for reading";
    return false;
  }
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    Row row{};
    if (!parse(line, &row)) {
      *error = "could not parse line in '" + path + "': " + line;
      return false;
    }
    out->push_back(row);
  }
  return true;
}

}  // namespace

Verdict run_brusselator_check(const std::string& u_log_path, const std::string& v_log_path) {
  Verdict v;

  std::vector<URow> u_rows;
  std::vector<VRow> v_rows;
  if (!read_rows<URow>(u_log_path, parse_u_line, &u_rows, &v.error)) return v;
  if (!read_rows<VRow>(v_log_path, parse_v_line, &v_rows, &v.error)) return v;

  if (u_rows.empty() || v_rows.empty()) {
    v.error = "both logs must have at least the window=0 baseline row";
    return v;
  }
  if (u_rows.size() != v_rows.size()) {
    v.error = "U's log has " + std::to_string(u_rows.size()) + " window(s), V's has " +
             std::to_string(v_rows.size()) + " -- the two sides disagree on how many windows ran";
    return v;
  }
  for (std::size_t i = 0; i < u_rows.size(); ++i) {
    if (u_rows[i].window != static_cast<int>(i) || v_rows[i].window != static_cast<int>(i)) {
      v.error = "window indices are not 0..N-1 in order at row " + std::to_string(i);
      return v;
    }
  }

  const std::size_t n = u_rows.size() - 1;  // number of windows advanced (row 0 is the baseline)
  for (std::size_t k = 1; k <= n; ++k) {
    const double mu0 = u_rows[k - 1].mean_u, mv0 = v_rows[k - 1].mean_v;
    const double d_external = u_rows[k].external_total - u_rows[k - 1].external_total;
    const double d_out = u_rows[k].exchange_out - u_rows[k - 1].exchange_out;
    const double d_in = v_rows[k].exchange_in - v_rows[k - 1].exchange_in;
    const double balance_err =
        std::abs((u_rows[k].mean_u + v_rows[k].mean_v - mu0 - mv0) - d_external);
    const double exchange_err = std::abs(d_out - d_in);
    v.max_window_balance_error = std::max(v.max_window_balance_error, balance_err);
    v.max_window_exchange_error = std::max(v.max_window_exchange_error, exchange_err);
  }

  v.windows = static_cast<int>(n);
  v.final_mean_u = u_rows.back().mean_u;
  v.final_mean_v = v_rows.back().mean_v;
  // main.cpp's own global_balance_error: |final mean_u + final mean_v -
  // initial_mass - pu->external_total()|, with initial_mass = the
  // baseline row's own mean_u + mean_v and external_total's own baseline
  // value folded in the same way the per-window d_external above does.
  v.global_balance_error =
      std::abs((u_rows.back().mean_u + v_rows.back().mean_v) -
               (u_rows.front().mean_u + v_rows.front().mean_v) -
               (u_rows.back().external_total - u_rows.front().external_total));

  if (v.max_window_balance_error > kMaxWindowBalanceThreshold) {
    v.failed_thresholds.push_back("max_window_balance_error " +
                                  std::to_string(v.max_window_balance_error) + " > " +
                                  std::to_string(kMaxWindowBalanceThreshold));
  }
  if (std::abs(v.global_balance_error) > kGlobalBalanceThreshold) {
    v.failed_thresholds.push_back("global_balance_error " +
                                  std::to_string(v.global_balance_error) + " > " +
                                  std::to_string(kGlobalBalanceThreshold));
  }
  if (v.max_window_exchange_error > kMaxWindowExchangeThreshold) {
    v.failed_thresholds.push_back("max_window_exchange_error " +
                                  std::to_string(v.max_window_exchange_error) + " > " +
                                  std::to_string(kMaxWindowExchangeThreshold));
  }

  v.ok = true;
  return v;
}

void print_summary(const Verdict& v) {
  if (!v.ok) {
    std::printf("FAIL(brusselator_check): %s\n", v.error.c_str());
    return;
  }
  std::printf(
      "brusselator_check: %d window(s)\n"
      "  final_mean_u=%.12g final_mean_v=%.12g\n"
      "  max_window_balance_error=%.3e global_balance_error=%.3e max_window_exchange_error=%.3e\n",
      v.windows, v.final_mean_u, v.final_mean_v, v.max_window_balance_error,
      v.global_balance_error, v.max_window_exchange_error);
  if (v.failed_thresholds.empty()) {
    std::printf("  --check: OK\n");
  } else {
    for (const auto& f : v.failed_thresholds) {
      std::printf("FAIL(--check): %s\n", f.c_str());
    }
  }
}

}  // namespace brusselator_check
