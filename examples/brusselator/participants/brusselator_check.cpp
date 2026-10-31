// Thin CLI wrapper around brusselator_check_core's own
// run_brusselator_check() (the assembler is a small C++ tool, not a shell
// script). Reads u_main.c's and v_main.f90's own per-window log files (format:
// brusselator_check_core.hpp's own header comment) and prints the SAME
// summary line examples/brusselator/main.cpp's own --check block prints,
// exiting 0 on a pass and 1 otherwise (matching that file's own
// convention) -- a host running the {mpmd,accept} ctest rows pipes both
// executables' own log files into this tool as the row's last step.

#include <cstdio>
#include <string>

#include "brusselator_check_core.hpp"

int main(int argc, char** argv) {
  std::string u_log, v_log;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&]() -> std::string {
      return (i + 1 < argc) ? std::string(argv[++i]) : std::string();
    };
    if (arg == "--u-log") {
      u_log = next();
    } else if (arg == "--v-log") {
      v_log = next();
    } else {
      std::fprintf(stderr, "unrecognized argument: %s\n", arg.c_str());
      return 1;
    }
  }
  if (u_log.empty() || v_log.empty()) {
    std::fprintf(stderr, "usage: brusselator_check --u-log path --v-log path\n");
    return 1;
  }

  const brusselator_check::Verdict v = brusselator_check::run_brusselator_check(u_log, v_log);
  brusselator_check::print_summary(v);
  return v.passed() ? 0 : 1;
}
