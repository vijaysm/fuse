#!/usr/bin/env bash
# tools/probe-mpi-dynamic.sh -- the acceptance test for the accept/connect
# launch mode (KNOWN_ISSUES.md D23, docs/user/multiple-executables.md).
#
# Builds tools/probe-mpi-dynamic.c once and launches it as TWO separate jobs
# PER LAYOUT, repeated several times per layout -- the shape a coupled run
# uses, and the shape whose intermittency a single 1-rank-a-side run cannot
# see. KNOWN_ISSUES.md D23's own account: the configure-time probe measured
# "working" at one rank a side on an MPI whose multi-rank accept rows then
# timed out six of six times in the next full suite. A probe that only ever
# tries 1+1 is not a measurement of a 3+2 row; this one tries every layout a
# registered row actually uses.
#
# Usage: tools/probe-mpi-dynamic.sh <layout> [layout ...]
#   Each <layout> is "NA+NB": NA ranks on the accept side, NB on the connect
#   side (e.g. "1+1", "2+3"). At least one layout is required; if none is
#   given, "1+1" is assumed (the single legacy case). The FIRST layout given
#   is the one cmake/FuseDependencies.cmake reads back for the single
#   overall verdict: FUSE_ENABLE_CONNECT_ACCEPT keeps following "1+1" alone
#   (KNOWN_ISSUES.md D23's final account, "nothing downstream changes").
#
# Prints one line per layout to STDOUT, in the exact form the caller parses
# (cmake/FuseMpiDynamic.cmake's fuse_mpi_dynamic_parse_probe_output()):
#     LAYOUT <na>+<nb> PASSED <p> OF <n>
# Everything else -- compiler/launcher diagnostics, a failing run's own
# accept/connect logs -- goes to STDERR, for a human reading the configure
# log, not for the caller to parse.
#
# Exit 0  every requested layout was actually tried (whatever its own pass
#         count came out to be -- THAT count is the measurement, not this
#         exit code; a caller wanting a capability verdict reads the FIRST
#         layout's own "PASSED p OF n" line, not this exit status).
# Exit 2  the probe could not be built or launched at all, for a reason
#         unrelated to the capability (no mpicc, no mpiexec, a compile
#         error, or this launcher cannot start any MPI process here). No
#         layout was tried; nothing below is a measurement of anything.
#
# Environment:
#   FUSE_PROBE_MPICC      compiler wrapper (default: mpicc)
#   FUSE_PROBE_MPIEXEC    launcher        (default: mpiexec)
#   FUSE_PROBE_BUILD_DIR  scratch         (default: a mktemp -d)
#   FUSE_PROBE_RUNS       repeats of EACH layout (default: 6)
#   FUSE_PROBE_TIMEOUT    seconds one run of one layout is allowed before it
#                         counts as failed (default: 20)

set -u

MPICC="${FUSE_PROBE_MPICC:-mpicc}"
MPIEXEC="${FUSE_PROBE_MPIEXEC:-mpiexec}"
RUNS="${FUSE_PROBE_RUNS:-6}"
TIMEOUT="${FUSE_PROBE_TIMEOUT:-20}"
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/probe-mpi-dynamic.c"

if [ "$#" -lt 1 ]; then
  set -- "1+1"
fi

if [ ! -f "$SRC" ]; then
  echo "probe-mpi-dynamic: $SRC is missing" >&2
  exit 2
fi
command -v "$MPICC"   >/dev/null 2>&1 || { echo "probe-mpi-dynamic: no $MPICC in PATH" >&2;   exit 2; }
command -v "$MPIEXEC" >/dev/null 2>&1 || { echo "probe-mpi-dynamic: no $MPIEXEC in PATH" >&2; exit 2; }

BUILD="${FUSE_PROBE_BUILD_DIR:-}"
OWN_BUILD=0
if [ -z "$BUILD" ]; then
  BUILD="$(mktemp -d 2>/dev/null)" || { echo "probe-mpi-dynamic: mktemp failed" >&2; exit 2; }
  OWN_BUILD=1
fi
mkdir -p "$BUILD" || { echo "probe-mpi-dynamic: cannot create $BUILD" >&2; exit 2; }

BIN="$BUILD/probe-mpi-dynamic"

if ! "$MPICC" -o "$BIN" "$SRC" > "$BUILD/compile.log" 2>&1; then
  echo "probe-mpi-dynamic: could not compile the probe:" >&2
  cat "$BUILD/compile.log" >&2
  [ "$OWN_BUILD" = 1 ] && rm -rf "$BUILD"
  exit 2
fi

# Can this launcher start an MPI process at all? If not, nothing below
# measures the capability -- it measures the sandbox -- and every layout
# must be reported "could not test", not silently skipped or marked broken.
if ! "$MPIEXEC" -n 1 "$BIN" selftest > "$BUILD/selftest.log" 2>&1; then
  echo "probe-mpi-dynamic: $MPIEXEC cannot launch an MPI process here, so the" >&2
  echo "accept/connect capability was NOT tested:" >&2
  sed 's/^/    /' "$BUILD/selftest.log" >&2
  [ "$OWN_BUILD" = 1 ] && rm -rf "$BUILD"
  exit 2
fi

OVERALL_RC=0
for LAYOUT in "$@"; do
  case "$LAYOUT" in
    [0-9]*+[0-9]*) ;;
    *)
      echo "probe-mpi-dynamic: '$LAYOUT' is not of the form NA+NB; skipped" >&2
      OVERALL_RC=2
      continue
      ;;
  esac
  NA="${LAYOUT%%+*}"
  NB="${LAYOUT##*+}"
  PASSED=0
  RUN=1
  while [ "$RUN" -le "$RUNS" ]; do
    PORT="$BUILD/port-${LAYOUT}-${RUN}.txt"
    A_LOG="$BUILD/accept-${LAYOUT}-${RUN}.log"
    B_LOG="$BUILD/connect-${LAYOUT}-${RUN}.log"
    rm -f "$PORT" "$A_LOG" "$B_LOG"

    # The acceptor first, in the background; it writes the port file the
    # connector waits for. Same two-separate-mpiexec-invocations shape as
    # the original single-layout probe, now with NA/NB ranks each.
    "$MPIEXEC" -n "$NA" "$BIN" accept  "$PORT" > "$A_LOG" 2>&1 &
    A_PID=$!
    "$MPIEXEC" -n "$NB" "$BIN" connect "$PORT" > "$B_LOG" 2>&1 &
    B_PID=$!

    # Bounded wait without relying on `timeout`, which is not present by
    # default on macOS. Poll both until each has exited or the budget for
    # THIS run runs out.
    A_RC=""; B_RC=""
    waited=0
    while [ "$waited" -lt "$TIMEOUT" ]; do
      if [ -z "$A_RC" ] && ! kill -0 "$A_PID" 2>/dev/null; then wait "$A_PID"; A_RC=$?; fi
      if [ -z "$B_RC" ] && ! kill -0 "$B_PID" 2>/dev/null; then wait "$B_PID"; B_RC=$?; fi
      [ -n "$A_RC" ] && [ -n "$B_RC" ] && break
      sleep 1
      waited=$((waited + 1))
    done

    TIMED_OUT=0
    if [ -z "$A_RC" ] || [ -z "$B_RC" ]; then
      TIMED_OUT=1
      kill -9 "$A_PID" "$B_PID" 2>/dev/null
      wait "$A_PID" 2>/dev/null; wait "$B_PID" 2>/dev/null
      [ -z "$A_RC" ] && A_RC=124
      [ -z "$B_RC" ] && B_RC=124
    fi

    if [ "$A_RC" -eq 0 ] && [ "$B_RC" -eq 0 ]; then
      PASSED=$((PASSED + 1))
    else
      if [ "$TIMED_OUT" = 1 ]; then
        echo "probe-mpi-dynamic: layout ${LAYOUT} run ${RUN}/${RUNS} did not finish within ${TIMEOUT}s" >&2
      else
        echo "probe-mpi-dynamic: layout ${LAYOUT} run ${RUN}/${RUNS}: accept exited $A_RC, connect exited $B_RC" >&2
      fi
      echo "--- accepting job (${NA} ranks) ---" >&2
      sed 's/^/    /' "$A_LOG" >&2 2>/dev/null
      echo "--- connecting job (${NB} ranks) ---" >&2
      sed 's/^/    /' "$B_LOG" >&2 2>/dev/null
    fi
    RUN=$((RUN + 1))
  done
  echo "LAYOUT ${LAYOUT} PASSED ${PASSED} OF ${RUNS}"
done

[ "$OWN_BUILD" = 1 ] && rm -rf "$BUILD"
exit "$OVERALL_RC"
