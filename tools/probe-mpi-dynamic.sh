#!/usr/bin/env bash
# tools/probe-mpi-dynamic.sh -- the acceptance test for the accept/connect
# launch mode (KNOWN_ISSUES.md D23, docs/user/multiple-executables.md).
#
# Builds tools/probe-mpi-dynamic.c and launches it as TWO separate jobs on
# this machine, one accepting and one connecting, which is the shape a
# coupled run uses and the shape that fails on an MPI whose dynamic process
# management does not work between same-node jobs.
#
# Exit 0  the two jobs connected, merged, and completed a collective over the
#         merged communicator.
# Exit 1  they did not -- measured, with the output of both jobs shown. The
#         caller should disable the accept/connect mode, not fail the build:
#         every other launch mode is unaffected.
# Exit 2  the probe could not be built or run for a reason unrelated to the
#         capability (no mpicc, no mpiexec, a compile error). Distinguished
#         from exit 1 so a caller never mistakes "untested" for "works" --
#         nor for "broken", which would disable a capability that may be fine.
#
# Usage: tools/probe-mpi-dynamic.sh
# Environment:
#   FUSE_PROBE_MPICC      compiler wrapper (default: mpicc)
#   FUSE_PROBE_MPIEXEC    launcher        (default: mpiexec)
#   FUSE_PROBE_BUILD_DIR  scratch         (default: a mktemp -d)
#   FUSE_PROBE_TIMEOUT    seconds to allow the pair (default: 60)

set -u

MPICC="${FUSE_PROBE_MPICC:-mpicc}"
MPIEXEC="${FUSE_PROBE_MPIEXEC:-mpiexec}"
TIMEOUT="${FUSE_PROBE_TIMEOUT:-60}"
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/probe-mpi-dynamic.c"

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
PORT="$BUILD/port.txt"
A_LOG="$BUILD/accept.log"
B_LOG="$BUILD/connect.log"
rm -f "$PORT" "$A_LOG" "$B_LOG"

if ! "$MPICC" -o "$BIN" "$SRC" > "$BUILD/compile.log" 2>&1; then
  echo "probe-mpi-dynamic: could not compile the probe:" >&2
  cat "$BUILD/compile.log" >&2
  [ "$OWN_BUILD" = 1 ] && rm -rf "$BUILD"
  exit 2
fi

# Can this launcher start an MPI process at all? If not, nothing below
# measures the capability -- it measures the sandbox -- and the verdict must
# be "could not test".
if ! "$MPIEXEC" -n 1 "$BIN" selftest > "$BUILD/selftest.log" 2>&1; then
  echo "probe-mpi-dynamic: $MPIEXEC cannot launch an MPI process here, so the" >&2
  echo "accept/connect capability was NOT tested:" >&2
  sed 's/^/    /' "$BUILD/selftest.log" >&2
  [ "$OWN_BUILD" = 1 ] && rm -rf "$BUILD"
  exit 2
fi

# The acceptor first, in the background; it writes the port file the
# connector waits for. Both are single-rank: a device that cannot do this at
# one rank each cannot do it at any size, and a bigger probe would only make
# a configure step slower.
"$MPIEXEC" -n 1 "$BIN" accept "$PORT"  > "$A_LOG" 2>&1 &
A_PID=$!
"$MPIEXEC" -n 1 "$BIN" connect "$PORT" > "$B_LOG" 2>&1 &
B_PID=$!

# Bounded wait without relying on `timeout`, which is not present by default
# on macOS. Poll both until each has exited or the budget runs out.
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
  echo "probe-mpi-dynamic: accept/connect works between two same-node jobs"
  [ "$OWN_BUILD" = 1 ] && rm -rf "$BUILD"
  exit 0
fi

if [ "$TIMED_OUT" = 1 ]; then
  echo "probe-mpi-dynamic: the pair did not finish within ${TIMEOUT}s"
else
  echo "probe-mpi-dynamic: accept exited $A_RC, connect exited $B_RC"
fi
echo "--- accepting job ---";  sed 's/^/    /' "$A_LOG" 2>/dev/null
echo "--- connecting job ---"; sed 's/^/    /' "$B_LOG" 2>/dev/null
[ "$OWN_BUILD" = 1 ] && rm -rf "$BUILD"
exit 1