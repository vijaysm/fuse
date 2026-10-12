#!/usr/bin/env bash
# tools/probe-mpi-appnum.sh -- the acceptance test for the "mpmd" launch mode
# (KNOWN_ISSUES.md D34, docs/user/multiple-executables.md).
#
# Builds tools/probe-mpi-appnum.c and launches it as a two-segment MPMD job
# (colon syntax), one rank each, telling each segment via argv which
# MPI_APPNUM it should see. fuse::transport::check_mpmd_layout() depends on
# MPI_APPNUM differing per segment to validate an mpmd launch's layout; some
# launchers (this host's Open MPI 4.1.5, measured) report appnum 0 for every
# rank regardless of segment.
#
# Exit 0  both segments reported the MPI_APPNUM they were told to expect.
# Exit 1  they did not -- measured. The caller should disable the mpmd launch
#         mode, not fail the build: the accept/connect mode is unaffected,
#         and is probed separately by tools/probe-mpi-dynamic.sh.
# Exit 2  the probe could not be built or run for a reason unrelated to the
#         capability (no mpicc, no mpiexec, a compile error, or a launcher
#         that cannot start even a single-segment job here). Distinguished
#         from exit 1 so a caller never mistakes "untested" for "works" --
#         nor for "broken", which would disable a capability that may be fine.
#
# Usage: tools/probe-mpi-appnum.sh
# Environment:
#   FUSE_PROBE_MPICC      compiler wrapper (default: mpicc)
#   FUSE_PROBE_MPIEXEC    launcher        (default: mpiexec)
#   FUSE_PROBE_BUILD_DIR  scratch         (default: a mktemp -d)

set -u

MPICC="${FUSE_PROBE_MPICC:-mpicc}"
MPIEXEC="${FUSE_PROBE_MPIEXEC:-mpiexec}"
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/probe-mpi-appnum.c"

if [ ! -f "$SRC" ]; then
  echo "probe-mpi-appnum: $SRC is missing" >&2
  exit 2
fi
command -v "$MPICC"   >/dev/null 2>&1 || { echo "probe-mpi-appnum: no $MPICC in PATH" >&2;   exit 2; }
command -v "$MPIEXEC" >/dev/null 2>&1 || { echo "probe-mpi-appnum: no $MPIEXEC in PATH" >&2; exit 2; }

BUILD="${FUSE_PROBE_BUILD_DIR:-}"
OWN_BUILD=0
if [ -z "$BUILD" ]; then
  BUILD="$(mktemp -d 2>/dev/null)" || { echo "probe-mpi-appnum: mktemp failed" >&2; exit 2; }
  OWN_BUILD=1
fi
mkdir -p "$BUILD" || { echo "probe-mpi-appnum: cannot create $BUILD" >&2; exit 2; }

BIN="$BUILD/probe-mpi-appnum"

if ! "$MPICC" -o "$BIN" "$SRC" > "$BUILD/compile.log" 2>&1; then
  echo "probe-mpi-appnum: could not compile the probe:" >&2
  cat "$BUILD/compile.log" >&2
  [ "$OWN_BUILD" = 1 ] && rm -rf "$BUILD"
  exit 2
fi

# Can this launcher start an MPI process at all? If not, nothing below
# measures the capability -- it measures the sandbox -- and the verdict must
# be "could not test", exactly probe-mpi-dynamic.sh's own selftest step.
if ! "$MPIEXEC" -n 1 "$BIN" 0 > "$BUILD/selftest.log" 2>&1; then
  echo "probe-mpi-appnum: $MPIEXEC cannot launch an MPI process here, so the" >&2
  echo "mpmd appnum capability was NOT tested:" >&2
  sed 's/^/    /' "$BUILD/selftest.log" >&2
  [ "$OWN_BUILD" = 1 ] && rm -rf "$BUILD"
  exit 2
fi

# The real measurement: two colon-syntax segments, each told what appnum it
# should see. A launcher whose mpiexec does not understand MPMD colon syntax
# at all also fails here with a non-zero exit, which still correctly disables
# a mode that cannot be used on this launcher either way.
LOG="$BUILD/run.log"
if ! "$MPIEXEC" -n 1 "$BIN" 0 : -n 1 "$BIN" 1 > "$LOG" 2>&1; then
  echo "probe-mpi-appnum: this MPI does not report a distinct MPI_APPNUM per MPMD segment:"
  sed 's/^/    /' "$LOG"
  [ "$OWN_BUILD" = 1 ] && rm -rf "$BUILD"
  exit 1
fi

echo "probe-mpi-appnum: MPI_APPNUM is distinct per MPMD segment"
sed 's/^/    /' "$LOG"
[ "$OWN_BUILD" = 1 ] && rm -rf "$BUILD"
exit 0
