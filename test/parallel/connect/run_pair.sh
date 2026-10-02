#!/usr/bin/env bash
# S6 phase 3, "MPI connect" track: the host-run driver behind
# parallel.connect.mpmd.* and parallel.connect.accept.*.
#
# MPMD mode is one mpiexec invocation with the ':' MPMD separator -- both
# launched in the same job, sharing MPI_COMM_WORLD from the start, so there
# is nothing to background. Accept mode is two SEPARATE mpiexec invocations
# (connect_a as the acceptor, connect_b as the connector) with no shared
# parent communicator at all; connect_a is started in the background first
# (it blocks in MPI_Comm_accept until connect_b publishes a connection, per
# connect.cpp's own connection handshake), then connect_b, then both are waited on.
#
# Either way, stdout from the ranks that play component "b" is the thing
# under test: every line is "rank=<merged rank> i=<local index>
# value=<hexfloat>" (connect_common.hpp's run_participant()), compared
# against the golden connect_golden produced (a pure function of na/nb,
# independent of which launch mode produced the real run -- see that
# tool's own header comment for why the two must agree).
#
# Usage:
#   run_pair.sh <mode: mpmd|accept> <mpiexec> <na> <nb> \
#               <connect_a_exe> <connect_b_exe> <golden_file> <scratch_dir> [timeout_s]
set -u

MODE="$1"; MPIEXEC="$2"; NA="$3"; NB="$4"
A_EXE="$5"; B_EXE="$6"; GOLDEN="$7"; SCRATCH="$8"; TIMEOUT="${9:-60}"

mkdir -p "$SCRATCH"
A_LOG="$SCRATCH/a.log"
B_LOG="$SCRATCH/b.log"
rm -f "$A_LOG" "$B_LOG"

RC=0

if [ "$MODE" = "mpmd" ]; then
  # Single launch, single MPI_COMM_WORLD: appnum 0 is connect_a's segment,
  # appnum 1 is connect_b's, in that command-line order -- the order
  # build_graph() (connect_common.hpp) declares them in, which is what
  # fuse::transport::validate_launch_layout() requires to agree.
  "$MPIEXEC" -n "$NA" "$A_EXE" --launch mpmd --na "$NA" --nb "$NB" \
      : -n "$NB" "$B_EXE" --launch mpmd --na "$NA" --nb "$NB" \
      > "$SCRATCH/combined.log" 2>&1
  RC=$?
  cp "$SCRATCH/combined.log" "$B_LOG"
elif [ "$MODE" = "accept" ]; then
  PORT_FILE="$SCRATCH/port.txt"
  rm -f "$PORT_FILE"
  # connect_a is the acceptor: it must be running (and past MPI_Open_port)
  # before connect_b's own connect() starts polling for the port file, so
  # it is started first and backgrounded -- the whole reason this needs two
  # mpiexec invocations rather than one.
  "$MPIEXEC" -n "$NA" "$A_EXE" --launch accept --na "$NA" --nb "$NB" \
      --port-file "$PORT_FILE" --timeout "$TIMEOUT" > "$A_LOG" 2>&1 &
  PID_A=$!
  "$MPIEXEC" -n "$NB" "$B_EXE" --launch connect --na "$NA" --nb "$NB" \
      --port-file "$PORT_FILE" --timeout "$TIMEOUT" > "$B_LOG" 2>&1 &
  PID_B=$!
  wait "$PID_A" || RC=1
  wait "$PID_B" || RC=1
else
  echo "run_pair.sh: unknown mode '$MODE' (expected mpmd or accept)" >&2
  exit 2
fi

echo "--- a.log ---"
cat "$A_LOG" 2>/dev/null
echo "--- b.log ---"
cat "$B_LOG" 2>/dev/null

if [ "$RC" -ne 0 ]; then
  echo "run_pair.sh: a launch exited non-zero" >&2
  exit 1
fi

grep -E '^rank=' "$B_LOG" | sort > "$SCRATCH/actual.txt"
sort "$GOLDEN" > "$SCRATCH/expected.txt"
if ! diff -u "$SCRATCH/expected.txt" "$SCRATCH/actual.txt"; then
  echo "run_pair.sh: delivered values do not match the in-process golden" >&2
  exit 1
fi

echo "run_pair.sh: OK ($(wc -l < "$SCRATCH/actual.txt") value(s) matched)"
exit 0
