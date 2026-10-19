#!/usr/bin/env bash
# KNOWN_ISSUES.md D18, round 2. The host-run driver behind
# parallel.connect.chain.{3,4} -- run_pair.sh's own "accept" mode,
# generalised to 3 or 4 separately launched executables instead of 2.
#
# Every link is its own mpiexec invocation (one rank each -- see
# connect_chain_common.hpp's own header comment for why this keeps the
# golden comparison reusing golden-1+1.txt regardless of chain length).
# Step 1 (a/b) is the SAME acceptor/connector handshake run_pair.sh's own
# "accept" mode uses, over port_ab; each later link (c, then d) is the
# acceptor for ITS OWN step, publishing its own port file, with the
# already-launched chain so far acting as the connector
# (fuse::transport::connect_chain()'s own account, src/transport/
# connect.cpp). No fixed sleep: every link is launched together and each
# later link's own connect() call polls for its predecessor's port file
# up to --timeout seconds (connect.hpp's own wait_for_port_file()).
#
# Usage (fixed positional arguments -- <d_exe> is always present, simply
# unused and never launched when count is 3):
#   run_chain.sh <count: 3|4> <mpiexec> \
#                <chain_a_exe> <chain_b_exe> <chain_c_exe> <chain_d_exe> \
#                <golden_file> <scratch_dir> [timeout_s]
#
# KNOWN_ISSUES.md D18, round 4 (the lead's own instruction): a 2026-10-19
# host re-run with FUSE_CONNECT_CHAIN_TRACE=1 set printed NOTHING in
# a.log/b.log/c.log. Unlike test/parallel/brusselator_participants/
# run_participants.sh's own FUSE_LOWFIVE_ATTACH_TRACE (which is never
# forwarded by this project's own CMake registrations at all -- a plain
# manually-set shell variable, relying entirely on ctest's own default
# environment inheritance through this script down to mpiexec's launched
# ranks), this script now forwards FUSE_CONNECT_CHAIN_TRACE EXPLICITLY,
# via `-genv`, to every rank it launches -- not because inheritance is
# known to be broken here (this sandbox cannot run mpiexec at all, C1, so
# that could not be measured), but because `-genv` removes the ambient-
# inheritance question entirely as a possible cause, and the diagnostic
# line below SAYS, in the ctest row's own captured output (not a.log/b.
# log/c.log, which are per-rank, post-launch), whether this script's own
# shell ever saw the variable at all -- the other candidate the task
# named ("the failure happens before the first traced step") is still
# possible and this script cannot rule it out; only the trace lines
# themselves (now definitely reaching every rank's own environment) can.
set -u

COUNT="$1"; MPIEXEC="$2"
A_EXE="$3"; B_EXE="$4"; C_EXE="$5"; D_EXE="$6"
GOLDEN="$7"; SCRATCH="$8"; TIMEOUT="${9:-60}"

mkdir -p "$SCRATCH"
PORT_AB="$SCRATCH/port_ab.txt"
PORT_C="$SCRATCH/port_c.txt"
PORT_D="$SCRATCH/port_d.txt"
rm -f "$PORT_AB" "$PORT_C" "$PORT_D"
A_LOG="$SCRATCH/a.log"; B_LOG="$SCRATCH/b.log"; C_LOG="$SCRATCH/c.log"; D_LOG="$SCRATCH/d.log"
rm -f "$A_LOG" "$B_LOG" "$C_LOG" "$D_LOG"

COMMON_ARGS="--count $COUNT --port-ab $PORT_AB --port-c $PORT_C --port-d $PORT_D --timeout $TIMEOUT"

# This script's own stderr (unlike a.log/b.log/c.log) is captured by
# ctest itself -- visible via `ctest -V`/`--output-on-failure` -- so this
# line answers "did FUSE_CONNECT_CHAIN_TRACE reach this script's own
# shell at all" independent of whatever each participant's own log shows.
GENV_ARGS=()
if [ -n "${FUSE_CONNECT_CHAIN_TRACE:-}" ]; then
  GENV_ARGS=(-genv FUSE_CONNECT_CHAIN_TRACE "$FUSE_CONNECT_CHAIN_TRACE")
  echo "run_chain.sh: FUSE_CONNECT_CHAIN_TRACE='$FUSE_CONNECT_CHAIN_TRACE' in this script's own environment -- forwarding to every rank via mpiexec -genv" >&2
else
  echo "run_chain.sh: FUSE_CONNECT_CHAIN_TRACE is NOT set in this script's own environment" >&2
fi

RC=0
"$MPIEXEC" "${GENV_ARGS[@]}" -n 1 "$A_EXE" $COMMON_ARGS > "$A_LOG" 2>&1 &
PID_A=$!
"$MPIEXEC" "${GENV_ARGS[@]}" -n 1 "$B_EXE" $COMMON_ARGS > "$B_LOG" 2>&1 &
PID_B=$!
"$MPIEXEC" "${GENV_ARGS[@]}" -n 1 "$C_EXE" $COMMON_ARGS > "$C_LOG" 2>&1 &
PID_C=$!
PID_D=""
if [ "$COUNT" -ge 4 ]; then
  "$MPIEXEC" "${GENV_ARGS[@]}" -n 1 "$D_EXE" $COMMON_ARGS > "$D_LOG" 2>&1 &
  PID_D=$!
fi

wait "$PID_A" || RC=1
wait "$PID_B" || RC=1
wait "$PID_C" || RC=1
if [ -n "$PID_D" ]; then
  wait "$PID_D" || RC=1
fi

echo "--- a.log ---"; cat "$A_LOG" 2>/dev/null
echo "--- b.log ---"; cat "$B_LOG" 2>/dev/null
echo "--- c.log ---"; cat "$C_LOG" 2>/dev/null
if [ "$COUNT" -ge 4 ]; then
  echo "--- d.log ---"; cat "$D_LOG" 2>/dev/null
  LAST_LOG="$D_LOG"
else
  LAST_LOG="$C_LOG"
fi

if [ "$RC" -ne 0 ]; then
  echo "run_chain.sh: a launch exited non-zero" >&2
  exit 1
fi

grep -E '^rank=' "$LAST_LOG" | sort > "$SCRATCH/actual.txt"
sort "$GOLDEN" > "$SCRATCH/expected.txt"
if ! diff -u "$SCRATCH/expected.txt" "$SCRATCH/actual.txt"; then
  echo "run_chain.sh: delivered values do not match the in-process golden" >&2
  exit 1
fi

echo "run_chain.sh: OK ($(wc -l < "$SCRATCH/actual.txt") value(s) matched)"
exit 0
