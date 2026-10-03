#!/usr/bin/env bash
# Conformance, Unit 2: the host-run driver behind
# verification.brusselator.participants.{mpmd,accept}.{3,5}. MPI cannot
# start in this sandbox at all (KNOWN_ISSUES.md C1), so every row this
# drives is a host_verification_needed item -- what IS verified here is
# that brusselator_graph_tool, brusselator_u_main, brusselator_v_main,
# brusselator_coupler_main and brusselator_check all BUILD at tier 1
# (examples/CMakeLists.txt). test/parallel/connect/run_pair.sh is the
# precedent this mirrors.
#
# Three steps, every mode: (1) brusselator_graph_tool writes the graph
# YAML for this row's own rank/launch layout; (2) the participant
# executables run against it; (3) brusselator_check reads both sides'
# logs and prints the SAME summary line examples/brusselator/main.cpp's
# own in-process --check prints, exiting 0 only on a pass.
#
# mpmd: ONE mpiexec, three ':'-separated app segments (U, V, coupler),
# appnum 0/1/2 in that order -- brusselator_graph_tool's own --launch
# mpmd branch lays out ranks to match exactly.
#
# accept: TWO mpiexec invocations. U's executable is the Connect-mode
# acceptor AND carries the coupler's own bundled ranks in its local
# MPI_COMM_WORLD (Q1's ruling, KNOWN_ISSUES.md D18) -- it is therefore
# started FIRST and backgrounded (it blocks in MPI_Comm_accept until V's
# executable, the connector, publishes nothing and instead waits for
# the acceptor's own published port file -- connect.cpp's own
# connect_pairwise() comment has the exact handshake), then V's
# executable, then both are waited on. `--ranks-u` tells brusselator_
# u_main how many of ITS OWN local ranks are real U-physics ranks; any
# remaining local ranks are the bundled coupler's own placeholder ranks
# (u_main.c's own header comment has the full account, including the
# one mechanical question not yet host-verified).
#
# Usage:
#   run_participants.sh <mode: mpmd|accept> <mpiexec> \
#     <ranks_u> <ranks_v> <ranks_coupler> \
#     <graph_tool_exe> <u_main_exe> <v_main_exe> <coupler_main_exe> \
#     <check_exe> <scratch_dir> [periods] [timeout_s]
set -u

MODE="$1"; MPIEXEC="$2"
RANKS_U="$3"; RANKS_V="$4"; RANKS_COUPLER="$5"
GRAPH_TOOL="$6"; U_MAIN="$7"; V_MAIN="$8"; COUPLER_MAIN="$9"
shift 9
CHECK="$1"; SCRATCH="$2"; PERIODS="${3:-2}"; TIMEOUT="${4:-60}"

mkdir -p "$SCRATCH"
GRAPH="$SCRATCH/graph.yaml"
PORT_FILE="$SCRATCH/port.txt"
U_LOG="$SCRATCH/u.log"
V_LOG="$SCRATCH/v.log"
rm -f "$GRAPH" "$PORT_FILE" "$U_LOG" "$V_LOG"

"$GRAPH_TOOL" --ranks-u "$RANKS_U" --ranks-v "$RANKS_V" --ranks-coupler "$RANKS_COUPLER" \
    --launch "$MODE" --out "$GRAPH" --port-file "$PORT_FILE" || exit 1

RC=0
if [ "$MODE" = "mpmd" ]; then
  "$MPIEXEC" -n "$RANKS_U" "$U_MAIN" --graph "$GRAPH" --log "$U_LOG" --periods "$PERIODS" \
      : -n "$RANKS_V" "$V_MAIN" --graph "$GRAPH" --log "$V_LOG" --periods "$PERIODS" \
      : -n "$RANKS_COUPLER" "$COUPLER_MAIN" --graph "$GRAPH" \
      > "$SCRATCH/combined.log" 2>&1
  RC=$?
elif [ "$MODE" = "accept" ]; then
  U_WORLD=$((RANKS_U + RANKS_COUPLER))
  "$MPIEXEC" -n "$U_WORLD" "$U_MAIN" --graph "$GRAPH" --log "$U_LOG" \
      --ranks-u "$RANKS_U" --periods "$PERIODS" > "$SCRATCH/u_stdout.log" 2>&1 &
  PID_U=$!
  "$MPIEXEC" -n "$RANKS_V" "$V_MAIN" --graph "$GRAPH" --log "$V_LOG" \
      --periods "$PERIODS" > "$SCRATCH/v_stdout.log" 2>&1 &
  PID_V=$!
  wait "$PID_U" || RC=1
  wait "$PID_V" || RC=1
else
  echo "run_participants.sh: unknown mode '$MODE' (want mpmd|accept)" >&2
  exit 2
fi

if [ "$RC" -ne 0 ]; then
  echo "run_participants.sh: participant run exited $RC" >&2
  exit 1
fi

"$CHECK" --u-log "$U_LOG" --v-log "$V_LOG"
exit $?
