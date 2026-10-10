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
#     <check_exe> <scratch_dir> [periods] [timeout_s] [transport] [solver]
#
# [transport] (S6 phase 3, "LowFive" track, Unit 3b): "fuse" (default) or
# "lowfive", passed through verbatim as --transport to graph_tool/u_main/
# v_main (brusselator_graph_tool.cpp's own --transport flag decides
# whether u_out/v_out get an hdf5_path at all; u_main.c's/v_main.f90's
# own --transport decides whether their loop bodies perform the real
# H5Dwrite/H5Dread this track adds). The VOL connector that makes LowFive
# interception apply at all is NOT this script's concern -- it is
# attached process-wide via HDF5_VOL_CONNECTOR/HDF5_PLUGIN_PATH, which
# the CALLER (examples/CMakeLists.txt's own lowfive ctest rows) sets in
# the test's own ENVIRONMENT, per docs/user/lowfive.md.
#
# [solver] (Physics track): "auto" (default -- direct at 1 rank, schur at
# >1), "direct", "schur", "pcg", or "gmres" (species_core.hpp's own
# SolverKind), passed through verbatim as --solver to u_main/v_main.
set -u

MODE="$1"; MPIEXEC="$2"
RANKS_U="$3"; RANKS_V="$4"; RANKS_COUPLER="$5"
GRAPH_TOOL="$6"; U_MAIN="$7"; V_MAIN="$8"; COUPLER_MAIN="$9"
shift 9
CHECK="$1"; SCRATCH="$2"; PERIODS="${3:-2}"; TIMEOUT="${4:-60}"; TRANSPORT="${5:-fuse}"
SOLVER="${6:-auto}"

mkdir -p "$SCRATCH"
GRAPH="$SCRATCH/graph.yaml"
PORT_FILE="$SCRATCH/port.txt"
U_LOG="$SCRATCH/u.log"
V_LOG="$SCRATCH/v.log"
rm -f "$GRAPH" "$PORT_FILE" "$U_LOG" "$V_LOG"

"$GRAPH_TOOL" --ranks-u "$RANKS_U" --ranks-v "$RANKS_V" --ranks-coupler "$RANKS_COUPLER" \
    --launch "$MODE" --out "$GRAPH" --port-file "$PORT_FILE" --transport "$TRANSPORT" || exit 1

RC=0
if [ "$MODE" = "mpmd" ]; then
  # Each app segment redirects its OWN stdout and stderr, through a shell
  # per segment, rather than all three sharing one redirect of the
  # launcher's. A single mpiexec gives one merged stream in which nothing
  # says which program spoke, and three programs failing at once is the
  # normal case here -- one of them is the cause and the other two are
  # reacting to a dead peer. Diagnosing the 2026-10-08 MPMD failure from
  # `combined.log` was not possible for exactly this reason: a Kokkos
  # error and a SIGABRT, from an unknown pair of the three. `combined.log`
  # is still written for the launcher's own messages (the BAD TERMINATION
  # report, which belongs to no segment).
  # ...and per RANK within a segment, since a segment may have several
  # (the 5-rank row runs U and V on two each) and they would otherwise
  # interleave into one file. PMI_RANK is the launcher's own, so the
  # suffix is the world rank; absent one, 0.
  SEG='f="$1"; shift; exec "$@" > "${f}.${PMI_RANK:-0}.log" 2>&1'
  "$MPIEXEC" -n "$RANKS_U" sh -c "$SEG" seg "$SCRATCH/u_stdout" \
      "$U_MAIN" --graph "$GRAPH" --log "$U_LOG" --periods "$PERIODS" \
      --transport "$TRANSPORT" --solver "$SOLVER" \
      : -n "$RANKS_V" sh -c "$SEG" seg "$SCRATCH/v_stdout" \
      "$V_MAIN" --graph "$GRAPH" --log "$V_LOG" --periods "$PERIODS" \
      --transport "$TRANSPORT" --ranks-u "$RANKS_U" --ranks-v "$RANKS_V" \
      --solver "$SOLVER" \
      : -n "$RANKS_COUPLER" sh -c "$SEG" seg "$SCRATCH/coupler_stdout" \
      "$COUPLER_MAIN" --graph "$GRAPH" \
      > "$SCRATCH/combined.log" 2>&1
  RC=$?
elif [ "$MODE" = "accept" ]; then
  U_WORLD=$((RANKS_U + RANKS_COUPLER))
  "$MPIEXEC" -n "$U_WORLD" "$U_MAIN" --graph "$GRAPH" --log "$U_LOG" \
      --ranks-u "$RANKS_U" --periods "$PERIODS" --transport "$TRANSPORT" \
      --solver "$SOLVER" \
      > "$SCRATCH/u_stdout.log" 2>&1 &
  PID_U=$!
  "$MPIEXEC" -n "$RANKS_V" "$V_MAIN" --graph "$GRAPH" --log "$V_LOG" \
      --periods "$PERIODS" --transport "$TRANSPORT" --solver "$SOLVER" \
      > "$SCRATCH/v_stdout.log" 2>&1 &
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
