#!/usr/bin/env bash
# examples/demo_b/run_demo.sh -- launches the producer and consumer as two
# SEPARATE mpiexec invocations (not one MPMD line: the whole point of the
# rendezvous in rendezvous.hpp is that the two executables do not need to be
# launched together), waits for both, and reports the combined result.
#
# Registered as the ctest row example.demo_b (examples/CMakeLists.txt).
#
# Usage: run_demo.sh <build-dir> [port_file] [hdf5_file] [dataset_name]

set -u -o pipefail

[ $# -ge 1 ] || { echo "usage: $0 <build-dir> [port_file] [hdf5_file] [dataset_name]" >&2; exit 2; }
BUILD_DIR=$1
PORT_FILE=${2:-$(mktemp -u)/demo_b_port}
HDF5_FILE=${3:-$(mktemp -u)/demo_b_producer_out.h5}
DATASET=${4:-/grid/values}
MPIEXEC=${MPIEXEC:-mpiexec}   # the ctest row passes the configured launcher

PRODUCER="$BUILD_DIR/examples/demo_b_producer"
CONSUMER="$BUILD_DIR/examples/demo_b_consumer"
for exe in "$PRODUCER" "$CONSUMER"; do
  [ -x "$exe" ] || { echo "run_demo.sh: '$exe' not found or not executable; build with -DFUSE_ENABLE_LOWFIVE=ON -DFUSE_ENABLE_MPI=ON -DFUSE_ENABLE_MOAB=ON first" >&2; exit 2; }
done

mkdir -p "$(dirname "$PORT_FILE")" "$(dirname "$HDF5_FILE")"
# Per-run output files next to the port file, so concurrent runs cannot
# overwrite each other (fixed /tmp names before 2026-10-14).
OUT_DIR=$(dirname "$PORT_FILE")
# A file left by an earlier run that fell back to real I/O would make the
# consumer's "nothing reached disk" check fail for a reason unrelated to
# this run; both are this script's own scratch files.
rm -f "$PORT_FILE" "$HDF5_FILE"

"$MPIEXEC" -n 1 "$PRODUCER" "$PORT_FILE" "$HDF5_FILE" "$DATASET" > "$OUT_DIR/demo_b_producer.out" 2>&1 &
PRODUCER_PID=$!

"$MPIEXEC" -n 1 "$CONSUMER" "$PORT_FILE" "$HDF5_FILE" "$DATASET" > "$OUT_DIR/demo_b_consumer.out" 2>&1 &
CONSUMER_PID=$!

PRODUCER_STATUS=0
CONSUMER_STATUS=0
wait "$PRODUCER_PID" || PRODUCER_STATUS=$?
wait "$CONSUMER_PID" || CONSUMER_STATUS=$?

echo "--- producer output ---"
cat "$OUT_DIR/demo_b_producer.out"
echo "--- consumer output ---"
cat "$OUT_DIR/demo_b_consumer.out"

rm -f "$PORT_FILE"

if [ "$PRODUCER_STATUS" -eq 0 ] && [ "$CONSUMER_STATUS" -eq 0 ]; then
  echo "run_demo.sh: PASS"
  exit 0
fi
echo "run_demo.sh: FAIL (producer exit=$PRODUCER_STATUS, consumer exit=$CONSUMER_STATUS)" >&2
exit 1
