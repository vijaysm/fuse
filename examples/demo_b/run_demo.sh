#!/usr/bin/env bash
# examples/demo_b/run_demo.sh -- launches the producer and consumer as two
# SEPARATE mpiexec invocations (not one MPMD line: the whole point of the
# rendezvous in rendezvous.hpp is that the two executables do not need to be
# launched together), waits for both, and reports the combined result.
#
# UNEXECUTED in this checkout -- see docs/user/lowfive.md. This is the host
# command this track's report names under host_verification_needed.
#
# Usage: run_demo.sh <build-dir> [port_file] [hdf5_file] [dataset_name]

set -u -o pipefail

[ $# -ge 1 ] || { echo "usage: $0 <build-dir> [port_file] [hdf5_file] [dataset_name]" >&2; exit 2; }
BUILD_DIR=$1
PORT_FILE=${2:-$(mktemp -u)/demo_b_port}
HDF5_FILE=${3:-$(mktemp -u)/demo_b_producer_out.h5}
DATASET=${4:-/grid/values}

PRODUCER="$BUILD_DIR/examples/demo_b_producer"
CONSUMER="$BUILD_DIR/examples/demo_b_consumer"
for exe in "$PRODUCER" "$CONSUMER"; do
  [ -x "$exe" ] || { echo "run_demo.sh: '$exe' not found or not executable; build with -DFUSE_ENABLE_LOWFIVE=ON -DFUSE_ENABLE_MPI=ON -DFUSE_ENABLE_MOAB=ON first" >&2; exit 2; }
done

mkdir -p "$(dirname "$PORT_FILE")" "$(dirname "$HDF5_FILE")"
rm -f "$PORT_FILE"

mpiexec -n 1 "$PRODUCER" "$PORT_FILE" "$HDF5_FILE" "$DATASET" > /tmp/demo_b_producer.out 2>&1 &
PRODUCER_PID=$!

mpiexec -n 1 "$CONSUMER" "$PORT_FILE" "$HDF5_FILE" "$DATASET" > /tmp/demo_b_consumer.out 2>&1 &
CONSUMER_PID=$!

PRODUCER_STATUS=0
CONSUMER_STATUS=0
wait "$PRODUCER_PID" || PRODUCER_STATUS=$?
wait "$CONSUMER_PID" || CONSUMER_STATUS=$?

echo "--- producer output ---"
cat /tmp/demo_b_producer.out
echo "--- consumer output ---"
cat /tmp/demo_b_consumer.out

rm -f "$PORT_FILE"

if [ "$PRODUCER_STATUS" -eq 0 ] && [ "$CONSUMER_STATUS" -eq 0 ]; then
  echo "run_demo.sh: PASS"
  exit 0
fi
echo "run_demo.sh: FAIL (producer exit=$PRODUCER_STATUS, consumer exit=$CONSUMER_STATUS)" >&2
exit 1
