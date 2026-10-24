# Demo B: two unmodified HDF5 programs coupled by interception

`examples/demo_b/` shows FUSE's distributed LowFive transport: a producer and a consumer, each
**plain HDF5 code with no FUSE header and no FUSE call**, exchange one dataset between two separate
processes with no HDF5 data reaching disk. Read it if you want to couple existing HDF5 codes without
modifying them. [](lowfive.md) covers the VOL-connector mechanism and its HDF5 requirements.

## Build and run

Needs MPI and LowFive built with its distributed archive (`liblowfive-dist`):

```sh
cmake --preset tier1 -DFUSE_ENABLE_LOWFIVE=ON -DFUSE_ENABLE_MPI=ON
cmake --build --preset tier1
examples/demo_b/run_demo.sh <build-dir>
```

`run_demo.sh` launches `demo_b_producer` and `demo_b_consumer` as two separate `mpiexec -n 1`
invocations, waits for both, prints each one's output and reports `PASS` or `FAIL`. Optional
arguments, in order: a port file path, an HDF5 file path, a dataset name (default `/grid/values`).
The two executables are built only when `FUSE_ENABLE_LOWFIVE`, `FUSE_ENABLE_MPI` and the
distributed archive are all available; otherwise `examples/CMakeLists.txt` prints why at configure
time and builds neither. The `example.demo_b` test runs the script when accept/connect is enabled
(see [](multiple-executables.md) for `FUSE_ENABLE_CONNECT_ACCEPT`). If MOAB is enabled in the same
configure, see [](lowfive.md) for the HDF5 requirement.

Expected output (exact timings can vary by platform):

```
demo_b_producer: wrote 8 values, served across the intercommunicator
demo_b_consumer: read 8 values, bit-for-bit match, zero bytes of HDF5 on disk
```

`run_demo.sh` prefixes these with `--- producer output ---` and `--- consumer output ---` and ends
with `run_demo.sh: PASS` (exit 0) or `run_demo.sh: FAIL (producer exit=.. consumer exit=..)`
(exit 1).

## What it does

The producer writes a fixed 8-element dataset (`{1, 2, 3, 5, 8, 13, 21, 34}`) with an ordinary
`H5Dwrite`; the consumer reads it with an ordinary `H5Dread`. LowFive's `DistMetadataVOL` intercepts
both calls and moves the data over an MPI intercommunicator the two processes form themselves, so
the file named on each command line is never created.

`producer_hdf5.cpp` and `consumer_hdf5.cpp` are the payload: plain HDF5 open/write/close and
open/read/close, with no mention of FUSE. `producer_main.cpp` and `consumer_main.cpp` are the FUSE
side: each forms the rendezvous, attaches the LowFive transport, calls the unmodified payload
function and tears down:

```{literalinclude} ../../examples/demo_b/producer_main.cpp
:language: cpp
:start-at: "fuse::LowFiveTransport transport;"
:end-at: "r = transport.serve_as_producer();"
```

**Rendezvous.** `examples/demo_b/rendezvous.hpp` is a minimal two-process handshake:
`MPI_Open_port` and `MPI_Comm_accept` on the producer side, `MPI_Comm_connect` on the consumer side,
with the port name exchanged through a file both processes can see. It is not
`fuse::transport::connect()`, which forms the base communicator of a whole coupled run; see
[](multiple-executables.md) for that.

**Pass/fail.** The consumer exits 0 if and only if (a) the values it read match the producer's bit
for bit, and (b) `stat()` on the HDF5 file path fails, meaning no real file reached disk. A file that
does exist is reported as an interception failure.

## Limitations

The example uses one rank per side, one file and one dataset, and one exchange per process. It
needs working MPI dynamic process management between two jobs (see
[](multiple-executables.md#which-launch-modes-this-build-offers)).

## See also

- [](lowfive.md): the LowFive transport and VOL interception
- [](multiple-executables.md): launch modes across separately built executables
- [](examples.md): the full list of examples
