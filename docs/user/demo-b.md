# Demo B: two unmodified HDF5 programs, coupled by interception alone

`examples/demo_b/` demonstrates FUSE's distributed LowFive transport (mode 4 of the coupling
specification, "no FUSE-specific call on either side" — see [LowFive](lowfive.md) for the mechanism
in full): a producer and a consumer, each **plain HDF5 code with no FUSE header and no FUSE call**,
exchange one dataset across a real OS process boundary with zero bytes of HDF5 ever reaching disk.
This page covers the example; [LowFive](lowfive.md) covers the VOL-connector mechanism and the
two-HDF5-image hazard it depends on.

## What it demonstrates

`demo_b_producer` and `demo_b_consumer` are two **separate executables**, launched as two separate
`mpiexec` invocations (not one MPMD job — the whole point of the rendezvous below is that the two
programs do not need to be launched together). The producer writes a fixed 8-element dataset
(`{1, 2, 3, 5, 8, 13, 21, 34}`) with an ordinary `H5Dwrite`; the consumer reads it back with an
ordinary `H5Dread`. Between them, LowFive's `DistMetadataVOL` intercepts both calls and moves the
data directly over an MPI intercommunicator the two processes form themselves, so the file named on
each side's command line is never actually created.

**The file split that makes "unmodified HDF5" a checkable claim, not an informal one.**
`producer_hdf5.cpp`/`consumer_hdf5.cpp` are the payload: plain HDF5, open/write/close and
open/read/close. Confirmed directly (`grep -i fuse` over both files and their headers): neither
mentions FUSE at all. `producer_main.cpp`/`consumer_main.cpp` are the FUSE side: each forms the
rendezvous, attaches the LowFive transport, calls into the unmodified payload function, and tears
down:

```{literalinclude} ../../examples/demo_b/producer_main.cpp
:language: cpp
:start-at: "fuse::LowFiveTransport transport;"
:end-at: "r = transport.serve_as_producer();"
```

**The rendezvous** (`examples/demo_b/rendezvous.hpp`) is Demo B's own minimal two-process handshake
— `MPI_Open_port`/`MPI_Comm_accept` on the producer side, `MPI_Comm_connect` on the consumer side,
the port name exchanged through a file both processes can see — deliberately **not**
`fuse::transport::connect()` (a different component's file, whose own rendezvous forms the whole
coupled run's base communicator, considerably more than a self-contained two-process demo needs).
Whether Demo B's rendezvous should eventually become a `connect()` call is an open question recorded
in `KNOWN_ISSUES.md` section D, not decided here.

**The consumer's own pass/fail check is the load-bearing assertion, not a printed message**: exit
code 0 if and only if (a) the read-back values match the producer's bit for bit, and (b) `stat()` on
the HDF5 file path fails — no real file reached disk. A file that *does* exist on disk is reported
as an interception failure, not silently accepted.

## Building and running it

Needs tier 1 (MPI) plus LowFive with the distributed archive, plus MOAB only if it is also enabled
in the same configure (the two-HDF5-image hazard this depends on is [LowFive](lowfive.md)'s own
concern):

```sh
cmake --preset tier1 -DFUSE_ENABLE_LOWFIVE=ON -DFUSE_ENABLE_MPI=ON
cmake --build --preset tier1
examples/demo_b/run_demo.sh <build-dir>
```

`run_demo.sh` launches `demo_b_producer` and `demo_b_consumer` as two separate `mpiexec -n 1`
invocations, waits for both, prints each one's stdout/stderr, and reports `PASS`/`FAIL` from their
combined exit status. The CMake configuration gates the two executables on
`FUSE_ENABLE_LOWFIVE AND FUSE_ENABLE_MPI AND LOWFIVE_HAVE_DIST` (the separate `liblowfive-dist`
archive `cmake/FindLowFive.cmake` records) — on a configure missing any of the three,
`examples/CMakeLists.txt` prints why and neither executable is built at all.

## What it prints

On success, each side prints one line:

```
demo_b_producer: wrote 8 values, served across the intercommunicator
demo_b_consumer: read 8 values, bit-for-bit match, zero bytes of HDF5 on disk
```

`run_demo.sh` prefixes each with `--- producer output ---`/`--- consumer output ---` and prints
`run_demo.sh: PASS` (exit 0) or `run_demo.sh: FAIL (producer exit=.. consumer exit=..)` (exit 1).

## Measured against

Nothing, in this sandbox. `demo_b_producer`/`demo_b_consumer` compile cleanly against this host's
real MOAB/LowFive/MPI headers (`KNOWN_ISSUES.md` D14, measured 2026-10-02), but MPI cannot start
here at all (C1), and this exact combination — FUSE's ABI over `DistMetadataVOL` — was never
implemented in the `upstream/soma` parent project either, whose own `lowfive_transport.cpp` is
single-process only (confirmed by inspection). `run_demo.sh` and the host-run commands it wraps are
recorded in `KNOWN_ISSUES.md` D14 under `host_verification_needed`; no host run of this specific
example has been recorded there as of this writing.

## State

**Measured, one configuration.** The ctest row `example.demo_b` runs `run_demo.sh`: one producer
process and one consumer process, launched separately and joined by `MPI_Comm_accept`/`MPI_Comm_connect`.
On the owner's machine (2026-10-14, MPICH 5.0.0a1, `build-lowfive`) it passed on its first run and in
five repeats (0.16–0.60 s). The producer printed `wrote 8 values, served across the intercommunicator`,
and the consumer printed `read 8 values, bit-for-bit match, zero bytes of HDF5 on disk`, which is the
consumer's own exit-code assertion. `run_demo.sh` with dataset paths `/values` and `/a/b/c/values`
also passed. The first run, before three fixes, failed, and both failures are recorded in
KNOWN_ISSUES.md D14: `set_location`'s writable-directory probe used a fixed name, so two processes
claiming one path refused each other; and the distributed VOL registered no intercommunicator for the
file, so the consumer's `H5Fopen` raised `no intercomm found`. The producer payload now also creates
the dataset's parent group, which `H5Dcreate2` requires in plain HDF5. Not measured: more than one
rank per side, more than one file or dataset, or a second exchange in the same process.
