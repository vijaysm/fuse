// A reproducer for the two-HDF5-image hazard (docs/user/lowfive.md):
// a binary that links both MOAB and LowFive may load TWO independent
// `libhdf5` images -- one through each library's own linkage -- because
// MOAB's HDF5 dependency is recorded as an absolute path while LowFive's may
// resolve, via its own LC_RPATH (macOS) / RPATH (Linux), to a different
// prefix. dyld/ld.so then treat them as two distinct shared objects, each
// with its own copy of HDF5's process-wide state (the property-list table,
// the VOL connector registry, free lists, ...), and a handle minted by one
// image is meaningless to the other.
//
// Despite the `.c` name (kept for historical reasons), this is a
// C++ translation unit: both MOAB's and LowFive's public APIs are C++
// classes with no C binding, so a probe that actually links against them
// cannot be written in C. tools/probe-hdf5-images.sh compiles it with a C++
// compiler (`-x c++`) rather than renaming the file.
//
// This file only has to link against both libraries and exercise one real,
// non-inlinable call through each; tools/probe-hdf5-images.sh does the
// counting, by running the resulting binary under a library-loader trace
// (DYLD_PRINT_LIBRARIES=1 on macOS, LD_DEBUG=libs on Linux) and counting the
// distinct resolved paths that match `libhdf5` / `libhdf5_hl`.
//
// WHY A SEPARATE H5Pcreate CALL IN main(), NOT JUST THE TWO LIBRARY CALLS.
// dyld maps every directly and transitively linked dylib at process launch
// whether or not any of its symbols are ever referenced, so the image count
// alone does not need a live call to appear. The H5Pcreate/H5Pclose pair
// below exists to reproduce the SILENT form of the hazard, not just the
// image count: it creates a file-access property list through whichever
// HDF5 this translation unit was linked against (the .sh wrapper's
// `--hdf5-prefix`, meant to mirror whatever `find_package(HDF5)` resolves to
// for the LowFive feature) and hands that property-list id to LowFive's
// `H5VOLProperty::apply()`, exactly as `fuse::LowFiveTransport::build_plan()`
// does (src/transport/lowfive_transport.cpp). When the two libraries are
// bound to different HDF5 images, `H5VOLProperty::apply()`'s own
// `H5Pset_vol` reports "not a file VOL ID" -- a property-list id from one
// image's id table is not a valid id in the other's -- even though the
// process links without error and nothing crashes. That silence, measured
// directly (see docs/user/lowfive.md), is the hazard this probe exists to
// catch before it reaches a coupled run.
#include <hdf5.h>
#include <moab/Core.hpp>
#include <lowfive/vol-metadata.hpp>
#include <lowfive/H5VOLProperty.hpp>

#include <cstdio>

int main(void) {
  // Forces linkage through MOAB's own copy of libhdf5 (every MOAB build in
  // this codebase's tier 2 links HDF5 unconditionally for mesh I/O).
  moab::Core core;
  (void)core;

  // Forces linkage through LowFive's own copy: constructing the VOL
  // singleton and applying it to a property list is exactly
  // fuse::LowFiveTransport's own sequence.
  LowFive::MetadataVOL& vol = LowFive::MetadataVOL::create_MetadataVOL();
  LowFive::H5VOLProperty prop(vol);

  const hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
  if (fapl < 0) {
    std::fprintf(stderr, "probe: H5Pcreate(H5P_FILE_ACCESS) failed\n");
    return 2;
  }
  prop.apply(fapl);

  // Read the connector back off the property list and compare it with the
  // one just registered -- not "the call returned 0", exactly the check
  // LowFiveTransport::build_plan() makes before trusting the connector is
  // live (E-TRANSPORT-LF-VOL).
  hid_t on_list = H5I_INVALID_HID;
  const herr_t got = H5Pget_vol_id(fapl, &on_list);
  const int matches = (got >= 0) && (on_list == prop.vol_id);
  if (on_list >= 0) H5VLclose(on_list);
  H5Pclose(fapl);

  std::printf("probe: vol_id=%lld fapl_vol_matches=%s\n",
              (long long)prop.vol_id, matches ? "yes" : "NO");
  return matches ? 0 : 1;
}
