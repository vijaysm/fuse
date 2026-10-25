#!/usr/bin/env bash
# tools/probe-hdf5-images.sh -- the acceptance test for the two-HDF5-image
# hazard (docs/user/lowfive.md).
#
# Builds tools/probe-hdf5-images.c against a given LowFive installation and
# MOAB (both are needed to reproduce the hazard: it does not occur with
# either library alone), runs the result under the platform's library-loader
# trace, and counts the distinct `libhdf5` / `libhdf5_hl` images actually
# mapped into the process. Exits 0 iff exactly one of each loaded; exits 1
# and names the duplicate paths plus the exact rebuild line otherwise; exits
# 2 if the probe could not be built or run for a reason unrelated to the
# hazard (MOAB or LowFive not found, a compiler error, ...) -- distinguished
# from exit 1 so a caller (cmake/FuseDependencies.cmake) never mistakes
# "untested" for "passed".
#
# Usage:
#   tools/probe-hdf5-images.sh <lowfive_prefix> [moab_dir] [hdf5_prefix]
#
#   <lowfive_prefix>  required. Must contain include/lowfive/*.hpp and
#                     lib/liblowfive.{dylib,so}.
#   [moab_dir]        default /opt/moab/cmake-all. Must contain
#                     include/moab/Core.hpp and lib/libMOAB.{dylib,so} (or a
#                     lib/cmake/MOAB/MOABConfig.cmake CMake ships, which this
#                     script does not parse -- point it at the lib/include
#                     root directly).
#   [hdf5_prefix]     default <lowfive_prefix> itself (co-located
#                     include/hdf5.h + lib/libhdf5.*). This is the HDF5 the
#                     probe's OWN H5Pcreate call is linked against, meant to
#                     mirror whatever `find_package(HDF5)` resolves to for a
#                     real FUSE_ENABLE_LOWFIVE build; override it to test a
#                     mismatched configuration deliberately.
#
# Environment overrides (all optional; defaults are this lab's /opt layout,
# matching cmake/FuseDependencies.cmake and docs/BUILDING.md elsewhere in
# this tree):
#   FUSE_PROBE_CXX                 C++ compiler/launcher (default: an MPI
#                                   wrapper, since MOAB here is built
#                                   USE_MPI=ON and its headers pull in
#                                   <mpi.h>): /opt/mpich/bin/mpicxx
#   FUSE_PROBE_MOAB_EXTRA_INCLUDES space-separated extra include dirs MOAB's
#                                   own headers need transitively (zoltan,
#                                   pnetcdf, netcdf, tempestremap on this
#                                   lab's MOAB build)
#   FUSE_PROBE_FMT_INCLUDE          include dir providing fmt/core.h for
#                                   LowFive's own headers (vol-metadata.hpp
#                                   includes <fmt/core.h> unconditionally).
#                                   Auto-detected under <lowfive_prefix>, else
#                                   DIY's vendored copy, else must be given.
#   FUSE_PROBE_BUILD_DIR            default: mktemp -d
#   FUSE_PROBE_KEEP_BUILD=1         keep FUSE_PROBE_BUILD_DIR instead of
#                                   deleting it on exit (for inspection)
#
# MEASURED on a development host (see docs/user/lowfive.md for the full
# record and the exact commands):
#   ./probe-hdf5-images.sh /opt/lowfive                 -> PASS (one image
#     each; the stock /opt/lowfive liblowfive.dylib already links MOAB's own
#     absolute /opt/hdf5 path) but the probe binary itself cannot run to
#     completion here: /opt/lowfive was built against an older MPICH ABI
#     (libmpicxx.12) than the one currently installed at /opt/mpich
#     (libmpicxx.0) -- a SEPARATE, unrelated environment defect this script
#     reports but does not fail on, because the loader trace reaches both
#     hdf5 images before that unrelated abort.
#   ./probe-hdf5-images.sh /opt/lowfive-shared           -> FAIL. Four hdf5
#     images load (/opt/hdf5 x2 via MOAB, /opt/hdf5-shared x2 via LowFive's
#     own LC_RPATH), and when the probe's own H5Pcreate is linked against
#     /opt/hdf5 (the realistic case: MOAB is always linked and brings that
#     path in), LowFive's H5Pset_vol reports "not a file VOL ID" -- the
#     silent-corruption symptom, caught here as a loud, non-zero exit
#     instead.

set -u -o pipefail

die() { echo "probe-hdf5-images.sh: $*" >&2; exit 2; }

[ $# -ge 1 ] || die "usage: $0 <lowfive_prefix> [moab_dir] [hdf5_prefix]"
LOWFIVE_PREFIX=$1
MOAB_DIR=${2:-/opt/moab/cmake-all}
HDF5_PREFIX=${3:-}

[ -f "$LOWFIVE_PREFIX/include/lowfive/vol-metadata.hpp" ] || \
  die "no include/lowfive/vol-metadata.hpp under '$LOWFIVE_PREFIX'"
[ -f "$MOAB_DIR/include/moab/Core.hpp" ] || \
  die "no include/moab/Core.hpp under '$MOAB_DIR'; pass moab_dir explicitly"

# HDF5_PREFIX defaults to whichever HDF5 liblowfive.dylib/.so itself depends
# on (parsed from its own load commands), not to co-location under
# LOWFIVE_PREFIX: a LowFive built shared (as a VOL plugin must be) commonly
# installs its headers and library without re-exporting HDF5's, and the
# realistic FUSE_ENABLE_LOWFIVE configuration links the probe's own direct
# HDF5 calls against whatever `find_package(HDF5)` resolves to for THIS
# LowFive -- i.e. the one it was built against.
if [ -z "$HDF5_PREFIX" ]; then
  LF_LIB="$LOWFIVE_PREFIX/lib/liblowfive.dylib"
  [ -f "$LF_LIB" ] || LF_LIB="$LOWFIVE_PREFIX/lib/liblowfive.so"
  if [ -f "$LF_LIB" ] && command -v otool >/dev/null 2>&1; then
    HDF5_LIB_PATH=$(otool -L "$LF_LIB" 2>/dev/null | grep -oE '/[^[:space:]]*/libhdf5\.[0-9][0-9A-Za-z.]*\.dylib' | head -1)
    [ -n "$HDF5_LIB_PATH" ] && HDF5_PREFIX=$(dirname "$(dirname "$HDF5_LIB_PATH")")
  elif [ -f "$LF_LIB" ] && command -v ldd >/dev/null 2>&1; then
    HDF5_LIB_PATH=$(ldd "$LF_LIB" 2>/dev/null | grep -oE '/[^[:space:]]*/libhdf5\.so[0-9.]*' | head -1)
    [ -n "$HDF5_LIB_PATH" ] && HDF5_PREFIX=$(dirname "$(dirname "$HDF5_LIB_PATH")")
  fi
  [ -z "$HDF5_PREFIX" ] && HDF5_PREFIX=/opt/hdf5
  echo "probe-hdf5-images.sh: hdf5_prefix not given; using '$HDF5_PREFIX' (detected from $LF_LIB's own linkage, or the /opt/hdf5 fallback)"
fi
[ -f "$HDF5_PREFIX/include/hdf5.h" ] || \
  die "no include/hdf5.h under '$HDF5_PREFIX'; pass hdf5_prefix explicitly"

CXX=${FUSE_PROBE_CXX:-/opt/mpich/bin/mpicxx}
command -v "$CXX" >/dev/null 2>&1 || die "C++ compiler '$CXX' not found (set FUSE_PROBE_CXX)"

# fmt: LowFive's vol-metadata.hpp includes <fmt/core.h> unconditionally.
FMT_INCLUDE=${FUSE_PROBE_FMT_INCLUDE:-}
if [ -z "$FMT_INCLUDE" ]; then
  if [ -f "$LOWFIVE_PREFIX/include/fmt/core.h" ]; then
    FMT_INCLUDE="$LOWFIVE_PREFIX/include"
  elif [ -f /opt/diy/include/diy/thirdparty/fmt/core.h ]; then
    FMT_INCLUDE=/opt/diy/include/diy/thirdparty
  else
    die "no fmt/core.h found (checked '$LOWFIVE_PREFIX/include' and DIY's vendored copy); set FUSE_PROBE_FMT_INCLUDE"
  fi
fi

MOAB_EXTRA_INCLUDES=${FUSE_PROBE_MOAB_EXTRA_INCLUDES:-"/opt/zoltan/include /opt/pnetcdf/include /opt/netcdf/include /opt/tempestremap/libraries/include"}

BUILD_DIR=${FUSE_PROBE_BUILD_DIR:-}
if [ -z "$BUILD_DIR" ]; then
  BUILD_DIR=$(mktemp -d 2>/dev/null) || BUILD_DIR="./.fuse-probe-build-$$"
fi
mkdir -p "$BUILD_DIR" || die "cannot create build directory '$BUILD_DIR' (set FUSE_PROBE_BUILD_DIR to a writable path)"
cleanup() { [ "${FUSE_PROBE_KEEP_BUILD:-0}" = "1" ] || rm -rf "$BUILD_DIR"; }
trap cleanup EXIT

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/probe-hdf5-images.c"
[ -f "$SRC" ] || die "cannot find probe-hdf5-images.c next to this script"

BIN="$BUILD_DIR/probe-hdf5-images"
BUILD_LOG="$BUILD_DIR/build.log"

INCLUDE_FLAGS=(-I"$MOAB_DIR/include")
for d in $MOAB_EXTRA_INCLUDES; do INCLUDE_FLAGS+=(-I"$d"); done
INCLUDE_FLAGS+=(-I"$HDF5_PREFIX/include" -I"$LOWFIVE_PREFIX/include" -I"$FMT_INCLUDE")

UNAME_S=$(uname -s)
RPATH_FLAGS=()
if [ "$UNAME_S" = "Darwin" ]; then
  RPATH_FLAGS=(-Wl,-rpath,"$LOWFIVE_PREFIX/lib" -Wl,-rpath,"$HDF5_PREFIX/lib" -Wl,-rpath,"$MOAB_DIR/lib")
else
  RPATH_FLAGS=(-Wl,-rpath,"$LOWFIVE_PREFIX/lib:$HDF5_PREFIX/lib:$MOAB_DIR/lib")
fi

# -x c++: see the comment in probe-hdf5-images.c on why this is C++ despite
# the filename. -DFMT_HEADER_ONLY=1: avoids needing a separately compiled
# libfmt whose ABI may not match LowFive's vendored fmt version.
if ! "$CXX" -std=c++17 -x c++ -DFMT_HEADER_ONLY=1 "${INCLUDE_FLAGS[@]}" \
      "$SRC" \
      -L"$MOAB_DIR/lib" -lMOAB \
      -L"$LOWFIVE_PREFIX/lib" -llowfive \
      -L"$HDF5_PREFIX/lib" -lhdf5 \
      "${RPATH_FLAGS[@]}" \
      -o "$BIN" >"$BUILD_LOG" 2>&1; then
  echo "probe-hdf5-images.sh: BUILD FAILED (unrelated to the hazard this probe tests)" >&2
  tail -n 60 "$BUILD_LOG" >&2
  exit 2
fi

RUN_LOG="$BUILD_DIR/run.log"
if [ "$UNAME_S" = "Darwin" ]; then
  DYLD_PRINT_LIBRARIES=1 "$BIN" >"$RUN_LOG" 2>&1
  RUN_STATUS=$?
  # Lines look like: dyld[1234]: <UUID> /opt/hdf5/lib/libhdf5.310.dylib
  HDF5_PATHS=$(grep -oE '/[^[:space:]]*/libhdf5\.[0-9][0-9A-Za-z.]*\.dylib' "$RUN_LOG" | sort -u)
  HDF5_HL_PATHS=$(grep -oE '/[^[:space:]]*/libhdf5_hl\.[0-9][0-9A-Za-z.]*\.dylib' "$RUN_LOG" | sort -u)
else
  LD_DEBUG=libs "$BIN" >"$RUN_LOG" 2>&1
  RUN_STATUS=$?
  # LD_DEBUG=libs prints lines such as "trying file=/opt/hdf5/lib/libhdf5.so.310".
  # Not verified on Linux in this sandbox (Darwin-only host) -- see
  # docs/user/lowfive.md's "Linux" note.
  HDF5_PATHS=$(grep -oE 'file=[^[:space:];]*/libhdf5\.so[0-9.]*' "$RUN_LOG" | sed 's/^file=//' | sort -u)
  HDF5_HL_PATHS=$(grep -oE 'file=[^[:space:];]*/libhdf5_hl\.so[0-9.]*' "$RUN_LOG" | sed 's/^file=//' | sort -u)
fi

N_HDF5=$(printf '%s\n' "$HDF5_PATHS" | grep -c . || true)
N_HDF5_HL=$(printf '%s\n' "$HDF5_HL_PATHS" | grep -c . || true)

echo "probe-hdf5-images.sh: lowfive_prefix=$LOWFIVE_PREFIX moab_dir=$MOAB_DIR hdf5_prefix=$HDF5_PREFIX"
echo "probe-hdf5-images.sh: libhdf5 images loaded ($N_HDF5):"
printf '%s\n' "$HDF5_PATHS" | sed 's/^/  /'
echo "probe-hdf5-images.sh: libhdf5_hl images loaded ($N_HDF5_HL):"
printf '%s\n' "$HDF5_HL_PATHS" | sed 's/^/  /'

if grep -q "Library not loaded\|cannot open shared object file" "$RUN_LOG"; then
  echo "probe-hdf5-images.sh: WARNING -- the probe binary could not run to" >&2
  echo "  completion (a dependency failed to load). This is reported" >&2
  echo "  separately from the two-image hazard verdict below, because the" >&2
  echo "  loader's trace up to the point of that failure already answers" >&2
  echo "  the image-count question; see docs/user/lowfive.md if this" >&2
  echo "  warning is unexpected." >&2
fi

if [ "$N_HDF5" -eq 1 ] && [ "$N_HDF5_HL" -eq 1 ]; then
  echo "probe-hdf5-images.sh: PASS -- exactly one libhdf5 and one libhdf5_hl image; safe to pair this LowFive with this MOAB."
  exit 0
fi

cat >&2 <<EOF
probe-hdf5-images.sh: FAIL -- the two-HDF5-image hazard is present for this
  pairing ($N_HDF5 distinct libhdf5 image(s), $N_HDF5_HL distinct libhdf5_hl
  image(s); exactly one of each is required). LowFive's VOL interception
  will silently not apply to MOAB's own HDF5 traffic, or vice versa -- no
  link error, no runtime error, data simply reaches disk unintercepted.

  Fix: rebuild LowFive against the SAME HDF5 prefix MOAB uses
  ($MOAB_DIR's MOABConfig.cmake names its HDF5 prefix; on this host that is
  /opt/hdf5, which is already a shared, VOL-pluggable build -- confirmed by
  'otool -L $MOAB_DIR/lib/libMOAB.dylib | grep hdf5'), not a separately
  built "shared HDF5" prefix:

    git clone --recursive https://github.com/diatomic/LowFive.git /tmp/lowfive-src
    cmake -B /tmp/lowfive-build -S /tmp/lowfive-src \\
          -DCMAKE_INSTALL_PREFIX=/opt/lowfive-hdf5-matched \\
          -DCMAKE_PREFIX_PATH="<MOAB's HDF5 prefix>;/opt/diy" \\
          -DCMAKE_CXX_COMPILER=<the MPI compiler MOAB was built with> \\
          -DBUILD_SHARED_LIBS=ON -Dbuild_examples=OFF
    cmake --build /tmp/lowfive-build -j8 && cmake --install /tmp/lowfive-build

  Then reconfigure FUSE with -DFUSE_ENABLE_LOWFIVE=ON and
  -DCMAKE_PREFIX_PATH pointed at /opt/lowfive-hdf5-matched (and at NO other
  HDF5 prefix -- find_package(HDF5) must resolve to the SAME path).
EOF
exit 1
