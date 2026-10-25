// `fuse-lowfive-vol`: the loadable HDF5 VOL plugin that docs/user/lowfive.md
// designs for the single-process case, extended here to the genuinely
// distributed, cross-process case.
//
// **UNMEASURED, like the rest of the distributed half** (docs/user/
// lowfive.md): this file compiles against real LowFive/HDF5/MPI headers on
// this host, but neither this plugin nor the H5PL_* loader path it serves
// has ever been exercised -- the host's LowFive still cannot even dlopen
// (a libfmt.8 conflict), so there is no way, in this sandbox, to confirm the
// design below past "it compiles against the public header contract as closely
// as that contract states it." Treat every claim below as a design, not a
// result.
//
// THE VOL-CONSTRUCTION HALF ONLY. lowfive_plugin_internal.hpp's own
// header comment has the full account of why this is a separate
// translation unit from lowfive_plugin_graph.cpp: this file is the ONE
// place in this plugin that ever names a `<lowfive/...>` header,
// `<hdf5.h>`, or `<mpi.h>` -- and the ONLY thing it ever reads from FUSE
// is `fuse/base/config.hpp` (pure preprocessor macros, no `fmt`, by
// construction) and the plain-data `Config` lowfive_plugin_graph.cpp's
// own object file builds. src/CMakeLists.txt enforces the isolation at
// the object-library level (this file's own object library links
// `fuse::lowfive` and NEVER `fuse::model`), not merely by source-level
// `#include` discipline.
//
// WHY THIS FILE, NOT JUST LowFive's OWN LIBRARY ON HDF5_PLUGIN_PATH.
// LowFive's own vol-base.hpp declares `H5PLget_plugin_type`/
// `H5PLget_plugin_info` at GLOBAL extern "C" scope (not inside
// `namespace LowFive`) -- LowFive's own shared library is ALREADY a
// complete HDF5 plugin by HDF5's own loader contract. But it carries no
// idea of FUSE's own field-to-hdf5-path bindings or of which rank set is
// the writer and which the reader for the DISTRIBUTED case -- those are
// THIS codebase's own concepts, read from the graph YAML named by
// FUSE_LOWFIVE_BINDINGS (lowfive_plugin_graph.cpp's own job). This
// plugin's two extern "C" functions are named identically to LowFive's
// own (HDF5's loader contract fixes the names) but defined in a
// SEPARATE shared library (this target, fuse-lowfive-vol) that
// `HDF5_PLUGIN_PATH` is pointed at INSTEAD of LowFive's own -- there is
// no way to "call LowFive's real H5PLget_plugin_info and then do more"
// from inside a translation unit that also defines a function of that
// same extern "C" name (that would be a second definition of the same
// symbol, not a delegation): this plugin answers HDF5's query itself
// using LowFive's own public class API (`LowFive::VOLBase::connector`,
// the static `H5VL_class_t` LowFive's own constructor populates --
// vol-base.hpp's own comment: "need this static for H5PL_* functions")
// rather than reaching for a same-named sibling function that cannot be
// reached by name from here.
#include "fuse/base/config.hpp"

#if FUSE_HAVE_LOWFIVE

#include "lowfive_plugin_internal.hpp"

#include <lowfive/H5VOLProperty.hpp>
#include <lowfive/vol-metadata.hpp>
#if defined(FUSE_LOWFIVE_DIST_LIB_FOUND) && FUSE_HAVE_MPI
#include <lowfive/vol-dist-metadata.hpp>
#endif

#include <H5PLextern.h>

#if FUSE_HAVE_MPI
#include <mpi.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

namespace {

// Registered once per process, on this plugin's own first H5PLget_
// plugin_info() call (HDF5 calls it exactly once per connector name it
// resolves, per the loader's own documented contract) -- a second
// VOL registration is not safe (lowfive_transport.cpp's own
// vol_singleton()/dist_vol_singleton() comment, which this mirrors).
std::once_flag g_configured_once;

void ApplyBindings(LowFive::MetadataVOL& vol,
                   const std::vector<fuse::lowfive_plugin_internal::FieldBinding>& bindings) {
  for (const auto& b : bindings) vol.set_memory(b.filename_pattern, b.path_pattern);
}

void ConfigureOnce() {
  std::call_once(g_configured_once, [] {
    // The SAME teardown-ordering fix VolSingleton's own constructor
    // registers (lowfive_transport.cpp) and "The HDF5_VOL_CONNECTOR
    // abort" section of docs/user/lowfive.md records in full --
    // registered FIRST, ahead of any native-VOL HDF5 call this process
    // might make before this plugin's own setup finishes, so it lands
    // ahead of HDF5's own atexit registration in the LIFO unwind order.
    std::atexit(+[] { H5close(); });

    fuse::lowfive_plugin_internal::Config cfg;
    const bool config_loaded = fuse::lowfive_plugin_internal::LoadConfig(&cfg);

    bool distributed_configured = false;
#if defined(FUSE_LOWFIVE_DIST_LIB_FOUND) && FUSE_HAVE_MPI
    // MPI_Comm_split and MPI_Intercomm_create are each collective over
    // MPI_COMM_WORLD (the split) or the union of the two local groups it
    // produces (the intercomm rendezvous), so the split must not sit inside
    // `if (cfg.distributed)` -- a rank the LowFive channel does not name
    // never takes `cfg.distributed`, so it would never call either, while a
    // rank that did would take the opposite branch and block in them
    // forever. The split below is issued on every rank
    // that reaches this line, participant or not: a non-participant
    // passes MPI_UNDEFINED, which MPI defines to hand it MPI_COMM_NULL and
    // exclude it from the result without excluding it from the
    // collective. Only a participant's `local` is non-null, so only a
    // participant goes on to the leader-to-leader rendezvous.
    //
    // This covers the participant-only-call hazard; it does not cover a
    // second one -- the split still runs only on a rank whose OWN first
    // H5P_DEFAULT call triggers this lazily-loaded plugin at all, and a rank
    // that never makes one (a pure orchestrator touching no HDF5 file) still
    // never reaches this line. Moving the rendezvous into an explicit step
    // every participant calls regardless of its own HDF5 usage (connect()'s own
    // MPMD choreography is the natural site, since every FUSE-aware process in
    // the launch already calls it) is the larger fix that closes that half; it
    // is not done here, and is unmeasured either way: this plugin has not run
    // on a host yet.
    {
      int split_color = MPI_UNDEFINED;
      int split_key   = 0;
      if (config_loaded && cfg.distributed) {
        split_color = cfg.rendezvous.split_color;
        split_key   = cfg.rendezvous.split_key;
      }
      MPI_Comm local = MPI_COMM_NULL;
      MPI_Comm_split(MPI_COMM_WORLD, split_color, split_key, &local);
      if (config_loaded && cfg.distributed) {
        MPI_Comm intercomm = MPI_COMM_NULL;
        MPI_Intercomm_create(local, cfg.rendezvous.local_leader, MPI_COMM_WORLD,
                             cfg.rendezvous.remote_leader, cfg.rendezvous.tag, &intercomm);
        LowFive::DistMetadataVOL& vol =
            LowFive::DistMetadataVOL::create_DistMetadataVOL(local, intercomm);
        ApplyBindings(vol, cfg.bindings);
        distributed_configured = true;
      }
    }
#endif

    // A failed LoadConfig() (missing FUSE_LOWFIVE_BINDINGS, a bad graph
    // path) or a distributed config this build cannot honour (no
    // DistMetadataVOL) must not attach MetadataVOL with zero bindings --
    // every later HDF5 call would then silently reach a real file instead
    // of refusing, which is exactly the substitution transport.hpp's own
    // contract and docs/user/lowfive.md both rule out ("it never silently
    // degrades to file I/O"). Both cases abort the job with the diagnostic
    // LoadConfig() or this block already printed, instead of constructing
    // a connector that cannot do the job it was loaded to do.
    if (!config_loaded || (cfg.distributed && !distributed_configured)) {
      if (cfg.distributed && !distributed_configured) {
        // fputs, not fmt::print: this file's whole reason for existing is
        // to never include a header that pulls fmt in either of its two
        // conflicting forms (lowfive_plugin_internal.hpp's own comment) --
        // raw stdio is the only formatter available here at all.
        std::fputs(  // NOLINT(fuse-raw-io)
            "fuse-lowfive-vol: a distributed rendezvous was computed but this plugin "
            "was built without DistMetadataVOL support -- refusing rather than silently "
            "falling back to a connector that will NOT cross processes\n",
            stderr);
      }
      std::fflush(stderr);
#if FUSE_HAVE_MPI
      int mpi_initialized = 0;
      MPI_Initialized(&mpi_initialized);
      if (mpi_initialized) {
        MPI_Abort(MPI_COMM_WORLD, 71);
      }
#endif
      std::abort();
    }

    if (!distributed_configured) {
      // Single-process connector -- identical to VolSingleton's own
      // construction (lowfive_transport.cpp), but with this plugin's
      // own, graph-specific bindings instead of one wildcard pattern.
      // Reached only when LoadConfig() succeeded and the config is
      // honestly single-process (cfg.distributed is false), never as a
      // degraded substitute for a distributed request that failed.
      LowFive::MetadataVOL& vol = LowFive::MetadataVOL::create_MetadataVOL();
      ApplyBindings(vol, cfg.bindings);
    }
  });
}

}  // namespace

extern "C" {

H5PL_type_t H5PLget_plugin_type(void) { return H5PL_TYPE_VOL; }

const void* H5PLget_plugin_info(void) {
  ConfigureOnce();
  // `LowFive::VOLBase::connector` -- vol-base.hpp's own comment: "need
  // this static for H5PL_* functions (for automatic plugin loading via
  // environment variables)" -- populated by whichever VOLBase-derived
  // object ConfigureOnce() constructed above (MetadataVOL or
  // DistMetadataVOL; both derive from VOLBase, and the static member is
  // shared, by construction, across either).
  return static_cast<const void*>(&LowFive::VOLBase::connector);
}

}  // extern "C"

#endif  // FUSE_HAVE_LOWFIVE
