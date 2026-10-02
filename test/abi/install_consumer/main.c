/* ABI-C-CONSUMER (build-evidence/abi-matrix.toml): a pure-C program, C11
 * -Wpedantic, that links ONLY the installed libfuse-abi and drives a
 * two-component graph through fuse_coupler_abi() -- no fuse:: C++ type
 * reaches this file at all. Its printed values are compared, byte for
 * byte, against ../reference.cpp's in-process C++ run of the identical
 * graph by run-smoke-test.cmake.
 *
 * The graph is the same "upstream" (interval 1) -Provides-> "downstream"
 * pair test/abi/graph_fixture.hpp builds in C++, spelled out here as YAML
 * text (this translation unit cannot include that C++ header) because
 * fuse_load_graph() accepts a YAML path or raw descriptor bytes, and a
 * path this program can write itself is the one of those two a pure-C
 * caller can produce without linking the model library that knows how to
 * serialize a graph.
 */
#include <fuse/abi/participant.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

static void require(int condition, const char* what) {
  if (!condition) {
    fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
  }
}

static const char* kGraphYaml =
    "version: 1\n"
    "meshes:\n"
    "  - {name: m1, source: \"generated:line:4\"}\n"
    "  - {name: m2, source: \"generated:line:4\"}\n"
    "regions:\n"
    "  - {name: r1, mesh: m1, dimension: 1}\n"
    "  - {name: r2, mesh: m2, dimension: 1}\n"
    "components:\n"
    "  - name: upstream\n"
    "    ranks: {first: 0, count: 1}\n"
    "    interval: 1\n"
    "    fields:\n"
    "      - {name: out, region: r1, disc: FV, components: 1, units: K,\n"
    "         conservation: None, intent: Provides}\n"
    "  - name: downstream\n"
    "    ranks: {first: 0, count: 1}\n"
    "    interval: 1\n"
    "    fields:\n"
    "      - {name: in, region: r2, disc: FV, components: 1, units: K,\n"
    "         conservation: None, intent: Requires}\n"
    "channels:\n"
    "  - name: link\n"
    "    source: upstream.out\n"
    "    target: downstream.in\n"
    "    remap: {flags: [Conservative]}\n"
    "iteration_regions: []\n";

static void write_graph_file(const char* path) {
  FILE* f = fopen(path, "w");
  require(f != NULL, "could not create the graph YAML file");
  if (f == NULL) return;
  fputs(kGraphYaml, f);
  fclose(f);
}

int main(void) {
  const fuse_participant_abi_v1* t = fuse_coupler_abi(FUSE_ABI_VERSION);
  require(t != NULL, "fuse_coupler_abi(FUSE_ABI_VERSION) returned NULL");
  if (t == NULL) return 1;

  const char* path = "fuse_abi_consumer_graph.yaml";
  write_graph_file(path);

  fuse_context_out co;
  memset(&co, 0, sizeof(co));
  co.struct_size = (uint32_t)sizeof(co);
  require(t->create(NULL, NULL, &co) == FUSE_OK, "create failed");

  fuse_graph_source src;
  memset(&src, 0, sizeof(src));
  src.struct_size = (uint32_t)sizeof(src);
  src.yaml_path = path;
  require(t->load_graph(co.ctx, &src, NULL) == FUSE_OK, "load_graph failed");

  double up_storage[4]   = {0, 0, 0, 0};
  double down_storage[4] = {0, 0, 0, 0};

  fuse_field_name fname;
  memset(&fname, 0, sizeof(fname));
  fname.struct_size = (uint32_t)sizeof(fname);
  fuse_field_ref fref;
  memset(&fref, 0, sizeof(fref));
  fref.struct_size = (uint32_t)sizeof(fref);
  fuse_field_view fview;
  memset(&fview, 0, sizeof(fview));
  fview.struct_size  = (uint32_t)sizeof(fview);
  fview.element_type = FUSE_ELEMENT_F64;
  fview.memspace     = FUSE_MEMSPACE_HOST;
  fview.ownership    = FUSE_OWNERSHIP_OWNED;
  fview.components   = 1;
  fview.count        = 4;
  fview.owned_count  = 4;
  fview.stride       = 1;

  fname.component = "upstream";
  fname.field     = "out";
  require(t->field_handle(co.ctx, &fname, &fref) == FUSE_OK, "field_handle(upstream.out) failed");
  fview.handle = fref.handle;
  fview.data   = up_storage;
  require(t->register_field(co.ctx, &fview, NULL) == FUSE_OK, "register_field(upstream.out) failed");

  fname.component = "downstream";
  fname.field     = "in";
  require(t->field_handle(co.ctx, &fname, &fref) == FUSE_OK, "field_handle(downstream.in) failed");
  fview.handle = fref.handle;
  fview.data   = down_storage;
  require(t->register_field(co.ctx, &fview, NULL) == FUSE_OK, "register_field(downstream.in) failed");

  require(t->initialize(co.ctx, NULL, NULL) == FUSE_OK, "initialize failed");

  fuse_step_request up_req;
  memset(&up_req, 0, sizeof(up_req));
  up_req.struct_size = (uint32_t)sizeof(up_req);
  up_req.component   = "upstream";
  fuse_step_request down_req;
  memset(&down_req, 0, sizeof(down_req));
  down_req.struct_size = (uint32_t)sizeof(down_req);
  down_req.component   = "downstream";

  for (int period = 0; period < 2; ++period) {
    fuse_step_report rep;
    memset(&rep, 0, sizeof(rep));
    rep.struct_size = (uint32_t)sizeof(rep);
    require(t->begin_step(co.ctx, &up_req, &rep) == FUSE_OK, "begin_step(upstream) failed");
    for (int i = 0; i < 4; ++i) up_storage[i] = 10.0 + (double)period;
    require(t->end_step(co.ctx, &up_req, &rep) == FUSE_OK, "end_step(upstream) failed");

    require(t->begin_step(co.ctx, &down_req, &rep) == FUSE_OK, "begin_step(downstream) failed");
    require(t->end_step(co.ctx, &down_req, &rep) == FUSE_OK, "end_step(downstream) failed");
    /* Read AFTER end_step(), not between begin_step()/end_step(): see
     * src/abi/context.hpp's header comment, "SECOND, NARROWER GAP" -- the
     * real exchange for this period runs inside THIS end_step() call
     * (downstream is the last externally-driven component to complete the
     * period), not inside begin_step(). */
    for (int i = 0; i < 4; ++i) {
      printf("period %d downstream.in[%d] = %a\n", period, i, down_storage[i]);
    }
  }

  require(t->destroy(co.ctx, NULL, NULL) == FUSE_OK, "destroy failed");
  if (failures == 0) fprintf(stderr, "installed C consumer ok\n");
  return failures == 0 ? 0 : 1;
}
