"""YAML round trips, split out from test_graph.py as its own file.

New relative to upstream: a Newton-configured implicit coupling group must round-trip through
YAML, not just a plain channel/field graph -- fuse::IterationBuilder's accelerator and Newton
configuration (include/fuse/model/iteration.hpp) has no soma equivalent at all (see
docs/dev/python-surface.md's evidence table), so upstream's YamlRoundTrip tests never had
anything like it to cover.

STATUS: acceptance test, written against the surface in docs/dev/python-surface.md.
"""

import os

import pytest

pfuse = pytest.importorskip("pfuse", reason="the _pfuse extension is not built")

DATA_DIR = os.environ.get("PFUSE_TEST_DATA_DIR")


def yaml_path(name):
    if not DATA_DIR:
        pytest.skip("PFUSE_TEST_DATA_DIR is not set")
    return os.path.join(DATA_DIR, name)


def build_pair():
    g = pfuse.CouplingGraph()
    m1 = g.add_mesh("fluid_mesh", "generated:box:8x8x8")
    m2 = g.add_mesh("solid_mesh", "generated:box:5x5x5")
    r1 = g.add_region("fluid_wall", m1, entity_set="wall", dimension=2)
    r2 = g.add_region("solid_wall", m2, entity_set="wall", dimension=2)
    fluid = g.add_component("fluid", ranks=pfuse.RankSet(0, 2), interval="1/2")
    solid = g.add_component("solid", ranks=pfuse.RankSet(2, 2), interval=1)
    src = g.add_field(fluid, "heat_flux", r1, units=pfuse.units.W_per_m2,
                       conservation=pfuse.Conservation.Extensive,
                       intent=pfuse.Intent.Provides)
    dst = g.add_field(solid, "q_wall", r2, units=pfuse.units.W_per_m2,
                       conservation=pfuse.Conservation.Extensive,
                       intent=pfuse.Intent.Requires)
    ch = g.add_channel("flux", src, dst)
    (g.channel(ch)
      .accumulate(pfuse.AccumOp.Mean, pfuse.Window.from_component(fluid))
      .remap(pfuse.RemapFlags.Conservative | pfuse.RemapFlags.Monotone)
      .remap_method("mono3")
      .spacetime(pfuse.TemporalBasis.DG, 2, pfuse.Window.explicit_length("2/5"))
      .relax(0.7))
    return g, src, dst


class TestPlainGraphRoundTrip:
    def test_to_yaml_and_back(self):
        g, _src, _dst = build_pair()
        text = g.to_yaml()
        restored = pfuse.from_yaml(text)
        assert pfuse.structurally_equal(g, restored)

    def test_yaml_and_python_produce_the_same_descriptor(self):
        """The claim spec section 3.2 makes, checked against a YAML file C++ also reads."""
        from_file = pfuse.load_yaml_file(yaml_path("pair.yaml"))
        g, _src, _dst = build_pair()
        assert g.to_bytes() == from_file.to_bytes(), \
            "a Python-authored graph must serialize to the same bytes as the YAML front-end's"
        assert pfuse.structurally_equal(g, from_file)

    def test_malformed_yaml_raises_rather_than_returning_a_flag(self):
        """from_yaml()/load_yaml_file() return fuse::Result<CouplingGraph> in C++ -- a parse
        or reference-resolution failure is a Result failure (E-YAML-PARSE/E-YAML-REF/...),
        never thrown, so it is a bare pfuse.FuseError carrying .code
        (docs/dev/python-surface.md's "Errors" section)."""
        with pytest.raises(pfuse.FuseError) as excinfo:
            pfuse.from_yaml("this: is: not: a: graph")
        assert excinfo.value.code == "E-YAML-PARSE"
        with pytest.raises(pfuse.FuseError) as excinfo:
            pfuse.load_yaml_file("/definitely/not/here.yaml")
        assert excinfo.value.code is not None


class TestNewtonConfiguredRegionRoundTrip:
    """New relative to upstream: Newton/accelerator configuration must survive to_yaml/from_yaml."""

    def _graph_with_newton_region(self):
        g, src, dst = build_pair()
        region = (g.iterate([pfuse.ChannelId(0)])
                   .until(pfuse.Converged.Relative(1e-8, "q_wall"))
                   .newton(pfuse.NewtonConfig(rtol=1e-9, max_nonlinear_iterations=30)))
        return g, region

    def test_newton_config_round_trips_through_yaml(self):
        g, region = self._graph_with_newton_region()
        restored = pfuse.from_yaml(g.to_yaml())
        assert pfuse.structurally_equal(g, restored)
        restored_region = restored.iteration_region(region.id())
        assert restored_region.acceleration.kind == pfuse.AccelKind.Newton
        assert restored_region.newton_config.rtol == pytest.approx(1e-9)
        assert restored_region.newton_config.max_nonlinear_iterations == 30
        assert restored_region.requires_residual_capability

    def test_newton_config_round_trips_through_the_descriptor(self):
        """Spec section 3.2: the descriptor is the ABI boundary and is symmetric too."""
        g, _region = self._graph_with_newton_region()
        blob = g.to_bytes()
        restored = pfuse.from_bytes(blob)
        assert pfuse.structurally_equal(g, restored)

    def test_accelerate_with_aitken_round_trips(self):
        g, src, dst = build_pair()
        g.iterate([pfuse.ChannelId(0)]).until(
            pfuse.Converged.Relative(1e-6, "q_wall")).accelerate(
            pfuse.AccelConfig(kind=pfuse.AccelKind.Aitken, initial_relaxation=0.1))
        restored = pfuse.from_yaml(g.to_yaml())
        assert pfuse.structurally_equal(g, restored)
