"""The Python front-end must build the same fuse::CouplingGraph as C++ and YAML.

Transplanted from upstream soma's test_graph.py (spec section 3.2) onto fuse's current model API
(include/fuse/model/coupling_graph.hpp, descriptor.hpp, iteration.hpp). See
docs/dev/python-surface.md's evidence table for the per-class carry-over/rename/new mapping this
file exercises.

STATUS: acceptance test, written against the surface in docs/dev/python-surface.md; `pfuse` does
not exist yet (Stage B builds it).
"""

import os
from fractions import Fraction

import numpy as np
import pytest

pfuse = pytest.importorskip("pfuse", reason="Stage B has not built the _pfuse extension yet")

DATA_DIR = os.environ.get("PFUSE_TEST_DATA_DIR")


def build_pair() -> "pfuse.CouplingGraph":
    """A small two-component conjugate-heat-transfer graph, the shape test/data/pair.yaml uses."""
    g = pfuse.CouplingGraph()
    m1 = g.add_mesh("fluid_mesh", "generated:box:8x8x8")
    m2 = g.add_mesh("solid_mesh", "generated:box:5x5x5")
    r1 = g.add_region("fluid_wall", m1, entity_set="wall", dimension=2)
    r2 = g.add_region("solid_wall", m2, entity_set="wall", dimension=2)

    fluid = g.add_component("fluid", ranks=pfuse.RankSet(0, 2), interval="1/2")
    solid = g.add_component("solid", ranks=pfuse.RankSet(2, 2), interval=1)

    q_out = g.add_field(fluid, "heat_flux", r1, units=pfuse.units.W_per_m2,
                         conservation=pfuse.Conservation.Extensive,
                         intent=pfuse.Intent.Provides)
    q_in = g.add_field(solid, "q_wall", r2, units=pfuse.units.W_per_m2,
                        conservation=pfuse.Conservation.Extensive,
                        intent=pfuse.Intent.Requires)

    ch = g.add_channel("flux", q_out, q_in)
    g.channel(ch).accumulate(pfuse.AccumOp.Mean).remap(
        pfuse.RemapFlags.Conservative | pfuse.RemapFlags.Monotone).relax(0.7)
    return g


class TestBuildsAGraph:
    def test_counts(self):
        g = build_pair()
        assert g.component_count == 2
        assert g.channel_count == 1
        assert g.field_count == 2
        assert g.mesh_count == 2
        assert g.region_count == 2

    def test_channel_builder_recorded_what_it_was_told(self):
        g = build_pair()
        ch = g.channel_at(pfuse.ChannelId(0))
        assert ch.name == "flux"
        assert ch.accumulate.op == pfuse.AccumOp.Mean
        assert ch.relax.enabled
        assert ch.relax.omega == pytest.approx(0.7)
        # RemapFlags is arithmetic (fuse/model/channel.hpp's operator| / operator&).
        assert pfuse.has(ch.remap.flags, pfuse.RemapFlags.Conservative)
        assert pfuse.has(ch.remap.flags, pfuse.RemapFlags.Monotone)
        assert not pfuse.has(ch.remap.flags, pfuse.RemapFlags.HighOrder)

    def test_field_metadata_survives(self):
        g = build_pair()
        f = g.field(pfuse.FieldId(0))
        assert f.name == "heat_flux"
        assert f.components == 1
        assert f.units == pfuse.units.W_per_m2
        assert f.intent == pfuse.Intent.Provides
        assert g.owner_of(pfuse.FieldId(0)) == g.find_component("fluid")

    def test_find_component_raises_rather_than_returning_a_sentinel(self):
        """find_component() wraps CouplingGraph::find_component()'s std::optional into a
        raise via FUSE_THROW(fuse::InvalidArgument, ...) -- there is no dedicated "not found"
        class among the six fuse::Error-mirroring subclasses (the owner's ruling fixes that
        list at exactly six), so a lookup failure is reported as the generic InvalidArgument
        ("the caller broke a documented precondition": naming a component that does not
        exist). InvalidArgument dual-inherits ValueError, not LookupError."""
        g = build_pair()
        with pytest.raises(pfuse.InvalidArgument):
            g.find_component("no_such_component")
        with pytest.raises(ValueError):
            g.find_component("no_such_component")

    def test_add_coupler_is_a_fieldless_placement(self):
        """fuse::CouplingGraph::add_coupler: new relative to soma, no fields, kind == Coupler."""
        g = pfuse.CouplingGraph()
        c = g.add_coupler("hub", ranks=pfuse.RankSet(0, 1))
        decl = g.component(c)
        assert decl.kind == pfuse.ComponentKind.Coupler
        assert decl.fields == []


class TestIntervalCoercion:
    def test_accepts_string_fraction_and_int(self):
        g = pfuse.CouplingGraph()
        a = g.add_component("a", interval="1/3")
        b = g.add_component("b", interval=Fraction(1, 7))
        c = g.add_component("c", interval=2)
        d = g.add_component("d", interval=pfuse.Rational(5, 10))
        assert g.component(a).interval == pfuse.Rational(1, 3)
        assert g.component(b).interval == pfuse.Rational(1, 7)
        assert g.component(c).interval == pfuse.Rational(2)
        # Rational reduces on construction, so 5/10 is 1/2.
        assert g.component(d).interval == pfuse.Rational(1, 2)

    def test_rejects_a_float(self):
        """Spec 4.3: a double end time makes 'is the run over?' rounding-dependent."""
        g = pfuse.CouplingGraph()
        with pytest.raises(pfuse.InvalidArgument, match="float"):
            g.add_component("bad", interval=0.5)
        # Even though 0.5 IS exactly representable: the rule is about the type, not the
        # value, because 1/3 is not and the caller cannot tell them apart at the call site.
        with pytest.raises(ValueError):
            g.add_component("bad2", interval=1.0)

    def test_rejects_a_zero_denominator(self):
        with pytest.raises(Exception):
            pfuse.Rational(1, 0)

    def test_rational_helper_matches_the_keyword(self):
        assert pfuse.rational("1/3") == pfuse.Rational(1, 3)
        assert pfuse.rational(Fraction(2, 6)) == pfuse.Rational(1, 3)
        assert pfuse.rational(4) == pfuse.Rational(4, 1)
        assert float(pfuse.Rational(1, 4)) == 0.25


class TestDescriptorRoundTrip:
    def test_round_trip(self):
        g = build_pair()
        blob = g.to_bytes()
        assert isinstance(blob, bytes)
        assert len(blob) > 0
        assert pfuse.structurally_equal(g, pfuse.from_bytes(blob))

    def test_malformed_descriptor_is_rejected(self):
        """deserialize()'s E-DESC-MAGIC is a fuse::Result failure, not a thrown fuse::Error --
        a bare pfuse.FuseError, per the owner's ruling (see TestExceptionHierarchy, below)."""
        with pytest.raises(pfuse.FuseError, match="E-DESC-MAGIC") as excinfo:
            pfuse.from_bytes(b"not a fuse descriptor")
        assert excinfo.value.code == "E-DESC-MAGIC"


class TestValidation:
    def test_reports_unit_mismatch(self):
        g = pfuse.CouplingGraph()
        m = g.add_mesh("m", "generated:box:2x2x2")
        r = g.add_region("r", m, dimension=3)
        a = g.add_component("a", ranks=pfuse.RankSet(0, 1))
        b = g.add_component("b", ranks=pfuse.RankSet(1, 1))
        s = g.add_field(a, "T", r, units=pfuse.units.K, intent=pfuse.Intent.Provides)
        t = g.add_field(b, "q", r, units=pfuse.units.W_per_m2, intent=pfuse.Intent.Requires)
        g.add_channel("bad", s, t)

        diags = pfuse.validate(g)
        assert diags.has_errors
        assert any(d.code == "E-UNITS" for d in diags.items)
        # Global Constraints (spec): every problem, together.
        assert len(list(diags)) == len(diags)
        assert "E-UNITS" in diags.to_string()

    def test_a_clean_graph_has_no_errors(self):
        diags = pfuse.validate(build_pair())
        assert not diags.has_errors


class TestLowering:
    def test_produces_an_ir_dump(self):
        g = build_pair()
        result = pfuse.lower(g)
        assert result.ok, result.diagnostics.to_string()
        text = pfuse.dump_text(result.dag)
        assert text.startswith("# fuse-ir v1"), text[:80]
        assert "Accumulate" in text
        assert "Remap" in text

    def test_graphviz_dump_is_a_digraph(self):
        # Was a strict xfail while include/fuse/ir/dump.hpp still wrote 'digraph soma_ir {'
        # (a leftover from the soma->fuse rename); the header has been fixed, so this is
        # now a plain assertion on the renamed banner.
        result = pfuse.lower(build_pair())
        assert pfuse.dump_graphviz(result.dag).startswith("digraph fuse_ir")


class TestExceptionHierarchy:
    """The owner's ruling (2026-10-12): pfuse.FuseError plus exactly SIX subclasses mirroring
    fuse::Error's own C++ hierarchy (error.hpp) -- no further triage of fuse::Result failure
    codes into more classes. Every fuse::Result failure (an E-* code, never thrown in C++) is
    raised as a BARE pfuse.FuseError carrying .code/.message; only a C++ fuse::Error that is
    actually THROWN crosses as one of the six below."""

    def test_every_pfuse_error_is_a_fuse_error(self):
        for cls in (pfuse.InvalidArgument, pfuse.NullPointer, pfuse.Refusal,
                    pfuse.NotImplementedFeature, pfuse.IoError, pfuse.InvariantViolation):
            assert issubclass(cls, pfuse.FuseError), cls

    def test_null_pointer_mirrors_the_cxx_hierarchy(self):
        """error.hpp: `class NullPointer : public InvalidArgument` -- not a direct child of
        Error. The Python hierarchy mirrors that exactly, not just the leaf-to-FuseError
        membership."""
        assert issubclass(pfuse.NullPointer, pfuse.InvalidArgument)

    def test_each_also_derives_from_the_closest_builtin(self):
        assert issubclass(pfuse.InvalidArgument, ValueError)
        assert issubclass(pfuse.NullPointer, ValueError)  # via InvalidArgument
        assert issubclass(pfuse.Refusal, ValueError)
        assert issubclass(pfuse.NotImplementedFeature, NotImplementedError)
        assert issubclass(pfuse.IoError, OSError)
        # pfuse.FuseError itself derives from RuntimeError (the closest builtin to
        # fuse::Error's own base, std::runtime_error), so EVERY subclass -- InvariantViolation
        # included -- is unavoidably a RuntimeError too; that is not a deliberate second
        # mixin. What IS deliberate: InvariantViolation gets no FURTHER, more specific
        # builtin the way the other five each do (ValueError/NotImplementedError/OSError) --
        # it signals a library bug, not a caller mistake one of those more specific builtins
        # would characterise correctly.
        assert issubclass(pfuse.InvariantViolation, RuntimeError)
        assert not issubclass(pfuse.InvariantViolation, (ValueError, LookupError, OSError,
                                                         NotImplementedError))

    def test_a_bare_result_failure_carries_its_e_code(self):
        """Every fuse::Result failure (never thrown in C++) is a bare pfuse.FuseError."""
        with pytest.raises(pfuse.FuseError) as excinfo:
            pfuse.from_bytes(b"not a fuse descriptor")
        assert excinfo.value.code == "E-DESC-MAGIC"
        assert excinfo.value.where is None  # no C++ exception crossed; nothing to report


class TestPyComponentBase:
    def test_constructor_records_the_name(self):
        class Surrogate(pfuse.Component):
            def __init__(self, n):
                super().__init__("surrogate")
                self.data = np.zeros(n)
                self.steps = 0

            def register_fields(self, registry, graph):
                cid = graph.find_component("surrogate")
                for fid in graph.component(cid).fields:
                    registry.bind_numpy(fid, self.data)

            def advance(self, ctx):
                self.steps += 1
                self.data[:] = float(self.steps)

        c = Surrogate(8)
        assert c.name == "surrogate"
        assert c.steps == 0

    def test_an_empty_name_is_refused(self):
        with pytest.raises(pfuse.InvalidArgument):
            pfuse.Component("")

    def test_the_base_class_refuses_to_pretend(self):
        """An unoverridden advance()/register_fields() must fail loudly, not silently no-op."""
        c = pfuse.Component("bare")
        with pytest.raises(TypeError):
            c.advance(None)
        with pytest.raises(TypeError):
            c.register_fields(None, None)


# ---------------------------------------------------------------------------------------
# New relative to upstream: field-by-field coverage of NewtonConfig/AccelConfig, extracted
# by grep from include/fuse/model/iteration.hpp for docs/dev/python-surface.md's evidence
# table, not retyped from memory. A binding that silently drops a struct field fails here.
# ---------------------------------------------------------------------------------------

NEWTON_CONFIG_FIELDS = [
    "line_search_kind", "forcing_choice", "forcing_gamma", "forcing_alpha", "eta_initial",
    "eta_max", "eta_min", "forcing_safeguard_threshold", "forcing_eta_constant", "atol",
    "rtol", "stol", "divergence_tolerance", "max_nonlinear_iterations",
    "max_function_evaluations", "armijo_alpha", "lambda_min", "max_backtracks", "secant_tol",
    "max_secant_steps", "damping", "gmres_restart", "gmres_max_iterations",
    "preconditioner_lag", "use_block_jacobi_preconditioner", "substrate", "norm",
    "residual_path",
]

ACCEL_CONFIG_FIELDS = [
    "kind", "initial_relaxation", "omega", "reuse", "filter",
]


class TestNewtonAndAccelConfigCoverage:
    def test_newton_config_exposes_every_cxx_field(self):
        cfg = pfuse.NewtonConfig()
        missing = [f for f in NEWTON_CONFIG_FIELDS if not hasattr(cfg, f)]
        assert missing == [], f"NewtonConfig is missing: {missing}"

    def test_accel_config_exposes_every_cxx_field(self):
        cfg = pfuse.AccelConfig()
        missing = [f for f in ACCEL_CONFIG_FIELDS if not hasattr(cfg, f)]
        assert missing == [], f"AccelConfig is missing: {missing}"

    def test_newton_config_defaults_match_the_cxx_struct(self):
        """Defaults copied from include/fuse/model/iteration.hpp's own field initialisers."""
        cfg = pfuse.NewtonConfig()
        assert cfg.line_search_kind == pfuse.LineSearchKind.BT
        assert cfg.forcing_choice == pfuse.EisenstatWalkerChoice.Choice2
        assert cfg.atol == pytest.approx(1e-50)
        assert cfg.rtol == pytest.approx(1e-8)
        assert cfg.max_nonlinear_iterations == 50
        assert cfg.use_block_jacobi_preconditioner is False
        assert cfg.substrate == pfuse.NewtonSubstrate.Native
        assert cfg.norm == pfuse.NewtonNorm.Plain
        assert cfg.residual_path == pfuse.NewtonResidualPath.Advance
