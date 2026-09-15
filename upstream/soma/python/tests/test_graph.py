"""The Python front-end must build the same graph object as C++ and YAML.

Spec section 3.2. These are the task brief's tests, ported to ``unittest`` (see
python/CMakeLists.txt for why the suite does not depend on pytest) and
corrected where the brief did not match the library:

  * ``soma.Conservation.None`` cannot be spelled in Python -- ``None`` is a
    keyword -- so the enumerator is ``soma.Conservation.Nil``. The brief never
    names it, but ``add_field``'s default does.
  * ``soma.validate(g).has_errors`` and ``.items`` are properties, as the brief
    assumed; ``soma.lower(g).diagnostics`` is spelled out rather than
    ``.diags``.
  * The brief's ``test_python_component_is_a_peer`` only constructed a
    component and asserted ``c.name == "surrogate"`` and ``c.steps == 0``. That
    passes against a binding in which nothing is wired up at all, so it is
    replaced here by a driven run (test_runtime.py) that transfers values
    between two Python peers and checks the numbers. The constructor checks are
    kept, because they are still true, but they are not the coverage.
"""

import os
import unittest
from fractions import Fraction

import numpy as np

import soma


DATA_DIR = os.environ.get("SOMA_TEST_DATA_DIR")


def build_pair() -> soma.CouplingGraph:
    """The brief's two-component conjugate-heat-transfer graph."""
    g = soma.CouplingGraph()
    m1 = g.add_mesh("fluid_mesh", "generated:box:8x8x8")
    m2 = g.add_mesh("solid_mesh", "generated:box:5x5x5")
    r1 = g.add_region("fluid_wall", m1, entity_set="wall", dimension=2)
    r2 = g.add_region("solid_wall", m2, entity_set="wall", dimension=2)

    fluid = g.add_component("fluid", first_rank=0, rank_count=2,
                            interval="1/2")
    solid = g.add_component("solid", first_rank=2, rank_count=2, interval=1)

    q_out = g.add_field(fluid, "heat_flux", r1, units=soma.units.W_per_m2,
                        conservation=soma.Conservation.Extensive,
                        intent=soma.Intent.Provides)
    q_in = g.add_field(solid, "q_wall", r2, units=soma.units.W_per_m2,
                       conservation=soma.Conservation.Extensive,
                       intent=soma.Intent.Requires)

    ch = g.add_channel("flux", q_out, q_in)
    g.channel(ch).accumulate(soma.AccumOp.Mean).remap(
        soma.RemapFlags.Conservative | soma.RemapFlags.Monotone).relax(0.7)
    return g


class BuildsAGraph(unittest.TestCase):
    def test_counts(self):
        g = build_pair()
        self.assertEqual(g.component_count, 2)
        self.assertEqual(g.channel_count, 1)
        self.assertEqual(g.field_count, 2)
        self.assertEqual(g.mesh_count, 2)
        self.assertEqual(g.region_count, 2)

    def test_channel_builder_recorded_what_it_was_told(self):
        g = build_pair()
        ch = g.channel_at(soma.ChannelId(0))
        self.assertEqual(ch.name, "flux")
        self.assertEqual(ch.accum_op, soma.AccumOp.Mean)
        self.assertTrue(ch.relax_enabled)
        self.assertAlmostEqual(ch.relax_omega, 0.7)
        # RemapFlags is arithmetic, so `|` composes and `&` tests.
        self.assertTrue(ch.remap_flags & soma.RemapFlags.Conservative)
        self.assertTrue(ch.remap_flags & soma.RemapFlags.Monotone)
        self.assertFalse(ch.remap_flags & soma.RemapFlags.HighOrder)

    def test_field_metadata_survives(self):
        g = build_pair()
        f = g.field(soma.FieldId(0))
        self.assertEqual(f.name, "heat_flux")
        self.assertEqual(f.components, 1)
        self.assertEqual(f.units, soma.units.W_per_m2)
        self.assertEqual(f.intent, soma.Intent.Provides)
        self.assertEqual(g.owner_of(soma.FieldId(0)),
                         g.find_component("fluid"))

    def test_find_component_raises_rather_than_returning_a_sentinel(self):
        g = build_pair()
        with self.assertRaises(soma.NotFound):
            g.find_component("no_such_component")
        # ... and NotFound is a LookupError, so idiomatic Python still works.
        with self.assertRaises(LookupError):
            g.find_component("no_such_component")


class IntervalCoercion(unittest.TestCase):
    def test_accepts_string_fraction_and_int(self):
        g = soma.CouplingGraph()
        a = g.add_component("a", interval="1/3")
        b = g.add_component("b", interval=Fraction(1, 7))
        c = g.add_component("c", interval=2)
        d = g.add_component("d", interval=soma.Rational(5, 10))
        self.assertEqual(g.component(a).interval, soma.Rational(1, 3))
        self.assertEqual(g.component(b).interval, soma.Rational(1, 7))
        self.assertEqual(g.component(c).interval, soma.Rational(2))
        # Rational reduces on construction, so 5/10 is 1/2.
        self.assertEqual(g.component(d).interval, soma.Rational(1, 2))

    def test_rejects_a_float(self):
        """Spec 4.3: a double end time makes 'is the run over?' rounding-dependent."""
        g = soma.CouplingGraph()
        with self.assertRaises(soma.InvalidArgument) as cm:
            g.add_component("bad", interval=0.5)
        self.assertIn("float", str(cm.exception))
        # Even though 0.5 IS exactly representable: the rule is about the type,
        # not the value, because 1/3 is not and the caller cannot tell.
        with self.assertRaises(ValueError):
            g.add_component("bad2", interval=1.0)

    def test_rejects_a_zero_denominator(self):
        with self.assertRaises(Exception):
            soma.Rational(1, 0)

    def test_rational_helper_matches_the_keyword(self):
        self.assertEqual(soma.rational("1/3"), soma.Rational(1, 3))
        self.assertEqual(soma.rational(Fraction(2, 6)), soma.Rational(1, 3))
        self.assertEqual(soma.rational(4), soma.Rational(4, 1))
        self.assertEqual(float(soma.Rational(1, 4)), 0.25)


class YamlRoundTrip(unittest.TestCase):
    def test_round_trip(self):
        g = build_pair()
        text = g.to_yaml()
        restored = soma.from_yaml(text)
        self.assertTrue(soma.structurally_equal(g, restored))

    def test_descriptor_round_trip(self):
        """Spec 3.2: the descriptor is the ABI boundary, and it is symmetric."""
        g = build_pair()
        blob = g.to_descriptor()
        self.assertIsInstance(blob, bytes)
        self.assertGreater(len(blob), 0)
        self.assertTrue(soma.structurally_equal(g, soma.from_descriptor(blob)))

    def test_yaml_and_python_produce_the_same_descriptor(self):
        """The claim in spec 3.2, tested against a YAML file C++ also reads."""
        if not DATA_DIR:
            self.skipTest("SOMA_TEST_DATA_DIR is not set")
        from_file = soma.load_yaml(os.path.join(DATA_DIR, "pair.yaml"))

        g = soma.CouplingGraph()
        m1 = g.add_mesh("fluid_mesh", "generated:box:8x8x8")
        m2 = g.add_mesh("solid_mesh", "generated:box:5x5x5")
        r1 = g.add_region("fluid_wall", m1, entity_set="wall", dimension=2)
        r2 = g.add_region("solid_wall", m2, entity_set="wall", dimension=2)
        fluid = g.add_component("fluid", first_rank=0, rank_count=2,
                                interval="1/2")
        solid = g.add_component("solid", first_rank=2, rank_count=2,
                                interval=1)
        src = g.add_field(fluid, "heat_flux", r1, units=soma.units.W_per_m2,
                          conservation=soma.Conservation.Extensive,
                          intent=soma.Intent.Provides)
        dst = g.add_field(solid, "q_wall", r2, units=soma.units.W_per_m2,
                          conservation=soma.Conservation.Extensive,
                          intent=soma.Intent.Requires)
        ch = g.add_channel("flux", src, dst)
        (g.channel(ch)
          .accumulate(soma.AccumOp.Mean, soma.Window.from_component(fluid))
          .remap(soma.RemapFlags.Conservative | soma.RemapFlags.Monotone)
          .remap_method("mono3")
          .spacetime(soma.TemporalBasis.DG, 2,
                     soma.Window.explicit_length("2/5"))
          .relax(0.7))

        self.assertEqual(g.to_descriptor(), from_file.to_descriptor(),
                         "a Python-authored graph must serialize to the same "
                         "bytes as the YAML front-end's")
        self.assertTrue(soma.structurally_equal(g, from_file))

    def test_malformed_yaml_raises_rather_than_returning_a_flag(self):
        with self.assertRaises(soma.InvalidArgument):
            soma.from_yaml("this: is: not: a: graph")
        with self.assertRaises(soma.InvalidArgument):
            soma.load_yaml("/definitely/not/here.yaml")


class Validation(unittest.TestCase):
    def test_reports_unit_mismatch(self):
        g = soma.CouplingGraph()
        m = g.add_mesh("m", "generated:box:2x2x2")
        r = g.add_region("r", m, dimension=3)
        a = g.add_component("a", first_rank=0, rank_count=1)
        b = g.add_component("b", first_rank=1, rank_count=1)
        s = g.add_field(a, "T", r, units=soma.units.K,
                        intent=soma.Intent.Provides)
        t = g.add_field(b, "q", r, units=soma.units.W_per_m2,
                        intent=soma.Intent.Requires)
        g.add_channel("bad", s, t)

        diags = soma.validate(g)
        self.assertTrue(diags.has_errors)
        self.assertTrue(any(d.code == "E-UNITS" for d in diags.items))
        # Global Constraints: every problem, together. Iteration and len() both
        # see the same list.
        self.assertEqual(len(list(diags)), len(diags))
        self.assertIn("E-UNITS", diags.to_string())

    def test_a_clean_graph_has_no_errors(self):
        diags = soma.validate(build_pair())
        self.assertFalse(diags.has_errors)


class Lowering(unittest.TestCase):
    def test_produces_an_ir_dump(self):
        g = build_pair()
        result = soma.lower(g)
        self.assertTrue(result.ok, result.diagnostics.to_string())
        text = soma.dump_ir(result.dag)
        self.assertTrue(text.startswith("# soma-ir v1"), text[:80])
        self.assertIn("Accumulate", text)
        self.assertIn("Remap", text)
        self.assertGreater(result.dag.node_count, 0)

    def test_graphviz_dump_is_a_digraph(self):
        result = soma.lower(build_pair())
        self.assertTrue(soma.dump_graphviz(result.dag).startswith(
            "digraph soma_ir"))

    def test_schedule_and_plan_dump(self):
        g = build_pair()
        dag, plan, _backend = soma.build_plan(g)
        self.assertGreater(dag.node_count, 0)
        self.assertGreater(plan.instruction_count, 0)
        # fluid runs at 1/2, solid at 1: the common period is 1.
        self.assertEqual(plan.period, soma.Rational(1))
        self.assertFalse(plan.lockstep)
        self.assertEqual(soma.dump_plan(plan), plan.to_string())

    def test_build_plan_reports_every_problem_at_once(self):
        g = soma.CouplingGraph()
        m = g.add_mesh("m", "generated:box:2x2x2")
        r = g.add_region("r", m, dimension=3)
        a = g.add_component("a")
        b = g.add_component("b", first_rank=1)
        s = g.add_field(a, "T", r, units=soma.units.K,
                        intent=soma.Intent.Provides)
        t = g.add_field(b, "q", r, units=soma.units.W_per_m2,
                        intent=soma.Intent.Requires)
        g.add_channel("bad", s, t)
        with self.assertRaises(soma.ValidationError) as cm:
            soma.build_plan(g)
        self.assertTrue(cm.exception.diagnostics.has_errors)
        self.assertTrue(
            any(d.code == "E-UNITS" for d in cm.exception.diagnostics.items))
        # ValidationError is an InvalidArgument is a ValueError.
        self.assertIsInstance(cm.exception, soma.InvalidArgument)
        self.assertIsInstance(cm.exception, ValueError)


class ExceptionHierarchy(unittest.TestCase):
    """Python callers get exceptions, never error codes."""

    def test_every_soma_error_is_a_soma_error(self):
        for cls in (soma.InvalidArgument, soma.NotFound, soma.StateError,
                    soma.SomaRuntimeError, soma.Unsupported,
                    soma.ValidationError):
            self.assertTrue(issubclass(cls, soma.SomaError), cls)

    def test_each_also_derives_from_the_closest_builtin(self):
        self.assertTrue(issubclass(soma.InvalidArgument, ValueError))
        self.assertTrue(issubclass(soma.NotFound, LookupError))
        self.assertTrue(issubclass(soma.StateError, RuntimeError))
        self.assertTrue(issubclass(soma.SomaRuntimeError, RuntimeError))
        self.assertTrue(issubclass(soma.Unsupported, NotImplementedError))

    def test_version_strings(self):
        self.assertRegex(soma.version(), r"^\d+\.\d+\.\d+")
        self.assertEqual(soma.abi_version(), 1)


class PyComponentBase(unittest.TestCase):
    def test_constructor_records_the_name(self):
        class Surrogate(soma.PyComponent):
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
        self.assertEqual(c.name, "surrogate")
        self.assertEqual(c.steps, 0)

    def test_an_empty_name_is_refused(self):
        with self.assertRaises(soma.InvalidArgument):
            soma.PyComponent("")

    def test_the_base_class_refuses_to_pretend(self):
        """An unoverridden advance() must fail loudly, not silently no-op."""
        c = soma.PyComponent("bare")
        with self.assertRaises(TypeError):
            c.advance(None)
        with self.assertRaises(TypeError):
            c.register_fields(None, None)


if __name__ == "__main__":
    unittest.main()
